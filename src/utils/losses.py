import torch
import torch.nn as nn

def cox_loss(risk_pred, events, times, method: str = "efron"):
    risk_pred = risk_pred.view(-1).float()
    events = events.view(-1).float()
    times = times.view(-1).float()

    valid_mask = torch.isfinite(risk_pred) & torch.isfinite(events) & torch.isfinite(times)
    risk_pred = risk_pred[valid_mask]
    events = events[valid_mask]
    times = times[valid_mask]

    if risk_pred.numel() == 0:
        return torch.tensor(0.0, device=risk_pred.device, requires_grad=True)

    order = torch.argsort(times, descending=True)
    risk_pred = risk_pred[order]
    events = events[order]
    times = times[order]

    event_mask = events > 0.5
    n_events = event_mask.sum()

    if n_events == 0:
        return torch.tensor(0.0, device=risk_pred.device, requires_grad=True)

    max_risk = torch.max(risk_pred)
    exp_risk = torch.exp(risk_pred - max_risk)

    cum_exp_risk = torch.cumsum(exp_risk, dim=0)

    loss = torch.tensor(0.0, device=risk_pred.device)

    unique_event_times = torch.unique(times[event_mask])

    for t in unique_event_times:
        tied_event_idx = (times == t) & event_mask
        d = tied_event_idx.sum()

        if d == 0:
            continue

        last_idx = torch.nonzero(times == t, as_tuple=False).max()
        risk_set_sum = cum_exp_risk[last_idx]

        tied_risk = risk_pred[tied_event_idx]
        tied_exp_sum = exp_risk[tied_event_idx].sum()

        if method.lower() == "breslow" or d == 1:
            loss = loss - (tied_risk.sum() - d.float() * torch.log(risk_set_sum + 1e-12) - d.float() * max_risk)
        elif method.lower() == "efron":
            log_denom = torch.tensor(0.0, device=risk_pred.device)
            d_float = d.float()
            for l in range(int(d.item())):
                frac = torch.tensor(float(l), device=risk_pred.device) / d_float
                denom_l = risk_set_sum - frac * tied_exp_sum
                log_denom = log_denom + torch.log(denom_l + 1e-12)
            loss = loss - (tied_risk.sum() - log_denom - d_float * max_risk)
        else:
            raise ValueError(f"Unsupported Cox ties method: {method}")

    return loss / (n_events.float() + 1e-12)

def distance_correlation_loss(z_a, z_b):
    if z_a is None or z_b is None or z_a.size(0) < 2:
        return torch.tensor(0.0, device=z_a.device if z_a is not None else z_b.device)
    
    def compute_distance_matrix(x):
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = xx + xx.t() - 2.0 * torch.mm(x, x.t())
        return torch.sqrt(torch.clamp(dist, min=1e-8))

    def double_center(dist):
        mean_row = torch.mean(dist, dim=1, keepdim=True)
        mean_col = torch.mean(dist, dim=0, keepdim=True)
        mean_all = torch.mean(dist)
        return dist - mean_row - mean_col + mean_all

    A = double_center(compute_distance_matrix(z_a))
    B = double_center(compute_distance_matrix(z_b))

    dcov_AB = torch.mean(A * B)
    dvar_A = torch.mean(A * A)
    dvar_B = torch.mean(B * B)

    dcor = dcov_AB / torch.sqrt(torch.clamp(dvar_A * dvar_B, min=1e-8))
    
    return 1.0 - dcor

class HybridSurvivalLoss(nn.Module):
    def __init__(
        self,
        cca_weight=0.1,
        intra_decor_weight=0.05,
        aux_weight=1.0,
        os_loss_weight=1.0,
        ttr_loss_weight=1.0
    ):
        super().__init__()
        self.cca_weight = cca_weight
        self.intra_decor_weight = intra_decor_weight
        self.aux_weight = aux_weight
        self.os_loss_weight = os_loss_weight
        self.ttr_loss_weight = ttr_loss_weight

    def forward(self, risk_os, risk_ttr, evt_os, tm_os, evt_ttr, tm_ttr, out_dict):
        l_os = cox_loss(risk_os, evt_os, tm_os)
        l_ttr = cox_loss(risk_ttr, evt_ttr, tm_ttr)
        
        feats = out_dict['features']
        
        z_ct_s = feats.get('ct_shared')
        z_wsi_s = feats.get('wsi_shared')

        l_cca = distance_correlation_loss(z_ct_s, z_wsi_s)
        
        l_intra = torch.tensor(0.0, device=risk_os.device)
        if 'aux_losses' in out_dict:
            l_intra = out_dict['aux_losses'].get('intra_decor_wsi', 0.0) + \
                      out_dict['aux_losses'].get('intra_decor_ct', 0.0)

        total_loss = self.os_loss_weight * l_os + self.ttr_loss_weight * l_ttr + \
                     self.aux_weight * (self.cca_weight * l_cca + \
                                      self.intra_decor_weight * l_intra)
        
        return total_loss, {
            'l_os': l_os.item(),
            'l_ttr': l_ttr.item(),
            'l_cca': l_cca.item() if torch.is_tensor(l_cca) else l_cca,
            'l_intra': l_intra.item() if torch.is_tensor(l_intra) else l_intra,
            'l_aux': (total_loss - self.os_loss_weight * l_os - self.ttr_loss_weight * l_ttr).item()
        }
