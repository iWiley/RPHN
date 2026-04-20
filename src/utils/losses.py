import torch
import torch.nn as nn

def cox_loss(risk_pred, events, times, method: str = "efron"):
    """
    Cox partial log-likelihood loss.

    Args:
        risk_pred: (B,) or (B, 1), higher means higher risk
        events:    (B,) binary event indicator (1=event, 0=censored)
        times:     (B,) observed follow-up times
        method:    "efron" (recommended) or "breslow"

    Returns:
        Scalar negative partial log-likelihood normalized by #events.
    """
    risk_pred = risk_pred.view(-1).float()
    events = events.view(-1).float()
    times = times.view(-1).float()

    # valid mask
    valid_mask = torch.isfinite(risk_pred) & torch.isfinite(events) & torch.isfinite(times)
    risk_pred = risk_pred[valid_mask]
    events = events[valid_mask]
    times = times[valid_mask]

    if risk_pred.numel() == 0:
        return torch.tensor(0.0, device=risk_pred.device, requires_grad=True)

    # sort by descending time so that risk set for index i is [0:i]
    order = torch.argsort(times, descending=True)
    risk_pred = risk_pred[order]
    events = events[order]
    times = times[order]

    event_mask = events > 0.5
    n_events = event_mask.sum()

    if n_events == 0:
        return torch.tensor(0.0, device=risk_pred.device, requires_grad=True)

    # numerical stabilization
    max_risk = torch.max(risk_pred)
    exp_risk = torch.exp(risk_pred - max_risk)

    # cumulative sum under descending time:
    # risk set sum for subject i includes all j with time_j >= time_i
    cum_exp_risk = torch.cumsum(exp_risk, dim=0)

    loss = torch.tensor(0.0, device=risk_pred.device)

    # unique event times only
    unique_event_times = torch.unique(times[event_mask])

    for t in unique_event_times:
        tied_event_idx = (times == t) & event_mask
        d = tied_event_idx.sum()

        if d == 0:
            continue

        # because sorted descending, all samples with time >= t are included
        # risk set sum can be read at the last index where time == t
        last_idx = torch.nonzero(times == t, as_tuple=False).max()
        risk_set_sum = cum_exp_risk[last_idx]

        tied_risk = risk_pred[tied_event_idx]
        tied_exp_sum = exp_risk[tied_event_idx].sum()

        if method.lower() == "breslow" or d == 1:
            loss = loss - (tied_risk.sum() - d.float() * torch.log(risk_set_sum + 1e-12) - d.float() * max_risk)
        elif method.lower() == "efron":
            # Efron approximation for ties
            # denominator: prod_{l=0}^{d-1} (R - l/d * D)
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
    """
    Distance Correlation (dCor) Loss.
    Computes non-linear alignment on pairwise distance matrices.
    Robust to B < D (Batch Size < Feature Dimension).
    """
    if z_a is None or z_b is None or z_a.size(0) < 2:
        return torch.tensor(0.0, device=z_a.device if z_a is not None else z_b.device)
    
    def compute_distance_matrix(x):
        # Pairwise Euclidean distance matrix safely
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = xx + xx.t() - 2.0 * torch.mm(x, x.t())
        # clamp to prevent NaN gradients at exactly 0 distance (diagonal)
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
    
    # Minimize correlation distance (1 - dCor)
    return 1.0 - dcor

class HybridSurvivalLoss(nn.Module):
    def __init__(
        self,
        cca_weight=0.1,
        intra_decor_weight=0.05,
        aux_weight=1.0,
        os_loss_weight=1.0,
        rfs_loss_weight=1.0
    ):
        super().__init__()
        self.cca_weight = cca_weight
        self.intra_decor_weight = intra_decor_weight
        self.aux_weight = aux_weight
        self.os_loss_weight = os_loss_weight
        self.rfs_loss_weight = rfs_loss_weight

    def forward(self, risk_os, risk_rfs, evt_os, tm_os, evt_rfs, tm_rfs, out_dict):
        # 1. Primary Survival Losses
        l_os = cox_loss(risk_os, evt_os, tm_os)
        l_rfs = cox_loss(risk_rfs, evt_rfs, tm_rfs)
        
        # 2. Auxiliary Structural Losses
        feats = out_dict['features']
        
        z_ct_s = feats.get('ct_shared')
        z_wsi_s = feats.get('wsi_shared')

        # Shared-space distance correlation alignment between CT and WSI projected shared features.
        l_cca = distance_correlation_loss(z_ct_s, z_wsi_s)
        
        # 3. Intra-modality decorrelation between concept and latent streams
        l_intra = torch.tensor(0.0, device=risk_os.device)
        if 'aux_losses' in out_dict:
            l_intra = out_dict['aux_losses'].get('intra_decor_wsi', 0.0) + \
                      out_dict['aux_losses'].get('intra_decor_ct', 0.0)

        # 4. Total Aggregation
        total_loss = self.os_loss_weight * l_os + self.rfs_loss_weight * l_rfs + \
                     self.aux_weight * (self.cca_weight * l_cca + \
                                      self.intra_decor_weight * l_intra)
        
        return total_loss, {
            'l_os': l_os.item(),
            'l_rfs': l_rfs.item(),
            'l_cca': l_cca.item() if torch.is_tensor(l_cca) else l_cca,
            'l_intra': l_intra.item() if torch.is_tensor(l_intra) else l_intra,
            'l_aux': (total_loss - self.os_loss_weight * l_os - self.rfs_loss_weight * l_rfs).item()
        }
