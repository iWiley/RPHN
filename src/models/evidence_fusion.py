import torch
import torch.nn as nn


class WSIPatchAttentionPool(nn.Module):
    """
    Summarizes WSI patch tokens into a patient-level vector without CT conditioning.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.token_norm = nn.LayerNorm(self.dim)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.score = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, 1),
        )
        self.output_norm = nn.LayerNorm(self.dim)

    def forward(self, wsi_patch_tokens):
        pooled_tokens = []
        attention_weights = []

        for patch_tokens in wsi_patch_tokens:
            if patch_tokens.dim() != 2:
                raise ValueError(
                    f"WSIPatchAttentionPool expects patch tokens shaped (N, D), got {tuple(patch_tokens.shape)}"
                )

            tokens = self.token_norm(patch_tokens)
            attn_logits = self.score(tokens).transpose(0, 1)
            attn = torch.softmax(attn_logits, dim=-1)
            values = self.value_proj(patch_tokens)
            pooled = torch.matmul(attn, values)

            pooled_tokens.append(self.output_norm(pooled.squeeze(0)))
            attention_weights.append(attn.squeeze(0))

        return torch.stack(pooled_tokens, dim=0), attention_weights


class ParallelEvidenceFusion(nn.Module):
    """
    Parallel multimodal fusion:
    CT and WSI are summarized independently, then fused as peer patient-level evidence.
    """

    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.wsi_pool = WSIPatchAttentionPool(self.input_dim)

        self.ct_context_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
        )
        self.wsi_context_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
        )

        self.risk_os = nn.Sequential(
            nn.Linear(self.input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.risk_ttr = nn.Sequential(
            nn.Linear(self.input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_ct_patient, wsi_patch_tokens):
        z_ct = self.ct_context_proj(z_ct_patient)
        z_wsi_raw, attn_weights = self.wsi_pool(wsi_patch_tokens)
        z_wsi = self.wsi_context_proj(z_wsi_raw)
        z_all = torch.cat([z_ct, z_wsi], dim=1)
        risk_os = self.risk_os(z_all)
        risk_ttr = self.risk_ttr(z_all)

        z_shared_aux = 0.5 * (z_ct + z_wsi)
        z_ct_unique = z_ct - z_shared_aux
        z_wsi_unique = z_wsi - z_shared_aux

        with torch.no_grad():
            attn_mass = torch.stack([w.max() for w in attn_weights]).unsqueeze(1)

        return {
            "features": {
                "shared": z_shared_aux,
                "unique_ct": z_ct_unique,
                "unique_wsi": z_wsi_unique,
                "ct_shared": z_ct,
                "wsi_shared": z_wsi,
                "consensus": z_all,
            },
            "weights": {
                "patch_attn_peak": attn_mass,
                "wsi_patch_attention": attn_weights,
            },
            "survival_risk_os": risk_os,
            "survival_risk_ttr": risk_ttr,
        }
