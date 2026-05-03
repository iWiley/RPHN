import torch
import torch.nn as nn

from .wsi import GraphNeuralNetworkWithSpatialInfo, MILModelWithPositionalEncoding, WSIProjector
from .ct import CTDualStreamEncoder, DEFAULT_CT_ROI_NAMES
from .evidence_fusion import ParallelEvidenceFusion

def batch_wise_correlation(z_a, z_b):
    if z_a.size(0) < 2:
        return torch.tensor(0.0, device=z_a.device)
    z_a_centered = z_a - z_a.mean(dim=0, keepdim=True)
    z_b_centered = z_b - z_b.mean(dim=0, keepdim=True)
    z_a_norm = z_a_centered / (z_a_centered.norm(dim=0, keepdim=True) + 1e-8)
    z_b_norm = z_b_centered / (z_b_centered.norm(dim=0, keepdim=True) + 1e-8)
    corr_mat = torch.mm(z_a_norm.t(), z_b_norm)
    return (corr_mat ** 2).mean()

class RPHN(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=256,
        wsi_feature_dim=1536,
        ct_feature_dim=384,
        dropout=.1,
        wsi_anchors_init=None,
        wsi_backbone=None,
        ct_backbone=None,
        ct_roi_names=None,
        ct_latent_pooling="attention",
        ct_latent_refine_depth=1,
    ):
        super().__init__()

        if wsi_backbone is None or ct_backbone is None:
            raise ValueError("RPHN requires frozen WSI and CT backbones in the active raw-input pipeline.")

        self.wsi_backbone = wsi_backbone
        self.ct_backbone = ct_backbone
        self.ct_roi_names = tuple(ct_roi_names or DEFAULT_CT_ROI_NAMES)
        
        # WSI Branch
        self.wsi_projector = WSIProjector(in_features=wsi_feature_dim, out_features=feature_dim)
        self.gnn = GraphNeuralNetworkWithSpatialInfo(
            in_channels=feature_dim,
            hidden_channels=feature_dim,
            out_channels=feature_dim,
            dropout=dropout,
        )
        self.wsi_latent_dim = feature_dim
        
        num_wsi_anchors = wsi_anchors_init.shape[0] if wsi_anchors_init is not None else 8
        self.mil_wsi = MILModelWithPositionalEncoding(
            self.wsi_projector,
            self.gnn,
            dropout=dropout,
            num_anchors=num_wsi_anchors,
            init_anchors=wsi_anchors_init,
            trainable_anchors=True,
        )
        wsi_output_dim = self.wsi_latent_dim + self.mil_wsi.concept_dim
        self.wsi_patch_feature_dim = feature_dim

        # CT Branch
        self.ct_latent_dim = 128
        self.ct_encoder = CTDualStreamEncoder(
            feature_dim=ct_feature_dim,
            target_dim=self.ct_latent_dim,
            roi_names=self.ct_roi_names,
            latent_pooling=ct_latent_pooling,
            latent_refine_depth=ct_latent_refine_depth,
        )
        ct_output_dim = self.ct_latent_dim * 2

        self.patho_proj = nn.Sequential(
            nn.Linear(self.wsi_patch_feature_dim + self.mil_wsi.concept_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.radio_proj = nn.Sequential(nn.Linear(ct_output_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.fusion = ParallelEvidenceFusion(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
        )

    @staticmethod
    def _as_sample_list(batch_obj):
        if torch.is_tensor(batch_obj):
            return [batch_obj[i] for i in range(batch_obj.size(0))]
        return list(batch_obj)

    @staticmethod
    def _aggregate_roi_stats(stats_list):
        valid_stats = [stats for stats in stats_list if stats]
        if not valid_stats:
            return {}

        roi_names = list(valid_stats[0].get("roi_names", []))
        aggregated = {
            "roi_names": roi_names,
            "mean_roi_coverage": {},
            "mean_roi_presence": {},
        }
        for key in ("mean_roi_coverage", "mean_roi_presence"):
            for roi_name in roi_names:
                values = [float(stats.get(key, {}).get(roi_name, 0.0)) for stats in valid_stats]
                aggregated[key][roi_name] = float(sum(values) / max(1, len(values)))
        return aggregated

    def _encode_ct_batch(self, batch_ct_data, batch_ct_masks):
        if torch.is_tensor(batch_ct_data) and torch.is_tensor(batch_ct_masks):
            if batch_ct_data.dim() != 5:
                raise ValueError(f"CT batches must be 5D, got {tuple(batch_ct_data.shape)}")
            if batch_ct_masks.dim() != 5:
                raise ValueError(f"CT ROI mask batches must be 5D, got {tuple(batch_ct_masks.shape)}")

            ct_model_inputs = self.ct_backbone(batch_ct_data)

            ct_explicit_vec = self.ct_encoder.encode_explicit(ct_model_inputs, batch_ct_masks)
            ct_latent_vec = self.ct_encoder.encode_latent(ct_model_inputs)
            return ct_latent_vec, ct_explicit_vec, self.ct_encoder.last_roi_stats

        ct_items = self._as_sample_list(batch_ct_data)
        mask_items = self._as_sample_list(batch_ct_masks)
        if len(ct_items) != len(mask_items):
            raise ValueError(
                f"CT batch length mismatch: {len(ct_items)} CT tensors vs {len(mask_items)} mask tensors."
            )

        latent_vecs = []
        explicit_vecs = []
        roi_stats = []
        for ct_item, mask_item in zip(ct_items, mask_items):
            if ct_item.dim() != 4:
                raise ValueError(f"Per-sample CT tensors must be 4D (C,D,H,W), got {tuple(ct_item.shape)}")
            if mask_item.dim() != 4:
                raise ValueError(f"Per-sample CT masks must be 4D (R,D,H,W), got {tuple(mask_item.shape)}")

            ct_feat = self.ct_backbone(ct_item.unsqueeze(0))

            mask_batch = mask_item.unsqueeze(0)
            explicit_vec = self.ct_encoder.encode_explicit(ct_feat, mask_batch)
            latent_vec = self.ct_encoder.encode_latent(ct_feat)
            explicit_vecs.append(explicit_vec.squeeze(0))
            latent_vecs.append(latent_vec.squeeze(0))
            roi_stats.append(self.ct_encoder.last_roi_stats)

        aggregated_roi_stats = self._aggregate_roi_stats(roi_stats)
        self.ct_encoder.last_roi_stats = aggregated_roi_stats
        return torch.stack(latent_vecs, dim=0), torch.stack(explicit_vecs, dim=0), aggregated_roi_stats

    def forward(self, batch_wsi_data, batch_wsi_position, batch_ct_data, batch_ct_masks):
        wsi_model_inputs = self._as_sample_list(batch_wsi_data)
        wsi_positions = self._as_sample_list(batch_wsi_position)
        if len(wsi_model_inputs) != len(wsi_positions):
            raise ValueError(
                f"WSI batch length mismatch: {len(wsi_model_inputs)} feature tensors vs "
                f"{len(wsi_positions)} coordinate tensors."
            )
        for wsi_item, pos_item in zip(wsi_model_inputs, wsi_positions):
            if wsi_item.dim() != 4:
                raise ValueError(f"WSI inputs must be shaped as raw patches (N,3,H,W), got {tuple(wsi_item.shape)}")
            if pos_item.dim() != 2 or pos_item.size(-1) != 2:
                raise ValueError(
                    f"WSI coordinates must be shaped (N,2), got {tuple(pos_item.shape)}"
                )
        encoded_wsi_inputs = [self.wsi_backbone(wsi_item) for wsi_item in wsi_model_inputs]

        wsi_patch_latents = self.mil_wsi.encode_patch_latent(encoded_wsi_inputs, wsi_positions)
        wsi_patch_concepts = self.mil_wsi.encode_patch_concepts(encoded_wsi_inputs)
        wsi_patch_tokens = [
            self.patho_proj(torch.cat([latent, concept], dim=1))
            for latent, concept in zip(wsi_patch_latents, wsi_patch_concepts)
        ]
        wsi_latent_patient = torch.stack([latent.mean(dim=0) for latent in wsi_patch_latents], dim=0)
        wsi_concept_patient = torch.stack([scores.mean(dim=0) for scores in wsi_patch_concepts], dim=0)

        ct_latent_vec, ct_explicit_vec, ct_roi_stats = self._encode_ct_batch(batch_ct_data, batch_ct_masks)
        ct_dual_feat = torch.cat([ct_latent_vec, ct_explicit_vec], dim=1)

        z_ct = self.radio_proj(ct_dual_feat)
        output_dict = self.fusion(z_ct, wsi_patch_tokens)
        output_dict['ct_roi_stats'] = ct_roi_stats
        
        output_dict['aux_losses'] = {
            'intra_decor_wsi': batch_wise_correlation(wsi_latent_patient, wsi_concept_patient),
            'intra_decor_ct': batch_wise_correlation(ct_latent_vec, ct_explicit_vec),
        }
        return output_dict['survival_risk_os'], output_dict['survival_risk_ttr'], output_dict

    @torch.no_grad()
    def apply_wsi_anchor_momentum(self, momentum: float = 0.99):
        if hasattr(self.mil_wsi, 'apply_anchor_momentum'):
            self.mil_wsi.apply_anchor_momentum(momentum=momentum)
