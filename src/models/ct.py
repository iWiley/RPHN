import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CT_ROI_NAMES = (
    "liver",
    "liver_lesion_or_tumor",
    "liver_peritumoral",
    "liver_vessels",
)


def get_norm_layer(num_features, num_groups=32):
    if num_features % num_groups != 0:
        for g in [16, 8, 4, 2, 1]:
            if num_features % g == 0:
                num_groups = g
                break
    return nn.GroupNorm(num_groups, num_features)


def get_attention_heads(dim):
    for heads in (8, 4, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.project(x)


class ResidualLatentRefiner(nn.Module):
    def __init__(self, dim, depth=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(max(0, int(depth)))
            ]
        )
        self.norm = nn.LayerNorm(dim) if self.blocks else nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class LearnedAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens), dim=1)
        return (weights * tokens).sum(dim=1)


class TokenMixerBlock(nn.Module):
    def __init__(self, dim, num_heads=None, mlp_ratio=2.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads or get_attention_heads(self.dim))
        hidden_dim = int(round(self.dim * float(mlp_ratio)))

        self.attn_norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, tokens, key_padding_mask=None, token_keep=None):
        attn_inputs = self.attn_norm(tokens)
        attn_out, _ = self.attn(
            attn_inputs,
            attn_inputs,
            attn_inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        if token_keep is not None:
            tokens = tokens * token_keep
        return tokens


class MultiROIExplicitEncoder(nn.Module):
    def __init__(self, feature_dim, target_dim, roi_names=None, mixer_depth=2):
        super().__init__()
        self.roi_names = tuple(roi_names or DEFAULT_CT_ROI_NAMES)
        self.target_dim = int(target_dim)
        self.num_rois = len(self.roi_names)
        self.projector = FeatureProjector(feature_dim, target_dim)
        self.global_token_embed = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.roi_token_embed = nn.Parameter(torch.zeros(1, self.num_rois, self.target_dim))
        self.coverage_embed = nn.Sequential(
            nn.Linear(1, self.target_dim),
            nn.GELU(),
            nn.Linear(self.target_dim, self.target_dim),
        )
        self.token_mixer = nn.ModuleList(
            [TokenMixerBlock(self.target_dim) for _ in range(max(1, int(mixer_depth)))]
        )
        self.output_norm = nn.LayerNorm(self.target_dim)
        self.eps = 1e-6
        self.last_roi_stats = {}

        nn.init.normal_(self.global_token_embed, std=0.02)
        nn.init.normal_(self.roi_token_embed, std=0.02)

    def _to_spatial_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return x
        raise ValueError(
            f"CT ROI encoding requires spatial feature maps shaped as (B,C,D,H,W), got {tuple(x.shape)}."
        )

    def _prepare_masks(
        self,
        roi_masks: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if roi_masks is None:
            raise ValueError("CT ROI masks are required for the explicit CT stream.")
        if roi_masks.dim() == 4:
            roi_masks = roi_masks.unsqueeze(0)
        if roi_masks.dim() != 5:
            raise ValueError(f"CT ROI masks must be 5D (B,R,D,H,W), got shape={tuple(roi_masks.shape)}")
        if roi_masks.size(1) != len(self.roi_names):
            raise ValueError(
                f"CT ROI mask channel mismatch: expected {len(self.roi_names)} channels "
                f"({self.roi_names}), got {roi_masks.size(1)}."
            )

        roi_masks = roi_masks.to(device=device, dtype=dtype)
        if tuple(int(v) for v in roi_masks.shape[-3:]) != tuple(int(v) for v in spatial_shape):
            current_shape = tuple(int(v) for v in roi_masks.shape[-3:])
            shrink_shape = tuple(min(src, dst) for src, dst in zip(current_shape, spatial_shape))
            if shrink_shape != current_shape:
                roi_masks = F.adaptive_avg_pool3d(roi_masks, output_size=shrink_shape)
            if shrink_shape != spatial_shape:
                roi_masks = F.interpolate(
                    roi_masks,
                    size=spatial_shape,
                    mode="trilinear",
                    align_corners=False,
                )
        return roi_masks.clamp_(0.0, 1.0)

    def forward(self, x: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
        feature_map = self._to_spatial_feature_map(x)
        roi_feature_map = self.projector(feature_map)
        roi_masks = self._prepare_masks(
            roi_masks,
            spatial_shape=tuple(int(v) for v in roi_feature_map.shape[-3:]),
            device=roi_feature_map.device,
            dtype=roi_feature_map.dtype,
        )

        flat_feat = roi_feature_map.flatten(2)
        flat_masks = roi_masks.flatten(2)
        mask_mass = flat_masks.sum(dim=-1, keepdim=True)
        roi_tokens = torch.einsum("brn,bcn->brc", flat_masks, flat_feat)
        roi_tokens = roi_tokens / mask_mass.clamp_min(self.eps)
        roi_presence = (mask_mass > self.eps).to(dtype=roi_feature_map.dtype).squeeze(-1)
        roi_tokens = roi_tokens * roi_presence.unsqueeze(-1)
        roi_coverage = flat_masks.mean(dim=-1)

        global_token = flat_feat.mean(dim=-1).unsqueeze(1) + self.global_token_embed
        coverage_tokens = self.coverage_embed(roi_coverage.unsqueeze(-1))
        roi_tokens = (
            roi_tokens
            + self.roi_token_embed * roi_presence.unsqueeze(-1)
            + coverage_tokens * roi_presence.unsqueeze(-1)
        )

        tokens = torch.cat([global_token, roi_tokens], dim=1)
        key_padding_mask = torch.cat(
            [
                torch.zeros(tokens.size(0), 1, dtype=torch.bool, device=tokens.device),
                roi_presence <= 0,
            ],
            dim=1,
        )
        token_keep = (~key_padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        for block in self.token_mixer:
            tokens = block(tokens, key_padding_mask=key_padding_mask, token_keep=token_keep)

        roi_vec = self.output_norm(tokens[:, 0])

        self.last_roi_stats = {
            "roi_names": list(self.roi_names),
            "mean_roi_coverage": {
                name: float(roi_coverage[:, idx].mean().detach().cpu())
                for idx, name in enumerate(self.roi_names)
            },
            "mean_roi_presence": {
                name: float(roi_presence[:, idx].mean().detach().cpu())
                for idx, name in enumerate(self.roi_names)
            },
        }
        return roi_vec


class CTDualStreamEncoder(nn.Module):
    """
    CT dual-stream encoder with:
    1. latent global stream from the CT feature map
    2. explicit multi-ROI stream from ROI-token interaction
    """

    def __init__(
        self,
        feature_dim=384,
        target_dim=128,
        roi_names=None,
        latent_pooling="attention",
        latent_refine_depth=1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)
        self.roi_names = tuple(roi_names or DEFAULT_CT_ROI_NAMES)
        self.latent_pooling = str(latent_pooling or "mean").strip().lower()
        if self.latent_pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported CT latent_pooling: {self.latent_pooling}")

        self.latent_projector = FeatureProjector(in_channels=self.feature_dim, out_channels=self.target_dim)
        self.spatial_attention_pool = (
            LearnedAttentionPool(self.target_dim) if self.latent_pooling == "attention" else None
        )
        self.latent_refiner = ResidualLatentRefiner(self.target_dim, depth=latent_refine_depth)

        self.explicit_encoder = MultiROIExplicitEncoder(
            feature_dim=self.feature_dim,
            target_dim=self.target_dim,
            roi_names=self.roi_names,
        )
        self.last_roi_stats = {}

    def encode_explicit(self, x: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
        roi_vec = self.explicit_encoder(x, roi_masks)
        self.last_roi_stats = self.explicit_encoder.last_roi_stats
        return roi_vec

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"CT latent encoding requires spatial feature maps shaped as (B,C,D,H,W), got {tuple(x.shape)}."
            )
        latent_map = self.latent_projector(x)
        if self.latent_pooling == "attention":
            latent_tokens = latent_map.flatten(2).transpose(1, 2)
            latent_vec = self.spatial_attention_pool(latent_tokens)
        else:
            latent_vec = latent_map.mean(dim=(2, 3, 4))
        return self.latent_refiner(latent_vec)

    def forward(self, x: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
        explicit_vec = self.encode_explicit(x, roi_masks)
        latent_vec = self.encode_latent(x)
        return torch.cat([latent_vec, explicit_vec], dim=1)
