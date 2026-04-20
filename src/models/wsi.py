import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data

class ExplicitConceptQuantifier(nn.Module):
    """
    Stream A: Explicit Pathologic Concept Quantifier (Continuous Mode)
    Quantifies pathologic patterns by computing cosine similarity between
    high-dimensional patch features and expert-defined anchors.
    """
    def __init__(
        self,
        input_dim=1536,
        num_concepts=8,
        init_anchors=None,
        trainable_anchors=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts

        if init_anchors is None:
            raise ValueError("The active WSI concept stream requires expert-initialized anchors.")
        if init_anchors.shape[1] != input_dim:
            raise ValueError(
                f"WSI anchor dimension mismatch: got {tuple(init_anchors.shape)}, expected (*, {input_dim})."
            )
        self.anchors = nn.Parameter(init_anchors.clone(), requires_grad=trainable_anchors)

        self.register_buffer('anchor_ema', self.anchors.data.clone())

    @property
    def output_dim(self):
        return self.num_concepts

    def compute_patch_scores(self, x):
        """
        Args:
            x: (B, N, D) or (N, D) - Patch Features
        Returns:
            scores: (B, N, K) - Patch-level concept similarity scores
        """
        if x.dim() == 2:
            x = x.unsqueeze(0) # (1, N, D)

        # 1. Normalize for Cosine Similarity
        x_norm = F.normalize(x, p=2, dim=-1)
        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)

        # 2. Similarity Map (B, N, K)
        return torch.matmul(x_norm, anchors_norm.t())

    def forward(self, x):
        """
        Args:
            x: (B, N, D) or (N, D) - Patch Features
        Returns:
            concept_profile: (B, K) - Patient-level concept ratios
        """
        scores = self.compute_patch_scores(x)

        # Patient-level concept ratio profiling.
        concept_profile = scores.mean(dim=1) # (B, K)

        return concept_profile

    @torch.no_grad()
    def apply_momentum_constraint(self, momentum: float = 0.99):
        if not self.anchors.requires_grad:
            return
        m = float(momentum)
        m = max(0.0, min(0.9999, m))
        self.anchor_ema.mul_(m).add_(self.anchors.data * (1.0 - m))
        self.anchors.data.copy_(self.anchor_ema)

class SinusoidalSpatialEncoding2D(nn.Module):
    """
    2D Spatial Positional Encoding mapping (x, y) coordinates directly into the latent space.
    """
    def __init__(self, d_model, temperature=10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D PE"

    def forward(self, x, pos):
        # Supports either (N, D)/(N, 2) or batched (B, N, D)/(B, N, 2).
        added_batch_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            added_batch_dim = True
        elif x.dim() != 3 or pos.dim() != 3:
            raise ValueError(f"Unexpected shapes for spatial encoding: x={tuple(x.shape)}, pos={tuple(pos.shape)}")

        # Normalize coordinates to [0, 1] relative to each WSI bounding box.
        pos_norm = pos - pos.min(dim=1, keepdim=True)[0]
        pos_max = pos_norm.max(dim=1, keepdim=True)[0]
        pos_norm = pos_norm / (pos_max + 1e-8)

        x_coord = pos_norm[..., 0]
        y_coord = pos_norm[..., 1]

        d_half = self.d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float32, device=x.device) * (-math.log(self.temperature) / d_half)
        )

        sin_x = torch.sin(x_coord.unsqueeze(-1) * div_term)
        cos_x = torch.cos(x_coord.unsqueeze(-1) * div_term)
        sin_y = torch.sin(y_coord.unsqueeze(-1) * div_term)
        cos_y = torch.cos(y_coord.unsqueeze(-1) * div_term)

        pe_x = torch.zeros((*x.shape[:-1], d_half), device=x.device, dtype=x.dtype)
        pe_y = torch.zeros((*x.shape[:-1], d_half), device=x.device, dtype=x.dtype)
        pe_x[..., 0::2] = sin_x.to(dtype=x.dtype)
        pe_x[..., 1::2] = cos_x.to(dtype=x.dtype)
        pe_y[..., 0::2] = sin_y.to(dtype=x.dtype)
        pe_y[..., 1::2] = cos_y.to(dtype=x.dtype)

        pe = torch.cat([pe_x, pe_y], dim=-1)
        out = x + pe
        return out.squeeze(0) if added_batch_dim else out

class WSIProjector(nn.Module):
    """
    Projector for WSI Features (Stream B Input).
    """
    def __init__(self, in_features=1536, out_features=256):
        super(WSIProjector, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapter = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
        self.spatial_pe = SinusoidalSpatialEncoding2D(d_model=out_features)

    def forward(self, x, pos):
        # Now expects `pos` to inject explicit 2D topology
        features = self.adapter(x)
        features = self.spatial_pe(features, pos)
        return features

def simple_radius_graph(x, r, loop=False, max_num_neighbors=32):
    dist = torch.cdist(x, x)
    mask = dist < r
    if not loop: mask.fill_diagonal_(False)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)

class GraphNeuralNetworkWithSpatialInfo(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.1):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.out_channels = out_channels

    def forward(self, x, edge_index, batch):
        residual = self.residual_proj(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)

class MILModelWithPositionalEncoding(nn.Module):
    def __init__(
        self,
        feature_extractor,
        gnn,
        dropout=.1,
        num_anchors=8,
        init_anchors=None,
        trainable_anchors=True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor # WSIProjector (Stream B)
        self.gnn = gnn # GNN (Stream B)
        
        # --- Stream A: Explicit Concept Quantifier ---
        raw_dim = feature_extractor.in_features
        self.concept_quantifier = ExplicitConceptQuantifier(
            input_dim=raw_dim,
            num_concepts=num_anchors,
            init_anchors=init_anchors,
            trainable_anchors=trainable_anchors,
        )
        self.concept_dim = self.concept_quantifier.output_dim
        self.dropout = nn.Dropout(p=dropout)

    def _ensure_patch_feature_list(self, images, positions):
        if isinstance(images, list):
            return [self.feature_extractor(img, pos) for img, pos in zip(images, positions)]

        features = self.feature_extractor(images, positions)
        if features.dim() == 3:
            return [features[i] for i in range(features.size(0))]
        return [features]

    def _build_graph_batch(self, patch_features, positions):
        if not patch_features:
            raise ValueError("patch_features must not be empty")

        device = patch_features[0].device
        data_list = []
        for feat, pos in zip(patch_features, positions):
            n_nodes = pos.size(0)
            if n_nodes > 1:
                edge_index = simple_radius_graph(pos, r=400.0, loop=False)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=feat.device)
            data_list.append(torch_geometric.data.Data(x=feat, edge_index=edge_index, pos=pos))
        return torch_geometric.data.Batch.from_data_list(data_list).to(device)

    def encode_concepts(self, concept_inputs):
        if concept_inputs is None:
            raise ValueError("concept_inputs must not be None")

        concept_profiles = []
        if isinstance(concept_inputs, list):
            for img in concept_inputs:
                cp = self.concept_quantifier(img)
                concept_profiles.append(cp)
            return torch.cat(concept_profiles, dim=0)
        else:
            return self.concept_quantifier(concept_inputs)

    def encode_patch_concepts(self, concept_inputs):
        if concept_inputs is None:
            raise ValueError("concept_inputs must not be None")

        if isinstance(concept_inputs, list):
            return [
                self.concept_quantifier.compute_patch_scores(img).squeeze(0)
                for img in concept_inputs
            ]

        scores = self.concept_quantifier.compute_patch_scores(concept_inputs)
        if scores.dim() == 3:
            return [scores[i] for i in range(scores.size(0))]
        return [scores]

    def encode_patch_latent(self, images, positions):
        patch_features = self._ensure_patch_feature_list(images, positions)
        batch_data = self._build_graph_batch(patch_features, positions)
        refined = self.gnn(batch_data.x, batch_data.edge_index, batch_data.batch)
        counts = torch.bincount(batch_data.batch, minlength=len(patch_features)).tolist()
        return list(torch.split(refined, counts, dim=0))

    def encode_latent(self, images, positions):
        patch_latents = self.encode_patch_latent(images, positions)
        return torch.stack([feat.mean(dim=0) for feat in patch_latents], dim=0)

    def forward(self, images, positions, concept_inputs=None):
        # images: (B, N, 1536) features
        if concept_inputs is None:
            concept_inputs = images
        concept_profiles = self.encode_concepts(concept_inputs)
        latent_spatial = self.encode_latent(images, positions)
        return torch.cat([latent_spatial, concept_profiles], dim=1)

    @torch.no_grad()
    def apply_anchor_momentum(self, momentum: float = 0.99):
        self.concept_quantifier.apply_momentum_constraint(momentum=momentum)
