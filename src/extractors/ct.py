from __future__ import annotations

from pathlib import Path

import torch

from src.utils.ctfm import DEFAULT_CTFM_FEATURE_SHAPE, ensure_canonical_ctfm_selection, unwrap_ctfm_output

try:
    from lighter_zoo import SegResEncoder
except ImportError:
    SegResEncoder = None


class CTFMFeatureEncoder(torch.nn.Module):
    def __init__(self, source: str | Path, device: torch.device | str):
        super().__init__()
        if SegResEncoder is None:
            raise ImportError("lighter_zoo is required for CT feature extraction.")
        self.backbone = SegResEncoder.from_pretrained(str(source)).eval().to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.device = torch.device(device)
        self.selected_output = "unknown"
        self.resolved_output_index = -1
        self.layer_offset_from_last = -1
        self.num_encoder_outputs = 0

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=(self.device.type == "cuda"))
        output = self.backbone(x)
        feature_map, selection = unwrap_ctfm_output(
            output,
            preferred_layer_offset_from_last=1,
            preferred_feature_shape=DEFAULT_CTFM_FEATURE_SHAPE,
        )
        ensure_canonical_ctfm_selection(selection)
        self.selected_output = selection.selected_output
        self.resolved_output_index = selection.resolved_output_index
        self.layer_offset_from_last = selection.layer_offset_from_last
        self.num_encoder_outputs = selection.num_encoder_outputs
        if feature_map.dim() == 4:
            feature_map = feature_map.unsqueeze(0)
        if feature_map.dim() != 5:
            raise ValueError(f"Expected CT-FM feature map with 5 dims, got {tuple(feature_map.shape)}")
        return feature_map
