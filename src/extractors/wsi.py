from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import imagecodecs
except ImportError:
    imagecodecs = None

from src.utils.feature_contracts import (
    CANONICAL_WSI_INPUT_SIZE,
    CANONICAL_WSI_NORMALIZE_MEAN,
    CANONICAL_WSI_NORMALIZE_STD,
    CANONICAL_WSI_TRUNCATED_BLOCKS,
)


NUM_LAYERS_TO_SKIP = CANONICAL_WSI_TRUNCATED_BLOCKS


def decode_wsi_jxl(img_bytes):
    if imagecodecs is None:
        raise ImportError("imagecodecs is required for JPEG XL decoding.")
    if isinstance(img_bytes, np.void):
        img_bytes = img_bytes.tobytes()
    elif isinstance(img_bytes, np.ndarray):
        img_bytes = img_bytes.tobytes()
    return imagecodecs.jpegxl_decode(img_bytes)


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def numpy_images_to_tensor_batch(images_np):
    tensors = []
    for image_np in images_np:
        image_arr = np.asarray(image_np)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image shaped (H,W,3), got {tuple(image_arr.shape)}")
        tensors.append(torch.from_numpy(image_arr).float().permute(2, 0, 1) / 255.0)
    if not tensors:
        raise ValueError("images_np must not be empty")
    return torch.stack(tensors, dim=0)


def normalize_wsi_patch_tensor_batch(batch: torch.Tensor) -> torch.Tensor:
    if batch.dim() != 4 or batch.size(1) != 3:
        raise ValueError(f"Expected patch batch shaped (N,3,H,W), got {tuple(batch.shape)}")
    batch = batch.float()
    if tuple(int(v) for v in batch.shape[-2:]) != CANONICAL_WSI_INPUT_SIZE:
        batch = F.interpolate(
            batch,
            size=CANONICAL_WSI_INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        )
    mean = batch.new_tensor(CANONICAL_WSI_NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(CANONICAL_WSI_NORMALIZE_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


def load_gigapath_backbone(model_name, layers_to_skip=NUM_LAYERS_TO_SKIP):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for WSI feature extraction.") from exc

    model_kwargs = {"num_classes": 0}
    if os.path.isdir(model_name):
        config_path = os.path.join(model_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                conf = json.load(handle)
            model_args = conf.get("model_args", {})
            if "patch_size" in model_args:
                model_kwargs["patch_size"] = model_args["patch_size"]
            if "img_size" in model_args:
                model_kwargs["img_size"] = model_args["img_size"]

    model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **model_kwargs)

    if os.path.isdir(model_name):
        weight_path = os.path.join(model_name, "pytorch_model.bin")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Could not find WSI weights at {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(model_name):
        state_dict = torch.load(model_name, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = timm.create_model(f"hf_hub:{model_name}", pretrained=True, num_classes=0)

    if layers_to_skip > 0:
        if hasattr(model, "blocks"):
            model.blocks = model.blocks[:-layers_to_skip]
            model.norm = nn.Identity()
            model.head = nn.Identity()
        elif hasattr(model, "layers"):
            model.layers = model.layers[:-layers_to_skip]

    return model


class GigapathFeatureEncoder(nn.Module):
    def __init__(self, model_name, device=None, layers_to_skip=NUM_LAYERS_TO_SKIP):
        super().__init__()
        self.device = torch.device(device if device else get_optimal_device())
        self.layers_to_skip = int(layers_to_skip)
        self.model = load_gigapath_backbone(model_name, layers_to_skip=self.layers_to_skip)
        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                pass

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = normalize_wsi_patch_tensor_batch(batch).to(self.device, non_blocking=(self.device.type == "cuda"))

        use_amp = self.device.type in {"cuda", "mps"}
        dtype = torch.float32
        if self.device.type == "mps":
            dtype = torch.float16
        elif self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            if use_amp:
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    feats = self.model(batch)
            else:
                feats = self.model(batch)
        except Exception:
            feats = self.model(batch)

        if torch.isnan(feats).any():
            raise RuntimeError("WSI feature encoder produced NaNs.")
        return feats.float()


class WSIFeatureExtractor:
    """Thin numpy-facing wrapper used by H5 feature-cache extraction."""

    def __init__(self, model_name, device=None):
        self.encoder = GigapathFeatureEncoder(model_name=model_name, device=device)

    def process_tensor_batch(self, batch):
        return self.encoder(batch).cpu().numpy()

    def process_batch(self, images_np):
        batch = numpy_images_to_tensor_batch(images_np)
        return self.process_tensor_batch(batch)
