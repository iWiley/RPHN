from __future__ import annotations

import h5py
import imagecodecs
import numpy as np

from src.utils.feature_contracts import DEFAULT_CT_TARGET_SHAPE
from src.utils.niijxl import decode_niijxl_bytes_to_xyz


CT_FM_HU_MIN = -1024.0
CT_FM_HU_MAX = 2048.0
CT_FM_BBOX_MARGIN_MM = 10.0


def decode_h5_slice(entry) -> bytes:
    if isinstance(entry, np.ndarray):
        return np.asarray(entry, dtype=np.uint8).tobytes()
    if isinstance(entry, np.void):
        return entry.tobytes()
    return bytes(entry)


def load_internal_ct_volume_zyx(ct_group: h5py.Group) -> np.ndarray:
    if "bundle" in ct_group:
        bundle_entry = ct_group["bundle"][()]
        bundle_bytes = decode_h5_slice(bundle_entry)
        data_xyz, _, img = decode_niijxl_bytes_to_xyz(bundle_bytes)
        expected_shape_xyz = tuple(int(v) for v in ct_group.attrs.get("shape_xyz", ()))
        if expected_shape_xyz and tuple(int(v) for v in data_xyz.shape) != expected_shape_xyz:
            raise ValueError(
                f"ct/bundle decoded shape {tuple(int(v) for v in data_xyz.shape)} "
                f"does not match ct attrs shape_xyz {expected_shape_xyz}"
            )
        if "zooms" in ct_group.attrs:
            expected_zooms = tuple(float(v) for v in ct_group.attrs["zooms"][:3])
            decoded_zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
            if not np.allclose(decoded_zooms, expected_zooms, atol=1e-6):
                raise ValueError(f"ct/bundle decoded zooms {decoded_zooms} do not match ct attrs zooms {expected_zooms}")
        return np.transpose(np.asarray(data_xyz, dtype=np.float32), (2, 1, 0)).astype(np.float32, copy=False)

    if "slices" not in ct_group:
        raise KeyError("Missing ct/bundle or ct/slices")
    if "shape_xyz" not in ct_group.attrs:
        raise KeyError("Missing ct attribute 'shape_xyz'")

    slices_ds = ct_group["slices"]
    shape_xyz = tuple(int(v) for v in ct_group.attrs["shape_xyz"])
    offset = int(ct_group.attrs.get("offset_for_uint16", 32768))
    source_type = str(ct_group.attrs.get("source_type", "ct_h5"))
    slope_raw = ct_group.attrs.get("slope", np.nan)
    inter_raw = ct_group.attrs.get("inter", np.nan)
    slope = None if np.isnan(slope_raw) else float(slope_raw)
    inter = None if np.isnan(inter_raw) else float(inter_raw)

    slices = []
    for idx in range(shape_xyz[2]):
        encoded = decode_h5_slice(slices_ds[idx])
        decoded_u16 = imagecodecs.jpegxl_decode(encoded)
        decoded_i16 = decoded_u16.astype(np.int32) - offset
        slices.append(decoded_i16.astype(np.int16))
    data_xyz = np.stack(slices, axis=2).astype(np.float32, copy=False)

    if source_type == "raw_dicom_rebuilt":
        if slope is not None:
            data_xyz = data_xyz * slope
        if inter is not None:
            data_xyz = data_xyz + inter
        slope = 1.0
        inter = 0.0

    if slope is not None:
        data_xyz = data_xyz * slope
    if inter is not None:
        data_xyz = data_xyz + inter

    return np.transpose(data_xyz, (2, 1, 0)).astype(np.float32, copy=False)


def normalize_ct_for_learning(volume_zyx: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume_zyx, dtype=np.float32)
    arr = np.clip(arr, CT_FM_HU_MIN, CT_FM_HU_MAX)
    arr = (arr - CT_FM_HU_MIN) / (CT_FM_HU_MAX - CT_FM_HU_MIN)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def load_primary_mask(mask_group: h5py.Group) -> np.ndarray:
    if "liver" not in mask_group:
        raise KeyError("Missing ct/mask/liver")
    return np.asarray(mask_group["liver"][:], dtype=np.float32)


def load_ct_roi_union_mask_zyx(mask_group: h5py.Group) -> np.ndarray:
    union_mask = None
    for key in ("liver", "liver_lesion_or_tumor", "liver_peritumoral", "liver_vessels"):
        if key not in mask_group:
            continue
        current = np.asarray(mask_group[key][:], dtype=bool)
        union_mask = current if union_mask is None else (union_mask | current)
    if union_mask is None:
        raise KeyError("Missing all CT ROI masks for union bbox computation.")
    return union_mask.astype(np.float32)


def compute_mask_bbox_zyx(
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float] | None = None,
    margin_mm: float = 0.0,
) -> tuple[slice, slice, slice]:
    mask = np.asarray(mask_zyx) > 0
    if mask.ndim != 3:
        raise ValueError(f"Mask for bbox must be 3D, got shape={tuple(mask.shape)}")
    if not mask.any():
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))

    coords = np.where(mask)
    z0, y0, x0 = (int(v.min()) for v in coords)
    z1, y1, x1 = (int(v.max()) + 1 for v in coords)
    if spacing_zyx is not None and float(margin_mm) > 0.0:
        pad_zyx = []
        for spacing in spacing_zyx:
            spacing = max(float(spacing), 1e-6)
            pad_zyx.append(int(np.ceil(float(margin_mm) / spacing)))
        z0 = max(0, z0 - pad_zyx[0])
        y0 = max(0, y0 - pad_zyx[1])
        x0 = max(0, x0 - pad_zyx[2])
        z1 = min(mask.shape[0], z1 + pad_zyx[0])
        y1 = min(mask.shape[1], y1 + pad_zyx[1])
        x1 = min(mask.shape[2], x1 + pad_zyx[2])
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def crop_volume_and_mask_to_bbox_zyx(
    volume_zyx: np.ndarray,
    bbox_mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float] | None = None,
    margin_mm: float = CT_FM_BBOX_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    bbox = compute_mask_bbox_zyx(bbox_mask_zyx, spacing_zyx=spacing_zyx, margin_mm=float(margin_mm))
    cropped_volume = np.asarray(volume_zyx)[bbox[0], bbox[1], bbox[2]]
    cropped_mask = np.asarray(bbox_mask_zyx)[bbox[0], bbox[1], bbox[2]]
    return (
        np.asarray(cropped_volume, dtype=np.float32, order="C"),
        np.asarray(cropped_mask, dtype=np.float32, order="C"),
        bbox,
    )


def resize_volume_zyx(volume_zyx: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    vol_t = torch.from_numpy(np.asarray(volume_zyx, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    if tuple(int(v) for v in vol_t.shape[-3:]) != tuple(int(v) for v in target_shape):
        vol_t = F.interpolate(vol_t, size=target_shape, mode="trilinear", align_corners=False)
    return vol_t.squeeze(0)


def resize_binary_mask_zyx(mask_zyx: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    mask_t = torch.from_numpy(np.asarray(mask_zyx, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    if tuple(int(v) for v in mask_t.shape[-3:]) != tuple(int(v) for v in target_shape):
        mask_t = F.interpolate(mask_t, size=target_shape, mode="trilinear", align_corners=False)
    # A single ROI mask should stay channel-free here; preprocess_ct_masks_for_ctfm
    # stacks multiple ROI volumes into the channel axis afterwards.
    return (mask_t >= 0.5).float().squeeze(0).squeeze(0)


def preprocess_ct_masks_for_ctfm(
    ct_group: h5py.Group,
    target_shape: tuple[int, int, int],
    mask_keys: tuple[str, ...] = ("liver", "liver_lesion_or_tumor", "liver_peritumoral", "liver_vessels"),
) -> torch.Tensor:
    import torch

    if "mask" not in ct_group:
        raise KeyError("Missing ct/mask")

    mask_group = ct_group["mask"]
    bbox_mask_zyx = load_ct_roi_union_mask_zyx(mask_group)
    zooms_xyz = tuple(float(v) for v in ct_group.attrs["zooms"][:3])
    spacing_zyx = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])
    _, _, bbox = crop_volume_and_mask_to_bbox_zyx(
        np.zeros_like(bbox_mask_zyx, dtype=np.float32),
        bbox_mask_zyx,
        spacing_zyx=spacing_zyx,
    )

    processed = []
    for key in mask_keys:
        if key not in mask_group:
            raise KeyError(f"Missing ct/mask/{key}")
        cropped = np.asarray(mask_group[key][bbox[0], bbox[1], bbox[2]], dtype=np.float32)
        resized = resize_binary_mask_zyx(cropped, target_shape)
        if resized.dim() != 3:
            raise ValueError(
                f"Each CT ROI mask must resize to 3D (D,H,W), got shape={tuple(resized.shape)} for key={key!r}."
            )
        processed.append(resized)
    masks = torch.stack(processed, dim=0)
    if masks.dim() != 4 or masks.size(0) != len(mask_keys):
        raise ValueError(
            f"CT ROI masks must stack to (R,D,H,W) with R={len(mask_keys)}, got shape={tuple(masks.shape)}."
        )
    return masks


def preprocess_raw_ct_for_ctfm(
    ct_group: h5py.Group,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    if "mask" not in ct_group:
        raise KeyError("Missing ct/mask")

    volume_zyx = load_internal_ct_volume_zyx(ct_group)
    bbox_mask_zyx = load_ct_roi_union_mask_zyx(ct_group["mask"])
    zooms_xyz = tuple(float(v) for v in ct_group.attrs["zooms"][:3])
    spacing_zyx = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])
    cropped_volume_zyx, _, _ = crop_volume_and_mask_to_bbox_zyx(
        normalize_ct_for_learning(volume_zyx),
        bbox_mask_zyx,
        spacing_zyx=spacing_zyx,
    )
    return resize_volume_zyx(cropped_volume_zyx, target_shape)
