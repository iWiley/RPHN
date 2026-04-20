from __future__ import annotations

from collections.abc import Mapping


CT_FEATURE_CONTRACT_VERSION = "2026-03-31-canonical-v1"
FEATURE_CONTRACT_VERSION = CT_FEATURE_CONTRACT_VERSION
CANONICAL_CT_INPUT_SHAPE = (128, 128, 128)
CANONICAL_CTFM_FEATURE_SHAPE = (256, 16, 16, 16)
DEFAULT_CT_TARGET_SHAPE = CANONICAL_CT_INPUT_SHAPE
CT_EXPECTED_MASK_POLICY = "roi_union_bbox_crop_10mm"
CT_EXPECTED_LAYER_OFFSET_FROM_LAST = 1

WSI_FEATURE_CONTRACT_VERSION = "2026-03-31-full-backbone-v2"
CANONICAL_WSI_INPUT_SIZE = (224, 224)
CANONICAL_WSI_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
CANONICAL_WSI_NORMALIZE_STD = (0.229, 0.224, 0.225)
WSI_NORMALIZE_MEAN = CANONICAL_WSI_NORMALIZE_MEAN
WSI_NORMALIZE_STD = CANONICAL_WSI_NORMALIZE_STD
CANONICAL_WSI_TRUNCATED_BLOCKS = 0
WSI_BACKBONE_CUT = "full_backbone"
WSI_FEATURE_DIM = 1536


def _tuple_from_attr(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(v) for v in value.tolist())
    except AttributeError:
        return None


def validate_ct_feature_attrs(attrs: Mapping[str, object], actual_shape: tuple[int, ...] | None = None) -> list[str]:
    errors: list[str] = []

    if attrs.get("feature_semantics") != "encoder_feature_map":
        errors.append("feature_semantics must be 'encoder_feature_map'")
    if attrs.get("backbone") != "ct-fm":
        errors.append("backbone must be 'ct-fm'")
    if attrs.get("backbone_family") != "ct_fm":
        errors.append("backbone_family must be 'ct_fm'")
    if attrs.get("feature_key") != "features":
        errors.append("feature_key must be 'features'")
    if attrs.get("feature_contract_version") != CT_FEATURE_CONTRACT_VERSION:
        errors.append(f"feature_contract_version must be '{CT_FEATURE_CONTRACT_VERSION}'")
    if attrs.get("mask_policy") != CT_EXPECTED_MASK_POLICY:
        errors.append(f"mask_policy must be '{CT_EXPECTED_MASK_POLICY}'")
    layer_offset = attrs.get("layer_offset_from_last")
    if layer_offset is None or int(layer_offset) != CT_EXPECTED_LAYER_OFFSET_FROM_LAST:
        errors.append(f"layer_offset_from_last must be {CT_EXPECTED_LAYER_OFFSET_FROM_LAST}")

    shape = _tuple_from_attr(attrs.get("shape"))
    if shape != CANONICAL_CTFM_FEATURE_SHAPE:
        errors.append(f"shape must be {CANONICAL_CTFM_FEATURE_SHAPE}")
    if actual_shape is not None and tuple(int(v) for v in actual_shape) != CANONICAL_CTFM_FEATURE_SHAPE:
        errors.append(f"dataset shape must be {CANONICAL_CTFM_FEATURE_SHAPE}")

    input_target_shape = _tuple_from_attr(attrs.get("input_target_shape"))
    if input_target_shape != CANONICAL_CT_INPUT_SHAPE:
        errors.append(f"input_target_shape must be {CANONICAL_CT_INPUT_SHAPE}")

    return errors


def validate_wsi_feature_attrs(attrs: Mapping[str, object], actual_shape: tuple[int, ...] | None = None) -> list[str]:
    errors: list[str] = []

    if attrs.get("feature_semantics") != "final_backbone_embedding":
        errors.append("feature_semantics must be 'final_backbone_embedding'")
    if attrs.get("backbone_family") != "gigapath":
        errors.append("backbone_family must be 'gigapath'")
    if attrs.get("feature_key") != "features":
        errors.append("feature_key must be 'features'")
    if attrs.get("feature_contract_version") != WSI_FEATURE_CONTRACT_VERSION:
        errors.append(f"feature_contract_version must be '{WSI_FEATURE_CONTRACT_VERSION}'")
    if attrs.get("input_normalization") != "imagenet":
        errors.append("input_normalization must be 'imagenet'")
    if attrs.get("backbone_cut") != WSI_BACKBONE_CUT:
        errors.append(f"backbone_cut must be '{WSI_BACKBONE_CUT}'")
    truncated_blocks = attrs.get("truncated_blocks")
    if truncated_blocks is None or int(truncated_blocks) != CANONICAL_WSI_TRUNCATED_BLOCKS:
        errors.append(f"truncated_blocks must be {CANONICAL_WSI_TRUNCATED_BLOCKS}")

    feature_dim = attrs.get("feature_dim")
    if feature_dim is None or int(feature_dim) != WSI_FEATURE_DIM:
        errors.append(f"feature_dim must be {WSI_FEATURE_DIM}")
    if actual_shape is not None:
        if len(actual_shape) != 2:
            errors.append(f"dataset shape must be 2D, got {tuple(int(v) for v in actual_shape)}")
        elif int(actual_shape[1]) != WSI_FEATURE_DIM:
            errors.append(f"dataset feature dim must be {WSI_FEATURE_DIM}")

    input_target_size = _tuple_from_attr(attrs.get("input_target_size"))
    if input_target_size != CANONICAL_WSI_INPUT_SIZE:
        errors.append(f"input_target_size must be {CANONICAL_WSI_INPUT_SIZE}")

    return errors
