from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.feature_contracts import CANONICAL_CTFM_FEATURE_SHAPE, CT_EXPECTED_LAYER_OFFSET_FROM_LAST

DEFAULT_CTFM_FEATURE_SHAPE = CANONICAL_CTFM_FEATURE_SHAPE


@dataclass(frozen=True)
class CTFMOutputSelection:
    selected_output: str
    resolved_output_index: int
    layer_offset_from_last: int
    num_encoder_outputs: int


def ensure_canonical_ctfm_selection(
    selection: CTFMOutputSelection,
    expected_layer_offset_from_last: int = CT_EXPECTED_LAYER_OFFSET_FROM_LAST,
) -> None:
    expected_offset = int(expected_layer_offset_from_last)
    if int(selection.layer_offset_from_last) != expected_offset:
        raise RuntimeError(
            "CT-FM output selection drifted from the canonical encoder layer: "
            f"got selected_output={selection.selected_output!r}, "
            f"layer_offset_from_last={selection.layer_offset_from_last}, "
            f"expected={expected_offset}."
        )


def unwrap_ctfm_output(
    output: object,
    preferred_layer_offset_from_last: int = 1,
    preferred_feature_shape: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, CTFMOutputSelection]:
    if isinstance(output, dict):
        for key in ("encoder", "feature_map", "features", "embedding"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value, CTFMOutputSelection(
                    selected_output=f"dict:{key}",
                    resolved_output_index=-1,
                    layer_offset_from_last=-1,
                    num_encoder_outputs=1,
                )
        for key, value in output.items():
            if torch.is_tensor(value):
                return value, CTFMOutputSelection(
                    selected_output=f"dict:{key}:first_tensor",
                    resolved_output_index=-1,
                    layer_offset_from_last=-1,
                    num_encoder_outputs=1,
                )

    if isinstance(output, (list, tuple)):
        tensor_candidates = [value for value in output if torch.is_tensor(value)]
        if not tensor_candidates:
            raise TypeError("CT-FM sequence output did not contain any tensor.")

        num_tensors = len(tensor_candidates)
        preferred_offset = int(preferred_layer_offset_from_last)
        if 0 <= preferred_offset < num_tensors:
            preferred_index = num_tensors - 1 - preferred_offset
            preferred_tensor = tensor_candidates[preferred_index]
            return preferred_tensor, CTFMOutputSelection(
                selected_output=f"sequence:tensor_{preferred_index}_of_{num_tensors}:nminus{preferred_offset}",
                resolved_output_index=preferred_index,
                layer_offset_from_last=preferred_offset,
                num_encoder_outputs=num_tensors,
            )

        raise RuntimeError(
            "CT-FM output does not expose the expected encoder layer. "
            f"preferred_layer_offset_from_last={preferred_offset}, "
            f"num_encoder_outputs={num_tensors}, "
            f"preferred_feature_shape={preferred_feature_shape}."
        )

    if torch.is_tensor(output):
        return output, CTFMOutputSelection(
            selected_output="tensor",
            resolved_output_index=-1,
            layer_offset_from_last=0,
            num_encoder_outputs=1,
        )

    raise TypeError(f"Unsupported CT-FM output type: {type(output).__name__}")
