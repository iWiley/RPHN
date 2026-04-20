from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import imagecodecs
import nibabel as nib
import numpy as np


MAGIC = b"NIIJXL1\x00"
VERSION = 1
UINT64 = struct.Struct("<Q")


def _encoding_plan(dtype: np.dtype) -> tuple[np.dtype, str]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return np.dtype(np.uint8), "native"
    if dtype == np.dtype(np.uint16):
        return np.dtype(np.uint16), "native"
    if dtype == np.dtype(np.uint32):
        return np.dtype(np.uint16), "split_uint32_to_uint16x2"
    if dtype == np.dtype(np.float32):
        return np.dtype(np.float32), "native"
    if dtype == np.dtype(np.int16):
        return np.dtype(np.uint16), "view_uint16"
    if dtype == np.dtype(np.int32):
        return np.dtype(np.uint16), "split_int32_to_uint16x2"
    if dtype == np.dtype(np.int8):
        return np.dtype(np.uint8), "view_uint8"
    raise ValueError(f"Unsupported dtype for niijxl encoding: {dtype}")


def _prepare_slice_for_jxl(slice_arr: np.ndarray, plan: str, encoded_dtype: np.dtype) -> np.ndarray:
    arr = np.ascontiguousarray(slice_arr)
    if plan == "native":
        return arr.astype(encoded_dtype, copy=False)
    if plan == "view_uint16":
        return arr.view(np.uint16)
    if plan in {"split_uint32_to_uint16x2", "split_int32_to_uint16x2"}:
        return arr.view(np.uint16).reshape(arr.shape + (2,))
    if plan == "view_uint8":
        return arr.view(np.uint8)
    raise ValueError(f"Unknown encoding plan: {plan}")


def _restore_slice_from_jxl(decoded: np.ndarray, original_dtype: np.dtype, plan: str) -> np.ndarray:
    decoded = np.ascontiguousarray(decoded)
    if plan == "native":
        return decoded.astype(original_dtype, copy=False)
    if plan == "view_uint16":
        return decoded.view(np.int16)
    if plan == "split_uint32_to_uint16x2":
        return np.ascontiguousarray(decoded).view(np.uint32).reshape(decoded.shape[:2])
    if plan == "split_int32_to_uint16x2":
        return np.ascontiguousarray(decoded).view(np.int32).reshape(decoded.shape[:2])
    if plan == "view_uint8":
        return decoded.view(np.int8)
    raise ValueError(f"Unknown encoding plan: {plan}")


def _read_exact(fp, n: int) -> bytes:
    data = fp.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data


def pack_nifti_image_to_niijxl_bytes(
    img: nib.Nifti1Image,
    *,
    slice_axis: int = 2,
    level: int = 100,
    effort: int = 7,
) -> bytes:
    arr = np.asanyarray(img.dataobj)
    if arr.ndim != 3:
        raise ValueError(f"niijxl currently supports 3D only, got ndim={arr.ndim}")
    encoded_dtype, plan = _encoding_plan(arr.dtype)
    axis_moved = np.moveaxis(arr, slice_axis, 0)
    encoded_slices: list[bytes] = []
    for slice_arr in axis_moved:
        prepared = _prepare_slice_for_jxl(slice_arr, plan, encoded_dtype)
        encoded_slices.append(imagecodecs.jpegxl_encode(prepared, level=level, effort=effort, lossless=True))

    meta = {
        "version": VERSION,
        "shape": list(arr.shape),
        "dtype": np.dtype(arr.dtype).str,
        "encoded_dtype": encoded_dtype.str,
        "encoding_plan": plan,
        "slice_axis": int(slice_axis),
        "slice_count": len(encoded_slices),
        "source_name": None,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    header_bytes = img.to_bytes()

    buffer = io.BytesIO()
    buffer.write(MAGIC)
    buffer.write(UINT64.pack(VERSION))
    buffer.write(UINT64.pack(len(meta_bytes)))
    buffer.write(meta_bytes)
    buffer.write(UINT64.pack(len(header_bytes)))
    buffer.write(header_bytes)
    buffer.write(UINT64.pack(len(encoded_slices)))
    for encoded in encoded_slices:
        buffer.write(UINT64.pack(len(encoded)))
        buffer.write(encoded)
    return buffer.getvalue()


def unpack_niijxl_bytes_to_image(bundle_bytes: bytes) -> tuple[nib.Nifti1Image, dict[str, object]]:
    fp = io.BytesIO(bundle_bytes)
    if _read_exact(fp, len(MAGIC)) != MAGIC:
        raise ValueError("Not a niijxl bundle")
    version = UINT64.unpack(_read_exact(fp, UINT64.size))[0]
    if version != VERSION:
        raise ValueError(f"Unsupported niijxl version: {version}")
    meta_len = UINT64.unpack(_read_exact(fp, UINT64.size))[0]
    meta = json.loads(_read_exact(fp, meta_len).decode("utf-8"))
    header_len = UINT64.unpack(_read_exact(fp, UINT64.size))[0]
    header_bytes = _read_exact(fp, header_len)
    slice_count = UINT64.unpack(_read_exact(fp, UINT64.size))[0]

    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["dtype"])
    encoded_dtype = np.dtype(meta["encoded_dtype"])
    plan = str(meta["encoding_plan"])
    slice_axis = int(meta["slice_axis"])

    moved_shape = (shape[slice_axis],) + tuple(shape[i] for i in range(len(shape)) if i != slice_axis)
    moved = np.empty(moved_shape, dtype=dtype)
    for i in range(slice_count):
        payload_len = UINT64.unpack(_read_exact(fp, UINT64.size))[0]
        payload = _read_exact(fp, payload_len)
        decoded = imagecodecs.jpegxl_decode(payload)
        if decoded.dtype != encoded_dtype:
            decoded = decoded.astype(encoded_dtype, copy=False)
        moved[i] = _restore_slice_from_jxl(decoded, dtype, plan)

    arr = np.moveaxis(moved, 0, slice_axis)
    img = nib.Nifti1Image.from_bytes(header_bytes)
    affine = np.asarray(img.affine, dtype=np.float64)
    restored = nib.Nifti1Image(arr, affine=affine, header=img.header.copy())
    return restored, meta


def pack_xyz_array_to_niijxl_bytes(
    data_xyz: np.ndarray,
    *,
    affine: np.ndarray,
    zooms_xyz: tuple[float, float, float],
    source_path: str = "",
    source_type: str = "nifti",
    slope: float = 1.0,
    inter: float = 0.0,
    effort: int = 7,
) -> bytes:
    arr = np.asarray(data_xyz)
    img = nib.Nifti1Image(arr, affine=np.asarray(affine, dtype=np.float64).reshape(4, 4))
    img.header.set_data_dtype(arr.dtype)
    img.header.set_zooms(tuple(float(v) for v in zooms_xyz))
    img.header.set_slope_inter(float(slope), float(inter))
    bundle = pack_nifti_image_to_niijxl_bytes(img, slice_axis=2, effort=effort)
    return bundle


def decode_niijxl_bytes_to_xyz(bundle_bytes: bytes) -> tuple[np.ndarray, dict[str, object], nib.Nifti1Image]:
    img, meta = unpack_niijxl_bytes_to_image(bundle_bytes)
    arr = np.asanyarray(img.dataobj)
    return arr, meta, img
