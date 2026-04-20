import atexit
import gc
import os
from queue import Queue
from threading import Thread

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.ct_h5 import preprocess_ct_masks_for_ctfm, preprocess_raw_ct_for_ctfm
from src.utils.feature_contracts import DEFAULT_CT_TARGET_SHAPE

try:
    import imagecodecs
except ImportError:
    imagecodecs = None


HEAVY_DATA_CACHE = {}
H5_FILE_CACHE = {}
CT_MASK_KEYS = ("liver", "liver_lesion_or_tumor", "liver_peritumoral", "liver_vessels")


def _close_cached_h5_files():
    for handle in H5_FILE_CACHE.values():
        try:
            handle.close()
        except Exception:
            pass
    H5_FILE_CACHE.clear()


def clear_dataset_caches(close_h5: bool = True, drop_heavy: bool = True) -> None:
    if drop_heavy:
        HEAVY_DATA_CACHE.clear()
    if close_h5:
        _close_cached_h5_files()
    gc.collect()


atexit.register(_close_cached_h5_files)


def _parse_patient_entry(entry) -> tuple[str, str]:
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise ValueError(
            f"Patient entries must be (h5_path, patient_id) pairs, got {entry!r}."
        )
    h5_path, patient_id = entry
    return str(h5_path), str(patient_id)


def _extract_patient_id(entry) -> str:
    return _parse_patient_entry(entry)[1]


def _cache_key(entry) -> tuple[str, str]:
    h5_path, patient_id = _parse_patient_entry(entry)
    return os.path.abspath(h5_path), patient_id


def _get_h5_handle(h5_path: str):
    cache_key = (os.getpid(), os.path.abspath(h5_path))
    handle = H5_FILE_CACHE.get(cache_key)
    if handle is None or not handle.id.valid:
        handle = h5py.File(h5_path, "r")
        H5_FILE_CACHE[cache_key] = handle
    return handle


def _has_wsi_raw(base) -> bool:
    return (
        "wsi" in base
        and "patches" in base["wsi"]
        and "images" in base["wsi"]["patches"]
        and "coords" in base["wsi"]["patches"]
    )


def _has_ct_raw(base) -> bool:
    return "ct" in base and ("slices" in base["ct"] or "bundle" in base["ct"])


def _has_ct_masks(base) -> bool:
    if "ct" not in base or "mask" not in base["ct"]:
        return False
    mask_group = base["ct"]["mask"]
    return all(key in mask_group for key in CT_MASK_KEYS)


def _ensure_raw_entry_ready(
    patient_entry,
    modality: str,
) -> None:
    h5_path, pname = _parse_patient_entry(patient_entry)
    handle = _get_h5_handle(h5_path)
    if pname not in handle:
        raise KeyError(f"Missing patient group '{pname}' in H5: {h5_path}")
    base = handle[pname]

    if modality == "wsi":
        ready = _has_wsi_raw(base)
    elif modality == "ct":
        ready = _has_ct_masks(base) and _has_ct_raw(base)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if ready:
        return
    raise RuntimeError(f"Missing required {modality.upper()} raw inputs for sample '{pname}' in {os.path.basename(h5_path)}.")


def decode_jxl_to_tensor(jxl_bytes):
    if imagecodecs is None:
        raise ImportError("imagecodecs is required for raw WSI JPEG XL decoding.")

    if isinstance(jxl_bytes, np.ndarray):
        jxl_bytes = jxl_bytes.tobytes()

    img_np = imagecodecs.jpegxl_decode(jxl_bytes)
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    return img_tensor.permute(2, 0, 1)


def _load_ct_masks_from_base(base) -> np.ndarray:
    if not _has_ct_masks(base):
        raise KeyError("Missing CT ROI masks under 'ct/mask'.")
    masks = _canonicalize_ct_masks(
        preprocess_ct_masks_for_ctfm(base["ct"], DEFAULT_CT_TARGET_SHAPE, mask_keys=CT_MASK_KEYS)
    )
    return masks.cpu().numpy()


def _canonicalize_ct_masks(ct_masks: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(ct_masks):
        ct_masks = torch.from_numpy(np.asarray(ct_masks)).float()
    else:
        ct_masks = ct_masks.float()

    # Older preprocessing emitted a singleton per-ROI channel, i.e. (R,1,D,H,W).
    if ct_masks.dim() == 5 and ct_masks.size(1) == 1:
        ct_masks = ct_masks.squeeze(1)

    if ct_masks.dim() != 4:
        raise ValueError(f"CT masks must be 4D (R,D,H,W), got shape={tuple(ct_masks.shape)}")
    if ct_masks.size(0) != len(CT_MASK_KEYS):
        raise ValueError(
            f"CT masks must provide {len(CT_MASK_KEYS)} ROI channels {CT_MASK_KEYS}, got shape={tuple(ct_masks.shape)}"
        )
    return ct_masks


def load_patient_data(patient_entry):
    h5_path, pname = _parse_patient_entry(patient_entry)

    try:
        handle = _get_h5_handle(h5_path)
        base = handle[pname]

        if not _has_wsi_raw(base):
            raise RuntimeError("WSI inputs require datasets under 'wsi/patches'.")
        if not _has_ct_raw(base):
            raise RuntimeError("CT inputs require raw CT under 'ct/bundle' or 'ct/slices'.")

        wsi_data = base["wsi"]["patches"]["images"][:]
        wsi_coords = base["wsi"]["patches"]["coords"][:]
        ct_data = preprocess_raw_ct_for_ctfm(base["ct"], DEFAULT_CT_TARGET_SHAPE).cpu().numpy()
        ct_masks = _load_ct_masks_from_base(base)
    except Exception as e:
        raise RuntimeError(f"Error loading H5 {h5_path} ({pname}): {e}") from e

    return pname, wsi_data, wsi_coords, ct_data, ct_masks


def _materialize_cached_sample(wsi_data, wsi_coords, ct_data, ct_masks):
    tensors = [decode_jxl_to_tensor(blob) for blob in wsi_data]
    if not tensors:
        raise ValueError("WSI raw data decoded to empty tensor batch.")

    wsi_tensor = torch.stack(tensors, dim=0)
    wsi_pos = torch.from_numpy(np.asarray(wsi_coords)).float()
    ct_tensor = ct_data if torch.is_tensor(ct_data) else torch.from_numpy(np.asarray(ct_data)).float()
    ct_masks = _canonicalize_ct_masks(ct_masks)

    return wsi_tensor, wsi_pos, ct_tensor, ct_masks


def _resize_ct_masks(ct_masks: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    ct_masks = _canonicalize_ct_masks(ct_masks)

    ct_masks = ct_masks.unsqueeze(0)
    if tuple(int(v) for v in ct_masks.shape[-3:]) != tuple(int(v) for v in target_shape):
        current_shape = tuple(int(v) for v in ct_masks.shape[-3:])
        shrink_shape = tuple(min(src, dst) for src, dst in zip(current_shape, target_shape))
        if shrink_shape != current_shape:
            ct_masks = F.adaptive_avg_pool3d(ct_masks, output_size=shrink_shape)
        if shrink_shape != tuple(int(v) for v in target_shape):
            ct_masks = F.interpolate(
                ct_masks,
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
    return ct_masks.squeeze(0).clamp_(0.0, 1.0)


class MultimodalDataset(Dataset):
    def __init__(
        self,
        patient_entries,
        surv_data=None,
        cache_mode="none",
        cache_workers=None,
    ):
        self.patient_entries = patient_entries
        self.cache_mode = cache_mode
        self.cache_workers = cache_workers
        self.surv_data = surv_data.copy() if surv_data is not None else None

        if self.surv_data is not None:
            self.surv_data.index = self.surv_data.index.map(str)

        self.surv_dict = self.surv_data.to_dict("index") if self.surv_data is not None else {}

        if self.surv_data is not None:
            missing_surv = [
                _extract_patient_id(p)
                for p in self.patient_entries
                if _extract_patient_id(p) not in self.surv_dict
            ]
            if missing_surv:
                preview = ", ".join(missing_surv[:8])
                raise KeyError(
                    f"Missing survival rows for {len(missing_surv)} patients. "
                    f"Examples: {preview}"
                )

        self._validate_raw_inputs()

        if self.cache_mode == "full":
            self._fill_cache()

    def _validate_raw_inputs(self):
        if not self.patient_entries:
            raise ValueError("patient_entries must not be empty.")
        for patient_entry in self.patient_entries:
            _ensure_raw_entry_ready(patient_entry, modality="wsi")
            _ensure_raw_entry_ready(patient_entry, modality="ct")

    def _fill_cache(self):
        cached_keys = set(HEAVY_DATA_CACHE.keys())
        missing = [p for p in self.patient_entries if _cache_key(p) not in cached_keys]
        if not missing:
            return

        print(
            f"Loading {len(missing)} patients into RAM..."
        )
        max_workers = self.cache_workers
        if max_workers is None:
            max_workers = 4
        max_workers = max(1, min(int(max_workers), len(missing)))

        if max_workers == 1:
            for patient_entry in tqdm(missing, total=len(missing)):
                try:
                    _, wsi_data, wsi_coords, ct_data, ct_masks = load_patient_data(patient_entry)
                    HEAVY_DATA_CACHE[_cache_key(patient_entry)] = _materialize_cached_sample(
                        wsi_data,
                        wsi_coords,
                        ct_data,
                        ct_masks,
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load {patient_entry}: {e}") from e
            gc.collect()
            return

        read_queue = Queue(maxsize=max_workers * 2)
        write_queue = Queue()
        stop_token = object()

        def consumer():
            while True:
                item = read_queue.get()
                if item is stop_token:
                    read_queue.task_done()
                    break

                patient_entry, payload = item
                try:
                    _, wsi_data, wsi_coords, ct_data, ct_masks = payload
                    write_queue.put(
                        (
                            patient_entry,
                            _materialize_cached_sample(wsi_data, wsi_coords, ct_data, ct_masks),
                            None,
                            None,
                        )
                    )
                except Exception as exc:
                    write_queue.put((patient_entry, None, exc, None))
                finally:
                    read_queue.task_done()

        producer_error_token = object()

        def producer():
            try:
                for patient_entry in missing:
                    payload = load_patient_data(patient_entry)
                    read_queue.put((patient_entry, payload))
            except Exception as exc:
                write_queue.put((None, None, exc, producer_error_token))
            finally:
                for _ in range(max_workers):
                    read_queue.put(stop_token)

        workers = [Thread(target=consumer, daemon=True) for _ in range(max_workers)]
        producer_thread = Thread(target=producer, daemon=True)
        for worker in workers:
            worker.start()
        producer_thread.start()

        try:
            for _ in tqdm(range(len(missing)), total=len(missing)):
                patient_entry, cache_payload, error, token = write_queue.get()
                if token is producer_error_token:
                    raise RuntimeError(f"Failed while reading H5 cache source: {error}") from error
                if error is not None:
                    raise RuntimeError(f"Failed to load {patient_entry}: {error}") from error
                HEAVY_DATA_CACHE[_cache_key(patient_entry)] = cache_payload

            read_queue.join()
        finally:
            producer_thread.join()
            for worker in workers:
                worker.join()
        gc.collect()

    def __len__(self):
        return len(self.patient_entries)

    def __getitem__(self, idx):
        patient_entry = self.patient_entries[idx]
        pname = _extract_patient_id(patient_entry)
        cache_key = _cache_key(patient_entry)

        if cache_key in HEAVY_DATA_CACHE:
            wsi_tensor, wsi_pos, ct_tensor, ct_masks = HEAVY_DATA_CACHE[cache_key]
        else:
            _, wsi_data, wsi_coords, ct_data, ct_masks = load_patient_data(
                patient_entry,
            )
            wsi_tensor, wsi_pos, ct_tensor, ct_masks = _materialize_cached_sample(
                wsi_data,
                wsi_coords,
                ct_data,
                ct_masks,
            )

        if ct_tensor.dim() != 4 or ct_tensor.size(0) != 1:
            raise ValueError(
                f"CT tensors must be preprocessed as (1,D,H,W), got shape={tuple(ct_tensor.shape)} "
                f"for patient {pname}."
            )

        ct_masks = _resize_ct_masks(ct_masks, tuple(int(v) for v in ct_tensor.shape[-3:]))

        if not self.surv_dict:
            raise RuntimeError("Survival data is required and cannot be empty.")
        if pname not in self.surv_dict:
            raise KeyError(f"Missing survival row for patient: {pname}")

        s_info = self.surv_dict[pname]

        def _require_float(key):
            if key not in s_info:
                raise KeyError(f"Missing survival column '{key}' for patient: {pname}")
            value = s_info[key]
            if value is None:
                raise ValueError(f"Survival value '{key}' is None for patient: {pname}")
            try:
                if np.isnan(value):
                    raise ValueError(f"Survival value '{key}' is NaN for patient: {pname}")
            except TypeError:
                pass
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Survival value '{key}' is not numeric for patient: {pname}, value={value!r}"
                ) from exc

        evt_os = _require_float("OS")
        tm_os = _require_float("OS_Time")
        evt_rfs = _require_float("RFS")
        tm_rfs = _require_float("RFS_Time")

        return (
            wsi_tensor,
            wsi_pos,
            ct_tensor,
            ct_masks,
            torch.tensor(evt_os, dtype=torch.float32),
            torch.tensor(tm_os, dtype=torch.float32),
            torch.tensor(evt_rfs, dtype=torch.float32),
            torch.tensor(tm_rfs, dtype=torch.float32),
            pname,
        )


def rphn_collate_fn(batch):
    wsi_t, wsi_p, ct_t, ct_masks, evt_os, tm_os, evt_rfs, tm_rfs, names = zip(*batch)

    wsi_same_shape = len({tuple(x.shape) for x in wsi_t}) == 1
    pos_same_shape = len({tuple(x.shape) for x in wsi_p}) == 1
    ct_same_shape = len({tuple(x.shape) for x in ct_t}) == 1
    mask_same_shape = len({tuple(x.shape) for x in ct_masks}) == 1

    return (
        torch.stack(wsi_t) if wsi_same_shape else list(wsi_t),
        torch.stack(wsi_p) if pos_same_shape else list(wsi_p),
        torch.stack(ct_t) if ct_same_shape else list(ct_t),
        torch.stack(ct_masks) if mask_same_shape else list(ct_masks),
        torch.stack(evt_os),
        torch.stack(tm_os),
        torch.stack(evt_rfs),
        torch.stack(tm_rfs),
        list(names),
    )
