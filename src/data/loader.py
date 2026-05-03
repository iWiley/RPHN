import os

import pandas as pd
import torch
from queue import Queue
from threading import Thread

from tqdm import tqdm

from src.data.dataset import MultimodalDataset, rphn_collate_fn


class ThreadedDataLoader:
    """
    Wrap a DataLoader to prefetch batches in a background thread.
    """

    def __init__(self, loader, queue_size=4):
        self.loader = loader
        self.queue_size = queue_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        queue = Queue(maxsize=self.queue_size)
        end_token = object()

        def producer():
            try:
                for batch in self.loader:
                    queue.put(batch)
                queue.put(end_token)
            except Exception as exc:
                queue.put(exc)

        thread = Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is end_token:
                break
            if isinstance(item, Exception):
                raise RuntimeError("ThreadedDataLoader producer failed") from item
            yield item


def _pin_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    if isinstance(batch, list):
        return [_pin_batch(x) for x in batch]
    if isinstance(batch, tuple):
        return tuple(_pin_batch(x) for x in batch)
    return batch


class PreBatchedLoader:
    """
    Materialize evaluation batches once and reuse them across epochs.
    This is useful when the dataset is already cached in RAM and repeated
    collate-time stacking becomes the bottleneck.
    """

    def __init__(self, dataset, batch_size, collate_fn, desc="Prebatching eval"):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.collate_fn = collate_fn
        self.desc = desc
        self._batches = []
        self._build()

    def _build(self):
        total = len(self.dataset)
        for start in tqdm(range(0, total, self.batch_size), desc=self.desc, leave=False):
            end = min(start + self.batch_size, total)
            samples = [self.dataset[idx] for idx in range(start, end)]
            self._batches.append(_pin_batch(self.collate_fn(samples)))

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        yield from self._batches


class CachedTensorLoader:
    """
    Materialize a full split into contiguous CPU tensors once, then serve
    shuffled/sliced batches without per-step __getitem__/collate overhead.
    """

    def __init__(self, dataset, batch_size, shuffle, desc="Tensorizing split"):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.desc = desc
        self.num_samples = len(dataset)
        self._build()

    def _build(self):
        rows = [self.dataset[idx] for idx in tqdm(range(self.num_samples), desc=self.desc, leave=False)]
        wsi_t, wsi_p, ct_t, ct_masks, evt_os, tm_os, evt_ttr, tm_ttr, names = zip(*rows)

        self.wsi_is_tensor = len({tuple(x.shape) for x in wsi_t}) == 1
        self.pos_is_tensor = len({tuple(x.shape) for x in wsi_p}) == 1
        self.ct_is_tensor = len({tuple(x.shape) for x in ct_t}) == 1
        self.mask_is_tensor = len({tuple(x.shape) for x in ct_masks}) == 1

        self.wsi_t = torch.stack(wsi_t) if self.wsi_is_tensor else list(wsi_t)
        self.wsi_p = torch.stack(wsi_p) if self.pos_is_tensor else list(wsi_p)
        self.ct_t = torch.stack(ct_t) if self.ct_is_tensor else list(ct_t)
        self.ct_masks = torch.stack(ct_masks) if self.mask_is_tensor else list(ct_masks)
        self.evt_os = torch.stack(evt_os)
        self.tm_os = torch.stack(tm_os)
        self.evt_ttr = torch.stack(evt_ttr)
        self.tm_ttr = torch.stack(tm_ttr)
        self.names = list(names)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            idx = order[start:start + self.batch_size]
            idx_list = idx.tolist()
            yield _pin_batch((
                self.wsi_t.index_select(0, idx) if self.wsi_is_tensor else [self.wsi_t[i] for i in idx_list],
                self.wsi_p.index_select(0, idx) if self.pos_is_tensor else [self.wsi_p[i] for i in idx_list],
                self.ct_t.index_select(0, idx) if self.ct_is_tensor else [self.ct_t[i] for i in idx_list],
                self.ct_masks.index_select(0, idx) if self.mask_is_tensor else [self.ct_masks[i] for i in idx_list],
                self.evt_os.index_select(0, idx),
                self.tm_os.index_select(0, idx),
                self.evt_ttr.index_select(0, idx),
                self.tm_ttr.index_select(0, idx),
                [self.names[i] for i in idx_list],
            ))


def normalize_surv(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.map(str)
    return df


def load_patient_dirs(params: dict) -> list[tuple[str, str]]:
    path = params.get("data_h5")
    if path and os.path.exists(path):
        import h5py

        with h5py.File(path, "r") as handle:
            return [(path, str(key)) for key in handle.keys()]
    raise FileNotFoundError(f"HDF5 file not found at: {path}")


def create_dataloader(split_cfg, batch_size, is_train: bool = False, loader_cfg: dict | None = None):
    if not split_cfg:
        return None, 0

    loader_cfg = loader_cfg or {}
    cpu_count = os.cpu_count() or 1
    default_workers = min(8, cpu_count)
    num_workers = int(loader_cfg.get("num_workers", default_workers))
    pin_memory = bool(loader_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))
    cache_mode = str(loader_cfg.get("cache_mode", "none"))
    use_threaded_loader = bool(loader_cfg.get("use_threaded_loader", num_workers == 0))
    cache_workers = loader_cfg.get("cache_workers")

    if cache_mode == "none":
        use_threaded_loader = False

    dataset = MultimodalDataset(
        patient_entries=load_patient_dirs(split_cfg),
        surv_data=normalize_surv(pd.read_csv(split_cfg["surv_csv"], index_col=0)),
        cache_mode=cache_mode,
        cache_workers=cache_workers,
    )

    if is_train and cache_mode == "full":
        loader = CachedTensorLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            desc=f"Tensorizing {split_cfg.get('name', 'train')}",
        )
    elif (not is_train) and cache_mode == "full":
        loader = PreBatchedLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=rphn_collate_fn,
            desc=f"Prebatching {split_cfg.get('name', 'eval')}",
        )
    else:
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": is_train,
            "collate_fn": rphn_collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = persistent_workers
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

        base_loader = torch.utils.data.DataLoader(**dataloader_kwargs)
        loader = ThreadedDataLoader(base_loader) if use_threaded_loader and num_workers == 0 else base_loader

    return loader, len(dataset)
