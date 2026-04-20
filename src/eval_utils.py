from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MANUSCRIPT_ROOT.parent
MANUSCRIPT_SRC = MANUSCRIPT_ROOT / "src"


def bootstrap_src_path() -> None:
    for path in (MANUSCRIPT_SRC, MANUSCRIPT_ROOT, WORKSPACE_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


def resolve_existing_path(raw_path: str | Path, *, include_cwd: bool = False) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = []
    if include_cwd:
        candidates.append(Path.cwd() / path)
    candidates.extend([MANUSCRIPT_ROOT / path, WORKSPACE_ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_repo_path(raw_path: str | Path | None) -> str | None:
    if not raw_path:
        return raw_path
    return str(resolve_existing_path(raw_path))


def default_config_path(name: str = "default.yaml") -> Path:
    return (MANUSCRIPT_ROOT / "configs" / name).resolve()


def load_runtime_config(cfg_path: str | Path) -> dict:
    cfg_path = Path(cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    for split_name in ("train", "val"):
        split_cfg = cfg.get(split_name)
        if split_cfg:
            for key in ("data_h5", "surv_csv"):
                split_cfg[key] = resolve_repo_path(split_cfg.get(key))

    for split_cfg in cfg.get("test", []):
        for key in ("data_h5", "surv_csv"):
            split_cfg[key] = resolve_repo_path(split_cfg.get(key))

    cfg["wsi_anchors_path"] = resolve_repo_path(cfg.get("wsi_anchors_path"))
    return cfg

bootstrap_src_path()
from src.extractors.ct import CTFMFeatureEncoder
from src.extractors.wsi import GigapathFeatureEncoder
from src.models.rphn import RPHN
from src.utils.feature_contracts import CANONICAL_CTFM_FEATURE_SHAPE


def load_config(cfg_path: str | Path) -> dict:
    return load_runtime_config(cfg_path)


def load_anchor_payload(path: str | None):
    if not path:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def anchor_tensor_from_payload(payload):
    if payload is None:
        return None
    return payload.get("anchors", payload) if isinstance(payload, dict) else payload


def require_anchor_tensor(path: str | None):
    payload = load_anchor_payload(path)
    anchors = anchor_tensor_from_payload(payload)
    if anchors is None:
        raise RuntimeError(
            "The active WSI branch requires an expert-initialized anchor payload. "
            "Please provide 'wsi_anchors_path' with a valid anchor tensor."
        )
    return anchors


def build_training_model(cfg: dict, device: torch.device) -> RPHN:
    gigapath_dir = resolve_existing_path("model/prov-gigapath")
    ctfm_dir = resolve_existing_path("model/ct-fm/ct_fm_feature_extractor")
    if not gigapath_dir.exists():
        raise FileNotFoundError(f"Could not find local GigaPath weights under {gigapath_dir}.")
    if not ctfm_dir.exists():
        raise FileNotFoundError(f"Could not find local CT-FM weights under {ctfm_dir}.")

    wsi_anchors_init = require_anchor_tensor(cfg.get("wsi_anchors_path"))
    return RPHN(
        hidden_dim=256,
        dropout=float(cfg.get("dropout", 0.1)),
        wsi_anchors_init=wsi_anchors_init,
        wsi_backbone=GigapathFeatureEncoder(str(gigapath_dir), device=device),
        ct_backbone=CTFMFeatureEncoder(ctfm_dir, device=device),
        ct_feature_dim=int(CANONICAL_CTFM_FEATURE_SHAPE[0]),
        ct_latent_pooling="attention",
        ct_latent_refine_depth=1,
    ).to(device)


def build_eval_model(cfg_path: str | Path, device: torch.device):
    cfg = load_config(cfg_path)
    return cfg, build_training_model(cfg, device)


def load_model_weights(model, model_path: str | Path, device: torch.device, strict: bool = False):
    payload = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(payload, dict):
        for key in ("ema_state_dict", "state_dict", "model_state_dict", "model"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return model.load_state_dict(state_dict, strict=strict)
    return model.load_state_dict(payload, strict=strict)


def select_eval_cohort(cfg: dict, split_name: str = "test", preferred_name: str | None = None):
    if split_name == "test":
        candidates = list(cfg.get("test", []))
    else:
        split_cfg = cfg.get(split_name)
        candidates = [split_cfg] if split_cfg else []

    if not candidates:
        raise KeyError(f"No cohorts found for split={split_name!r}.")

    if preferred_name is None:
        return 0, candidates[0]

    preferred_name = str(preferred_name)
    for idx, item in enumerate(candidates):
        if str(item.get("name")) == preferred_name:
            return idx, item
    raise KeyError(f"Could not find cohort name={preferred_name!r} in split={split_name!r}.")


def load_anchor_names(path: str | None, prefix: str = "anchor") -> list[str]:
    payload = load_anchor_payload(resolve_repo_path(path))
    if payload is None:
        return []

    if isinstance(payload, dict):
        for key in ("anchor_names", "names", "labels", "classes"):
            names = payload.get(key)
            if names is not None:
                return [str(name) for name in names]

    anchors = anchor_tensor_from_payload(payload)
    if anchors is None or not hasattr(anchors, "shape"):
        return []
    return [f"{prefix}_{idx}" for idx in range(int(anchors.shape[0]))]
