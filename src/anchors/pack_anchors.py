from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


FNAME_RE = re.compile(r"^feat_(?P<sample>.+)_(?P<idx>\d+)\.(npy|npz)$")
WSI_CLASS_LABELS = {
    1: "Tumor-High",
    2: "Tumor-Low",
    3: "Immune",
    4: "Necrosis",
    5: "Fibrotic",
    6: "Vascular",
    7: "Steatosis",
    8: "Background",
}


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def load_feature(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        feat = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "feat" in data:
                feat = data["feat"]
            elif "data" in data:
                feat = data["data"]
            else:
                feat = data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported feature format: {path}")

    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim != 1:
        feat = feat.reshape(-1)
    return feat


def parse_class_id(class_dir: Path) -> int:
    name = class_dir.name
    if not name.startswith("class_"):
        raise ValueError(f"Invalid class directory: {class_dir}")
    return int(name.split("_")[1])


def collect_class_dirs(annotation_dir: Path) -> list[Path]:
    class_dirs = [path for path in annotation_dir.iterdir() if path.is_dir() and path.name.startswith("class_")]
    class_dirs.sort(key=parse_class_id)
    return class_dirs


def resolve_class_name(class_dir: Path, class_id: int) -> str:
    if re.fullmatch(r"class_\d+", class_dir.name):
        return WSI_CLASS_LABELS.get(class_id, class_dir.name)
    return class_dir.name


def load_class_features(
    annotation_dir: Path,
    normalize_instance: bool,
) -> tuple[list[np.ndarray], list[int], list[str], list[int]]:
    class_dirs = collect_class_dirs(annotation_dir)
    if not class_dirs:
        raise ValueError(f"No class_* dirs found in {annotation_dir}")

    mats = []
    class_ids = []
    class_names = []
    counts = []

    for class_dir in class_dirs:
        class_id = parse_class_id(class_dir)
        if class_id == 8:
            print(f"Skipping class {class_id} ({class_dir.name}) - Background/Blank")
            continue

        files = sorted(list(class_dir.glob("feat_*.npy")) + list(class_dir.glob("feat_*.npz")))
        files = [path for path in files if FNAME_RE.match(path.name)]
        if not files:
            continue

        features = [load_feature(path) for path in files]
        mat = np.stack(features).astype(np.float32)
        if normalize_instance:
            mat = l2_normalize(mat, axis=1)

        mats.append(mat)
        class_ids.append(class_id)
        class_names.append(resolve_class_name(class_dir, class_id))
        counts.append(len(files))

    if not mats:
        raise ValueError(f"No feat_*.npy/npz found under {annotation_dir}")

    return mats, class_ids, class_names, counts


def pairwise_cosine(anchors: np.ndarray) -> np.ndarray:
    normed = l2_normalize(anchors, axis=1)
    return np.clip(normed @ normed.T, -1.0, 1.0)


def init_centroids(mats: list[np.ndarray], normalize_centroid: bool) -> np.ndarray:
    centroids = np.stack([mat.mean(axis=0) for mat in mats]).astype(np.float32)
    if normalize_centroid:
        centroids = l2_normalize(centroids, axis=1)
    return centroids


def margin_to_weights(margin: np.ndarray, temperature: float, min_weight: float) -> np.ndarray:
    logits = np.clip(margin / max(temperature, 1e-6), -30.0, 30.0)
    weights = 1.0 / (1.0 + np.exp(-logits))
    return (min_weight + (1.0 - min_weight) * weights).astype(np.float32)


def refine_centroids(
    mats: list[np.ndarray],
    class_ids: list[int],
    class_names: list[str],
    normalize_centroid: bool,
    temperature: float,
    min_weight: float,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    centroids = init_centroids(mats, normalize_centroid=normalize_centroid)
    initial_centroids = centroids.copy()

    per_class_stats = []
    final_weights = []
    final_margins = []

    for _ in range(iterations):
        normed_centroids = l2_normalize(centroids, axis=1)
        next_centroids = []
        final_weights = []
        final_margins = []

        for idx, mat in enumerate(mats):
            sims = mat @ normed_centroids.T
            self_sim = sims[:, idx]
            rival_sims = np.delete(sims, idx, axis=1)
            rival_sim = rival_sims.max(axis=1) if rival_sims.size else np.zeros_like(self_sim)
            margin = self_sim - rival_sim
            weights = margin_to_weights(margin, temperature=temperature, min_weight=min_weight)

            weighted = (mat * weights[:, None]).sum(axis=0) / np.clip(weights.sum(), 1e-8, None)
            if normalize_centroid:
                weighted = l2_normalize(weighted[None, :], axis=1)[0]

            next_centroids.append(weighted.astype(np.float32))
            final_weights.append(weights)
            final_margins.append(margin.astype(np.float32))

        centroids = np.stack(next_centroids).astype(np.float32)

    initial_cos = pairwise_cosine(initial_centroids)
    final_cos = pairwise_cosine(centroids)

    for idx, (class_id, class_name, weights, margin) in enumerate(zip(class_ids, class_names, final_weights, final_margins)):
        per_class_stats.append(
            {
                "class_id": int(class_id),
                "name": class_name,
                "count_raw": int(mats[idx].shape[0]),
                "count_effective": float(weights.sum()),
                "weight_mean": float(weights.mean()),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "margin_mean": float(margin.mean()),
                "margin_median": float(np.median(margin)),
                "margin_p10": float(np.percentile(margin, 10)),
                "margin_p90": float(np.percentile(margin, 90)),
            }
        )

    diagnostics = {
        "method": "overlap_aware_refined_centroid",
        "iterations": int(iterations),
        "temperature": float(temperature),
        "min_weight": float(min_weight),
        "pairwise_cosine_before": initial_cos.tolist(),
        "pairwise_cosine_after": final_cos.tolist(),
        "per_class": per_class_stats,
    }
    return centroids, diagnostics


def save_pth(
    anchor_matrix: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    counts: list[int],
    output_path: Path,
    diagnostics: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors": torch.from_numpy(anchor_matrix),
        "class_ids": class_ids,
        "names": class_names,
        "counts": counts,
        "anchor_build": diagnostics,
    }
    torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack refined WSI anchors with overlap-aware down-weighting of ambiguous samples."
    )
    parser.add_argument("--annotation-dir", required=True, help="Directory containing class_1 ... class_K")
    parser.add_argument("--out-pth", required=True, help="Output .pth path consumed by train.py")
    parser.add_argument("--no-instance-l2", action="store_true", help="Disable per-feature L2 normalization before refinement")
    parser.add_argument("--no-centroid-l2", action="store_true", help="Disable centroid L2 normalization")
    parser.add_argument("--temperature", type=float, default=0.05, help="Soft margin temperature for overlap-aware weighting")
    parser.add_argument("--min-weight", type=float, default=0.15, help="Lower bound for ambiguous-sample weights")
    parser.add_argument("--iterations", type=int, default=2, help="Number of centroid refinement iterations")
    parser.add_argument("--save-json", default="", help="Optional diagnostics JSON output path")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation-dir not found: {annotation_dir}")

    mats, class_ids, class_names, counts = load_class_features(
        annotation_dir=annotation_dir,
        normalize_instance=not args.no_instance_l2,
    )
    anchor_matrix, diagnostics = refine_centroids(
        mats=mats,
        class_ids=class_ids,
        class_names=class_names,
        normalize_centroid=not args.no_centroid_l2,
        temperature=args.temperature,
        min_weight=args.min_weight,
        iterations=args.iterations,
    )

    out_pth = Path(args.out_pth).expanduser().resolve()
    save_pth(anchor_matrix, class_ids, class_names, counts, out_pth, diagnostics)

    if args.save_json:
        out_json = Path(args.save_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Saved refined anchors: {out_pth} | shape={anchor_matrix.shape}")
    print(f"Classes: {class_ids}")
    print(f"Counts : {counts}")
    for item in diagnostics["per_class"]:
        print(
            f"[{item['class_id']}] {item['name']}: "
            f"raw={item['count_raw']} effective={item['count_effective']:.1f} "
            f"margin_mean={item['margin_mean']:.4f} weight_mean={item['weight_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
