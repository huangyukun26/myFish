from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_classes(path: Optional[Path], fallback: List[str]) -> List[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_candidate_features(path: Path, candidates: List[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidates missing from {path}; first={missing[:10]}")
    idx = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize_features(payload["features"].float()[idx])


def reorder_features_by_image_id(source_payload: dict, target_image_ids: List[str]) -> torch.Tensor:
    by_image = {image_id: idx for idx, image_id in enumerate(source_payload["image_ids"])}
    missing = [image_id for image_id in target_image_ids if image_id not in by_image]
    if missing:
        raise RuntimeError(f"{len(missing)} image ids missing from visual query features; first={missing[:10]}")
    indices = torch.tensor([by_image[image_id] for image_id in target_image_ids], dtype=torch.long)
    return normalize_features(source_payload["features"].float()[indices])


def exclusion_set(path: Optional[Path], exclude_genera: bool) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    classes = set(load_classes(path, []))
    genera = {genus(name) for name in classes} if exclude_genera else set()
    return classes, genera


def filter_train_features(train_payload: dict, exclude_classes: set[str], exclude_genera: set[str]) -> tuple[torch.Tensor, List[str]]:
    labels = list(train_payload["labels"])
    keep = []
    kept_labels = []
    for idx, label in enumerate(labels):
        if label in exclude_classes:
            continue
        if exclude_genera and genus(label) in exclude_genera:
            continue
        keep.append(idx)
        kept_labels.append(label)
    if not keep:
        raise RuntimeError("No train features left after exclusions")
    keep_t = torch.tensor(keep, dtype=torch.long)
    return normalize_features(train_payload["features"].float()[keep_t]), kept_labels


def aggregate_genus_scores(
    *,
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: List[str],
    candidate_genera: List[str],
    top_indices: torch.Tensor,
    neighbor_topk: int,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    train_genera = [genus(label) for label in train_labels]
    train_features_gpu = train_features.to(device=device, dtype=torch.float16)
    query_features = query_features.float()
    neighbor_topk = min(neighbor_topk, train_features.shape[0])
    out = torch.zeros(top_indices.shape, dtype=torch.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, query_features.shape[0], batch_size), desc="visual_genus_scores"):
            end = min(start + batch_size, query_features.shape[0])
            q = query_features[start:end].to(device=device, dtype=torch.float16)
            sims = q @ train_features_gpu.T
            values, indices = sims.topk(neighbor_topk, dim=1)
            values = values.float().cpu()
            indices = indices.cpu()
            for local_idx in range(end - start):
                by_genus: dict[str, list[float]] = defaultdict(list)
                for value, train_idx in zip(values[local_idx].tolist(), indices[local_idx].tolist()):
                    by_genus[train_genera[int(train_idx)]].append(float(value))
                if mode == "max":
                    genus_score = {key: max(items) for key, items in by_genus.items()}
                elif mode == "mean":
                    genus_score = {key: sum(items) / len(items) for key, items in by_genus.items()}
                elif mode == "sum":
                    genus_score = {key: sum(items) for key, items in by_genus.items()}
                elif mode == "count":
                    genus_score = {key: float(len(items)) for key, items in by_genus.items()}
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                row_idx = start + local_idx
                for col, candidate_idx in enumerate(top_indices[row_idx].tolist()):
                    out[row_idx, col] = genus_score.get(candidate_genera[int(candidate_idx)], 0.0)
    return row_zscore(out)


def compute_ranks(indices: torch.Tensor, labels: List[str], candidates: List[str], miss_rank: int) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    ranks = []
    missing = 0
    for row_idx, label in enumerate(labels):
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            missing += 1
            continue
        hit = (indices[row_idx] == true_idx).nonzero(as_tuple=False)
        ranks.append(int(hit[0, 0].item()) + 1 if hit.numel() else miss_rank)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    return {
        "rank_known": len(ranks),
        "missing_labels": missing,
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "median_rank": float(ranks_t.float().median().item()),
        "mean_rank": float(ranks_t.float().mean().item()),
    }


def write_predictions(path: Path, image_ids: List[str], indices: torch.Tensor, candidates: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred_idx in zip(image_ids, indices[:, 0].tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(pred_idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image-features", type=Path, required=True)
    parser.add_argument("--visual-query-features", type=Path, required=True)
    parser.add_argument("--visual-train-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", type=Path, default=None)
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--exclude-classes", type=Path, default=None)
    parser.add_argument("--exclude-genera", action="store_true")
    parser.add_argument("--extra-weight-grid", default="0")
    parser.add_argument("--genus-weight-grid", default="0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--neighbor-topk", type=int, default=256)
    parser.add_argument("--genus-mode", choices=["max", "mean", "sum", "count"], default="max")
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-best-predictions", action="store_true")
    args = parser.parse_args()

    base_payload = torch.load(args.base_image_features, map_location="cpu", weights_only=False)
    base_image_features = normalize_features(base_payload["features"].float())
    labels = list(base_payload.get("labels", []))
    image_ids = list(base_payload.get("image_ids", []))
    visual_query_payload = torch.load(args.visual_query_features, map_location="cpu", weights_only=False)
    visual_query_features = reorder_features_by_image_id(visual_query_payload, image_ids)
    visual_train_payload = torch.load(args.visual_train_features, map_location="cpu", weights_only=False)
    exclude_classes, exclude_genera = exclusion_set(args.exclude_classes, args.exclude_genera)
    visual_train_features, visual_train_labels = filter_train_features(
        visual_train_payload,
        exclude_classes=exclude_classes,
        exclude_genera=exclude_genera,
    )

    base_text_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(base_text_payload["classes"]))
    candidate_genera = [genus(name) for name in candidates]
    base_text_features = load_candidate_features(args.base_text_features, candidates)
    extra_text_features = load_candidate_features(args.extra_text_features, candidates) if args.extra_text_features else None

    base_logits = base_image_features @ base_text_features.T
    if args.score_normalization == "zscore":
        base_logits = row_zscore(base_logits)
    extra_logits = None
    if extra_text_features is not None:
        extra_logits = base_image_features @ extra_text_features.T
        if args.score_normalization == "zscore":
            extra_logits = row_zscore(extra_logits)

    rows = []
    best = None
    best_key = None
    best_indices = None
    best_scores = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for extra_weight in parse_grid(args.extra_weight_grid):
        if extra_logits is None:
            if extra_weight != 0:
                raise ValueError("extra weight must be 0 without extra text features")
            combined = base_logits
        else:
            combined = base_logits * (1.0 - extra_weight) + extra_logits * extra_weight
        k = min(args.topk, combined.shape[1])
        top_scores, top_indices = combined.topk(k, dim=1)
        genus_scores = aggregate_genus_scores(
            query_features=visual_query_features,
            train_features=visual_train_features,
            train_labels=visual_train_labels,
            candidate_genera=candidate_genera,
            top_indices=top_indices,
            neighbor_topk=args.neighbor_topk,
            mode=args.genus_mode,
            batch_size=args.batch_size,
            device=device,
        )
        for genus_weight in parse_grid(args.genus_weight_grid):
            final_scores = top_scores + genus_weight * genus_scores
            order = final_scores.argsort(dim=1, descending=True)
            reranked_indices = torch.gather(top_indices, 1, order)
            reranked_scores = torch.gather(final_scores, 1, order)
            metrics = compute_ranks(reranked_indices, labels, candidates, miss_rank=k + 1)
            row = {
                "extra_weight": extra_weight,
                "genus_weight": genus_weight,
                "genus_mode": args.genus_mode,
                "neighbor_topk": args.neighbor_topk,
                "topk": k,
                **metrics,
            }
            rows.append(row)
            if metrics:
                key = (row["top1"], row["top5"], row["mrr"], -abs(genus_weight))
            else:
                key = (-abs(genus_weight), -abs(extra_weight))
            if best_key is None or key > best_key:
                best_key = key
                best = row
                best_indices = reranked_indices
                best_scores = reranked_scores

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.write_best_predictions and best_indices is not None:
        write_predictions(args.out_dir / "best_predictions.csv", image_ids, best_indices, candidates)
        torch.save({"indices": best_indices, "scores": best_scores, "candidates": candidates, "image_ids": image_ids}, args.out_dir / "best_topk.pt")
    summary = {
        "base_image_features": str(args.base_image_features),
        "visual_query_features": str(args.visual_query_features),
        "visual_train_features": str(args.visual_train_features),
        "base_text_features": str(args.base_text_features),
        "extra_text_features": str(args.extra_text_features) if args.extra_text_features else None,
        "candidate_classes": str(args.candidate_classes) if args.candidate_classes else None,
        "candidate_count": len(candidates),
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_genera": args.exclude_genera,
        "visual_train_rows_after_exclusion": len(visual_train_labels),
        "rows": len(labels),
        "score_normalization": args.score_normalization,
        "topk": args.topk,
        "best": best,
        "out_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
