from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_classes(path: Optional[Path], fallback: List[str]) -> List[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def genus(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def load_candidate_features(path: Path, candidates: List[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} missing from {path}; first={missing[:10]}")
    idx = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize_features(payload["features"].float()[idx])


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


def make_genus_scores(top_scores: torch.Tensor, top_indices: torch.Tensor, candidates: List[str], mode: str) -> torch.Tensor:
    out = torch.zeros_like(top_scores)
    candidate_genera = [genus(name) for name in candidates]
    for row in range(top_indices.shape[0]):
        values_by_genus = defaultdict(list)
        for col in range(top_indices.shape[1]):
            cls_idx = int(top_indices[row, col].item())
            values_by_genus[candidate_genera[cls_idx]].append(float(top_scores[row, col].item()))
        if mode == "max":
            genus_score = {key: max(values) for key, values in values_by_genus.items()}
        elif mode == "mean":
            genus_score = {key: sum(values) / len(values) for key, values in values_by_genus.items()}
        elif mode == "sum":
            genus_score = {key: sum(values) for key, values in values_by_genus.items()}
        elif mode == "count":
            genus_score = {key: float(len(values)) for key, values in values_by_genus.items()}
        else:
            raise ValueError(f"unknown genus mode {mode}")
        for col in range(top_indices.shape[1]):
            cls_idx = int(top_indices[row, col].item())
            out[row, col] = genus_score[candidate_genera[cls_idx]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", type=Path, default=None)
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--extra-weight-grid", default="0")
    parser.add_argument("--genus-weight-grid", default="0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--genus-mode", choices=["max", "mean", "sum", "count"], default="sum")
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_features = normalize_features(image_payload["features"].float())
    labels = list(image_payload.get("labels", []))
    base_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(base_payload["classes"]))
    base_features = load_candidate_features(args.base_text_features, candidates)
    extra_features = load_candidate_features(args.extra_text_features, candidates) if args.extra_text_features else None

    base_logits = image_features @ base_features.T
    if args.score_normalization == "zscore":
        base_logits = row_zscore(base_logits)
    extra_logits = None
    if extra_features is not None:
        extra_logits = image_features @ extra_features.T
        if args.score_normalization == "zscore":
            extra_logits = row_zscore(extra_logits)

    rows = []
    best = None
    for extra_weight in parse_grid(args.extra_weight_grid):
        if extra_logits is None:
            if extra_weight != 0:
                raise ValueError("extra weight must be 0 without extra text features")
            combined = base_logits
        else:
            combined = base_logits * (1.0 - extra_weight) + extra_logits * extra_weight
        k = min(args.topk, combined.shape[1])
        top_scores, top_indices = combined.topk(k, dim=1)
        genus_scores = make_genus_scores(top_scores, top_indices, candidates, mode=args.genus_mode)
        genus_scores = row_zscore(genus_scores)
        for genus_weight in parse_grid(args.genus_weight_grid):
            final_scores = top_scores + genus_weight * genus_scores
            order = final_scores.argsort(dim=1, descending=True)
            reranked_indices = torch.gather(top_indices, 1, order)
            metrics = compute_ranks(reranked_indices, labels, candidates, miss_rank=k + 1)
            row = {
                "extra_weight": extra_weight,
                "genus_weight": genus_weight,
                "genus_mode": args.genus_mode,
                "topk": k,
                **metrics,
            }
            rows.append(row)
            if best is None or (row["top1"], row["top5"], row["mrr"]) > (best["top1"], best["top5"], best["mrr"]):
                best = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "image_features": str(args.image_features),
        "base_text_features": str(args.base_text_features),
        "extra_text_features": str(args.extra_text_features) if args.extra_text_features else None,
        "candidate_classes": str(args.candidate_classes) if args.candidate_classes else None,
        "candidate_count": len(candidates),
        "rows": len(labels),
        "topk": args.topk,
        "genus_mode": args.genus_mode,
        "score_normalization": args.score_normalization,
        "best": best,
        "out_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
