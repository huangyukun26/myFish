from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import torch


def load_classes(path: Optional[Path], text_classes: List[str]) -> List[str]:
    if path is None:
        return text_classes
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def row_zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_candidate_features(path: Path, candidates: List[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidates missing from {path}; first={missing[:10]}")
    idx = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize_features(payload["features"].float()[idx])


def compute_metrics(logits: torch.Tensor, labels: List[str], candidates: List[str], topk: int) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    k = min(topk, len(candidates))
    top_indices = logits.topk(k, dim=1).indices
    ranks = []
    missing = 0
    for row_index, label in enumerate(labels):
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            missing += 1
            continue
        true_score = logits[row_index, true_idx]
        rank = int((logits[row_index] > true_score).sum().item()) + 1
        ranks.append(rank)
    if not ranks:
        raise RuntimeError("No labels were present in candidate classes")
    ranks_tensor = torch.tensor(ranks)
    return {
        "rank_known": len(ranks),
        "missing_labels": missing,
        "top1": float((ranks_tensor <= 1).float().mean().item()),
        "top5": float((ranks_tensor <= 5).float().mean().item()),
        f"top{topk}": float((ranks_tensor <= topk).float().mean().item()),
        "mrr": float((1.0 / ranks_tensor.float()).mean().item()),
        "median_rank": float(ranks_tensor.float().median().item()),
        "mean_rank": float(ranks_tensor.float().mean().item()),
    }


def compute_topk_rerank_metrics(
    base_logits: torch.Tensor,
    rerank_logits: torch.Tensor,
    labels: List[str],
    candidates: List[str],
    first_stage_topk: int,
    rerank_weight: float,
    margin_threshold: Optional[float] = None,
) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    k = min(first_stage_topk, len(candidates))
    base_scores, base_indices = base_logits.topk(k, dim=1)
    desc_scores = torch.gather(rerank_logits, 1, base_indices)
    desc_scores = row_zscore(desc_scores)
    final_scores = base_scores + rerank_weight * desc_scores
    if margin_threshold is not None and k > 1:
        margins = base_scores[:, 0] - base_scores[:, 1]
        should_rerank = margins <= margin_threshold
        final_scores = torch.where(should_rerank[:, None], final_scores, base_scores)
    reranked_order = final_scores.argsort(dim=1, descending=True)
    reranked_indices = torch.gather(base_indices, 1, reranked_order)
    ranks = []
    missing = 0
    for row_index, label in enumerate(labels):
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            missing += 1
            continue
        matches = (reranked_indices[row_index] == true_idx).nonzero(as_tuple=False)
        if matches.numel() == 0:
            ranks.append(first_stage_topk + 1)
        else:
            ranks.append(int(matches[0, 0].item()) + 1)
    ranks_tensor = torch.tensor(ranks)
    result = {
        "rank_known": len(ranks),
        "missing_labels": missing,
        "top1": float((ranks_tensor <= 1).float().mean().item()),
        "top5": float((ranks_tensor <= 5).float().mean().item()),
        f"top{first_stage_topk}": float((ranks_tensor <= first_stage_topk).float().mean().item()),
        "mrr": float((1.0 / ranks_tensor.float()).mean().item()),
        "median_rank": float(ranks_tensor.float().median().item()),
        "mean_rank": float(ranks_tensor.float().mean().item()),
    }
    if margin_threshold is not None and k > 1:
        result["rerank_triggered"] = int(should_rerank.sum().item())
        result["rerank_triggered_frac"] = float(should_rerank.float().mean().item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", type=Path, default=None)
    parser.add_argument("--rerank-text-features", type=Path, default=None)
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--extra-weight-grid", default="0")
    parser.add_argument("--rerank-weight-grid", default="0")
    parser.add_argument(
        "--rerank-margin-grid",
        default="",
        help="Optional top1-top2 margin thresholds for conditional topK rerank, e.g. '0.05,0.1,0.2'.",
    )
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_features = normalize_features(image_payload["features"].float())
    labels = list(image_payload.get("labels", []))
    base_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(base_payload["classes"]))
    base_features = load_candidate_features(args.base_text_features, candidates)
    extra_features = load_candidate_features(args.extra_text_features, candidates) if args.extra_text_features else None
    rerank_features = load_candidate_features(args.rerank_text_features, candidates) if args.rerank_text_features else None

    base_logits = image_features @ base_features.T
    if args.score_normalization == "zscore":
        base_logits = row_zscore(base_logits)
    extra_logits = None
    if extra_features is not None:
        extra_logits = image_features @ extra_features.T
        if args.score_normalization == "zscore":
            extra_logits = row_zscore(extra_logits)
    rerank_logits = None
    if rerank_features is not None:
        rerank_logits = image_features @ rerank_features.T

    rows = []
    extra_weights = parse_grid(args.extra_weight_grid)
    rerank_weights = parse_grid(args.rerank_weight_grid)
    rerank_margins = parse_grid(args.rerank_margin_grid) if args.rerank_margin_grid.strip() else []
    for extra_weight in extra_weights:
        if extra_logits is None:
            if extra_weight != 0:
                raise ValueError("--extra-weight-grid must be 0 when --extra-text-features is absent")
            combined = base_logits
        else:
            combined = base_logits * (1.0 - extra_weight) + extra_logits * extra_weight
        base_metrics = compute_metrics(combined, labels, candidates, args.topk)
        rows.append(
            {
                "mode": "base",
                "extra_weight": extra_weight,
                "rerank_weight": 0.0,
                "margin_threshold": "",
                "rerank_triggered": "",
                "rerank_triggered_frac": "",
                **base_metrics,
            }
        )
        if rerank_logits is not None:
            for rerank_weight in rerank_weights:
                metrics = compute_topk_rerank_metrics(
                    combined,
                    rerank_logits,
                    labels,
                    candidates,
                    args.topk,
                    rerank_weight,
                )
                rows.append(
                    {
                        "mode": "topk_rerank",
                        "extra_weight": extra_weight,
                        "rerank_weight": rerank_weight,
                        "margin_threshold": "",
                        "rerank_triggered": "",
                        "rerank_triggered_frac": "",
                        **metrics,
                    }
                )
                for margin_threshold in rerank_margins:
                    metrics = compute_topk_rerank_metrics(
                        combined,
                        rerank_logits,
                        labels,
                        candidates,
                        args.topk,
                        rerank_weight,
                        margin_threshold=margin_threshold,
                    )
                    rows.append(
                        {
                            "mode": "conditional_topk_rerank",
                            "extra_weight": extra_weight,
                            "rerank_weight": rerank_weight,
                            "margin_threshold": margin_threshold,
                            **metrics,
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda item: (item["top1"], item["top5"], item[f"top{args.topk}"], item["mrr"]))
    summary = {
        "image_features": str(args.image_features),
        "base_text_features": str(args.base_text_features),
        "extra_text_features": str(args.extra_text_features) if args.extra_text_features else None,
        "rerank_text_features": str(args.rerank_text_features) if args.rerank_text_features else None,
        "candidate_classes": str(args.candidate_classes) if args.candidate_classes else None,
        "candidate_count": len(candidates),
        "rows": len(labels),
        "score_normalization": args.score_normalization,
        "topk": args.topk,
        "best": best,
        "out_csv": str(csv_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
