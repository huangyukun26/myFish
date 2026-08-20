from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from transductive_active_sinkhorn import (
    active_indices,
    class_prior,
    compute_ensemble_logits,
    compute_logits,
    load_classes,
    load_text_features,
    normalize,
    parse_paths,
    parse_weights,
    row_zscore,
    sinkhorn,
)


def topk_metrics(top_indices: torch.Tensor, labels: list[str], candidates: list[str]) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    ranks: list[int] = []
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        row = top_indices[row_idx].tolist()
        ranks.append(row.index(true_idx) + 1 if true_idx in row else len(row) + 1)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    return {
        "known": len(ranks),
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "median_rank": float(ranks_t.float().median().item()),
        "mean_rank": float(ranks_t.float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", default="")
    parser.add_argument("--text-weights", default="")
    parser.add_argument("--logit-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--active-count", type=int, default=11598)
    parser.add_argument("--active-mode", default="max")
    parser.add_argument("--union-topk", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--blend", type=float, default=5.0)
    parser.add_argument("--prior-mode", default="logsumexp")
    parser.add_argument("--prior-alpha", type=float, default=0.5)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.95)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))
    image_features = normalize(image_payload["features"])

    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_paths = [args.text_features] + parse_paths(args.extra_text_features)
    text_weights = parse_weights(args.text_weights, len(text_paths))
    text_features = [load_text_features(path, candidates) for path in text_paths]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(text_features) == 1:
        logits = compute_logits(image_features, text_features[0], args.score_batch_size, device)
    else:
        logits = compute_ensemble_logits(
            image_features,
            text_features,
            text_weights,
            args.score_batch_size,
            device,
            args.logit_normalization,
        )

    active = active_indices(logits.clone(), args.active_mode, args.active_count, args.union_topk)
    active_logits = logits[:, active].to(device)
    prior = class_prior(active_logits, args.prior_mode, args.prior_alpha, args.prior_uniform_mix)
    balanced = sinkhorn(active_logits, tau=args.tau, iters=args.sinkhorn_iters, prior=prior)
    active_scores = row_zscore(active_logits) + args.blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))
    active_scores = active_scores.cpu()

    k = min(args.topk, active_scores.shape[1])
    top_scores, top_pos = active_scores.topk(k, dim=1)
    top_indices = active[top_pos]
    pred_indices = top_indices[:, 0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred_idx in zip(image_ids, pred_indices.tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(pred_idx)]})

    topk_path = args.out_dir / "topk.csv"
    with topk_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image_id",
                "label",
                "prediction",
                "margin_top1_top2",
                "top_classes",
                "top_scores",
            ],
        )
        writer.writeheader()
        for row_idx, image_id in enumerate(image_ids):
            cls = [candidates[int(idx)] for idx in top_indices[row_idx].tolist()]
            scores = [float(v) for v in top_scores[row_idx].tolist()]
            margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
            writer.writerow(
                {
                    "image_id": image_id,
                    "label": labels[row_idx] if row_idx < len(labels) else "",
                    "prediction": cls[0],
                    "margin_top1_top2": margin,
                    "top_classes": "|".join(cls),
                    "top_scores": "|".join(f"{score:.7g}" for score in scores),
                }
            )

    torch.save(
        {
            "image_ids": image_ids,
            "labels": labels,
            "candidates": candidates,
            "active_indices": active,
            "top_indices": top_indices,
            "top_scores": top_scores,
            "best": {
                "active_count": int(args.active_count),
                "active_actual": int(len(active)),
                "active_mode": args.active_mode,
                "union_topk": int(args.union_topk),
                "tau": float(args.tau),
                "blend": float(args.blend),
                "prior_mode": args.prior_mode,
                "prior_alpha": float(args.prior_alpha),
                "prior_uniform_mix": float(args.prior_uniform_mix),
            },
        },
        args.out_dir / "topk_scores.pt",
    )

    summary = {
        "image_features": str(args.image_features),
        "text_features": [str(path) for path in text_paths],
        "text_weights": text_weights,
        "candidate_classes": str(args.candidate_classes) if args.candidate_classes else None,
        "rows": len(image_ids),
        "candidates": len(candidates),
        "best": {
            "active_count": int(args.active_count),
            "active_actual": int(len(active)),
            "active_mode": args.active_mode,
            "union_topk": int(args.union_topk),
            "tau": float(args.tau),
            "blend": float(args.blend),
            "prior_mode": args.prior_mode,
            "prior_alpha": float(args.prior_alpha),
            "prior_uniform_mix": float(args.prior_uniform_mix),
        },
        "metrics_if_labeled": topk_metrics(top_indices, labels, candidates),
        "predictions_csv": str(predictions_path),
        "topk_csv": str(topk_path),
        "score_file": str(args.out_dir / "topk_scores.pt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
