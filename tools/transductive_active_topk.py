from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transductive_active_sinkhorn import (
    active_indices,
    class_prior,
    compute_logits,
    load_classes,
    load_text_features,
    normalize,
    row_zscore,
    sinkhorn,
)


def write_topk_jsonl(
    path: Path,
    image_ids: list[str],
    labels: list[str],
    candidates: list[str],
    scores: torch.Tensor,
    topk: int,
) -> None:
    k = min(topk, scores.shape[1])
    values, indices = scores.topk(k, dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for image_id, label, row_indices, row_values in zip(
            image_ids,
            labels,
            indices.tolist(),
            values.tolist(),
        ):
            preds = [candidates[int(idx)] for idx in row_indices]
            row = {
                "image_id": image_id,
                "label": label,
                "predictions": preds,
                "scores": [float(value) for value in row_values],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--active-count", type=int, default=11598)
    parser.add_argument("--active-mode", default="max")
    parser.add_argument("--union-topk", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.04)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--prior-mode", default="logsumexp")
    parser.add_argument("--prior-alpha", type=float, default=0.25)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.95)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))
    image_features = normalize(image_payload["features"])
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_features = load_text_features(args.text_features, candidates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = compute_logits(image_features, text_features, args.score_batch_size, device)
    active = active_indices(logits.clone(), args.active_mode, args.active_count, args.union_topk)
    active_logits = logits[:, active].to(device)
    if args.blend == 0:
        final = active_logits
    else:
        prior = class_prior(active_logits, args.prior_mode, args.prior_alpha, args.prior_uniform_mix)
        balanced = sinkhorn(active_logits, tau=args.tau, iters=args.sinkhorn_iters, prior=prior)
        final = row_zscore(active_logits) + args.blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))
    final = final.cpu()
    active_candidates = [candidates[int(idx)] for idx in active.tolist()]
    write_topk_jsonl(args.out_jsonl, image_ids, labels, active_candidates, final, args.topk)
    summary = {
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "candidate_classes": str(args.candidate_classes),
        "out_jsonl": str(args.out_jsonl),
        "rows": len(image_ids),
        "active_actual": len(active_candidates),
        "topk": args.topk,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
