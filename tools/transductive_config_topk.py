from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from transductive_active_sinkhorn import class_prior, compute_logits, load_classes, load_text_features, normalize, row_zscore, sinkhorn


def write_topk(path: Path, image_ids: list[str], candidates: list[str], scores: torch.Tensor, topk: int) -> None:
    k = min(topk, scores.shape[1])
    top_scores, top_indices = scores.topk(k, dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["image_id", "prediction", "margin_top1_top2", "top_classes", "top_scores"],
        )
        writer.writeheader()
        for image_id, indices, values in zip(image_ids, top_indices.tolist(), top_scores.tolist()):
            classes = [candidates[int(idx)] for idx in indices]
            margin = float(values[0] - values[1]) if len(values) > 1 else 0.0
            writer.writerow(
                {
                    "image_id": image_id,
                    "prediction": classes[0],
                    "margin_top1_top2": margin,
                    "top_classes": "|".join(classes),
                    "top_scores": "|".join(f"{float(value):.8f}" for value in values),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.04)
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--prior-mode", default="logsumexp")
    parser.add_argument("--prior-alpha", type=float, default=0.25)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.98)
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
    logits = compute_logits(image_features, text_features, args.score_batch_size, device).to(device)
    if args.blend == 0:
        final = logits
    else:
        prior = class_prior(logits, args.prior_mode, args.prior_alpha, args.prior_uniform_mix)
        balanced = sinkhorn(logits, tau=args.tau, iters=args.sinkhorn_iters, prior=prior)
        final = row_zscore(logits) + args.blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))
    final = final.cpu()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_topk(args.out_dir / "topk.csv", image_ids, candidates, final, args.topk)
    with (args.out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        pred = final.argmax(dim=1).tolist()
        for image_id, idx in zip(image_ids, pred):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(idx)]})
    summary = {
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "candidate_classes": str(args.candidate_classes),
        "rows": len(image_ids),
        "labels": int(sum(bool(label) for label in labels)),
        "topk_csv": str(args.out_dir / "topk.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
        "config": {
            "tau": args.tau,
            "blend": args.blend,
            "prior_mode": args.prior_mode,
            "prior_alpha": args.prior_alpha,
            "prior_uniform_mix": args.prior_uniform_mix,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
