from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from train_topk_meta_reranker import (
    build_features,
    load_split,
    metrics,
    parse_grid,
    parse_split,
    row_zscore,
    standardize,
    target_indices,
    train_model,
)


def write_predictions(path: Path, split: dict[str, Any], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, image_id in enumerate(split["image_ids"]):
            preds = split["predictions"][row_idx]
            pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": image_id,
                    "prediction": pred,
                    "base_prediction": preds[0],
                    "changed": pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", action="append", required=True, help="name=score1.pt|score2.pt")
    parser.add_argument("--apply-split", required=True, help="public=score1.pt|score2.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=8e-2)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--rerank-weight", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_splits = [load_split(*parse_split(value)) for value in args.train_split]
    apply_split = load_split(*parse_split(args.apply_split))

    train_features = {split["name"]: build_features(split) for split in train_splits}
    train_targets = {split["name"]: target_indices(split) for split in train_splits}
    x_train = torch.cat([train_features[split["name"]] for split in train_splits], dim=0)
    y_train = torch.cat([train_targets[split["name"]] for split in train_splits], dim=0)
    x_apply = build_features(apply_split)
    x_train_std, x_apply_std, mean = standardize(x_train, x_apply)
    model, losses = train_model(
        x_train_std,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    device = next(model.parameters()).device
    with torch.inference_mode():
        meta_scores = model(x_apply_std.to(device)).cpu()
    final_scores = apply_split["base_scores"] + args.rerank_weight * row_zscore(meta_scores)
    indices = final_scores.argsort(dim=1, descending=True)
    setattr(indices, "triggered", len(apply_split["image_ids"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / "predictions.csv", apply_split, indices)
    torch.save(
        {
            "image_ids": apply_split["image_ids"],
            "predictions": apply_split["predictions"],
            "base_scores": apply_split["base_scores"],
            "adapter_scores": apply_split["adapter_scores"],
            "meta_scores": meta_scores,
            "final_scores": final_scores,
            "labels": apply_split["labels"],
        },
        args.out_dir / "meta_topk_scores.pt",
    )
    summary = {
        "train_splits": [{"name": split["name"], "paths": split["paths"], "rows": len(split["labels"])} for split in train_splits],
        "apply_split": {"name": apply_split["name"], "paths": apply_split["paths"], "rows": len(apply_split["labels"])},
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "rerank_weight": args.rerank_weight,
        "losses": losses,
        "changed": int((indices[:, 0] != 0).sum().item()),
        "metrics_if_labeled": metrics(indices, apply_split) if any(apply_split["labels"]) else {},
        "predictions_csv": str(args.out_dir / "predictions.csv"),
        "score_file": str(args.out_dir / "meta_topk_scores.pt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
