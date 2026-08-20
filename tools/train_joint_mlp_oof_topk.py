from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_targets(payload: dict[str, Any], classes: list[str]) -> torch.Tensor:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    if "labels" in payload:
        labels = list(payload["labels"])
        missing = [label for label in labels if label not in label_to_idx]
        if missing:
            raise RuntimeError(f"labels missing from classes; first={missing[:5]}")
        targets = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)
        if "class_ids" in payload:
            class_ids = torch.as_tensor(payload["class_ids"], dtype=torch.long)
            if not torch.equal(targets, class_ids):
                raise RuntimeError("payload labels and class_ids disagree")
        return targets
    if "class_ids" not in payload:
        raise KeyError("train cache must contain labels or class_ids")
    return torch.as_tensor(payload["class_ids"], dtype=torch.long)


def make_stratified_folds(targets: torch.Tensor, folds: int, seed: int) -> torch.Tensor:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    result = torch.full((targets.numel(),), -1, dtype=torch.long)
    class_count = int(targets.max().item()) + 1
    too_small: list[tuple[int, int]] = []
    for class_id in range(class_count):
        indices = (targets == class_id).nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            too_small.append((class_id, int(indices.numel())))
            continue
        generator = torch.Generator().manual_seed(seed + class_id * 104729)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        offset = (seed + class_id * 9973) % folds
        assignments = (torch.arange(indices.numel()) + offset) % folds
        result[indices] = assignments
    if too_small:
        raise RuntimeError(
            "strict OOF requires at least two train images per class; "
            f"found {len(too_small)} undersized classes, first={too_small[:10]}"
        )
    if bool((result < 0).any()):
        raise RuntimeError("some rows were not assigned to a fold")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fold(
    fold: int,
    x: torch.Tensor,
    y: torch.Tensor,
    fold_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = fold_dir / "heldout_topk.pt"
    summary_path = fold_dir / "summary.json"
    if args.resume and prediction_path.exists() and summary_path.exists():
        saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return saved["top_scores"], saved["top_indices"], summary

    train_indices = (fold_ids != fold).nonzero(as_tuple=False).flatten()
    heldout_indices = (fold_ids == fold).nonzero(as_tuple=False).flatten()
    fold_seed = args.seed + fold * 100003
    seed_everything(fold_seed)
    model = MLPClassifier(x.shape[1], args.hidden_dim, len(args.classes), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    checkpoint_path = fold_dir / "last.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    metrics_path = fold_dir / "train_metrics.csv"
    existing_rows: list[dict[str, Any]] = []
    if metrics_path.exists() and start_epoch > 1:
        with metrics_path.open("r", encoding="utf-8", newline="") as fp:
            existing_rows = list(csv.DictReader(fp))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(fold_seed + epoch * 1009)
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        loss_sum = 0.0
        seen = 0
        for start in range(0, order.numel(), args.batch_size):
            rows = order[start : start + args.batch_size]
            xb = x[rows].to(device, non_blocking=True)
            yb = y[rows].to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * rows.numel()
            seen += rows.numel()
        row = {"fold": fold, "epoch": epoch, "train_loss": loss_sum / max(1, seen)}
        existing_rows.append(row)
        print(json.dumps(row), flush=True)
        with metrics_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["fold", "epoch", "train_loss"])
            writer.writeheader()
            writer.writerows(existing_rows)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "fold": fold,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    model.eval()
    score_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, heldout_indices.numel(), args.eval_batch_size):
            rows = heldout_indices[start : start + args.eval_batch_size]
            logits = model(x[rows].to(device, non_blocking=True))
            scores, indices = logits.topk(args.topk, dim=1)
            score_chunks.append(scores.half().cpu())
            index_chunks.append(indices.int().cpu())
    top_scores = torch.cat(score_chunks, dim=0)
    top_indices = torch.cat(index_chunks, dim=0)
    heldout_targets = y[heldout_indices]
    top1_correct = top_indices[:, 0].long().eq(heldout_targets)
    topk_correct = top_indices.long().eq(heldout_targets[:, None]).any(dim=1)
    summary = {
        "fold": fold,
        "train_rows": int(train_indices.numel()),
        "heldout_rows": int(heldout_indices.numel()),
        "epochs": args.epochs,
        "top1_correct": int(top1_correct.sum()),
        "topk_correct": int(topk_correct.sum()),
        "top1": float(top1_correct.float().mean()),
        f"top{args.topk}": float(topk_correct.float().mean()),
        "oracle_complement": int((topk_correct & ~top1_correct).sum()),
    }
    torch.save(
        {
            "row_indices": heldout_indices,
            "top_scores": top_scores,
            "top_indices": top_indices,
            "targets": heldout_targets,
        },
        prediction_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # The fold is complete. Replace the much larger optimizer checkpoint with a
    # lean reproducibility checkpoint while retaining the exact final weights.
    torch.save(
        {
            "epoch": args.epochs,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "fold": fold,
            "arch": {
                "in_dim": x.shape[1],
                "hidden_dim": args.hidden_dim,
                "out_dim": len(args.classes),
                "dropout": args.dropout,
            },
        },
        checkpoint_path,
    )
    return top_scores, top_indices, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict stratified OOF joint MLPs and save held-out top-k logits.")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    args = parser.parse_args()
    if args.topk < 1:
        raise ValueError("topk must be positive")
    torch.set_float32_matmul_precision(args.matmul_precision)
    seed_everything(args.seed)
    payload = load_payload(args.train_cache)
    classes = list(payload["classes"])
    args.classes = classes
    x = F.normalize(torch.as_tensor(payload["features"]).float(), dim=1)
    y = get_targets(payload, classes)
    image_ids = list(payload["image_ids"])
    if x.shape[0] != y.numel() or x.shape[0] != len(image_ids):
        raise RuntimeError("feature, target, and image_id row counts disagree")
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError("non-finite feature values detected")
    if int(y.min()) < 0 or int(y.max()) >= len(classes):
        raise RuntimeError("target IDs are outside the class list")
    fold_ids = make_stratified_folds(y, args.folds, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_counts = torch.bincount(fold_ids, minlength=args.folds)
    assignment = {
        "rows": int(x.shape[0]),
        "classes": len(classes),
        "folds": args.folds,
        "fold_counts": fold_counts.tolist(),
        "seed": args.seed,
    }
    (args.out_dir / "fold_assignment_summary.json").write_text(
        json.dumps(assignment, indent=2) + "\n", encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_scores = torch.empty((x.shape[0], args.topk), dtype=torch.float16)
    oof_indices = torch.empty((x.shape[0], args.topk), dtype=torch.int32)
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(args.folds):
        scores, indices, summary = train_fold(fold, x, y, fold_ids, args, device)
        rows = (fold_ids == fold).nonzero(as_tuple=False).flatten()
        if scores.shape[0] != rows.numel():
            raise RuntimeError(f"fold {fold} prediction row count mismatch")
        oof_scores[rows] = scores
        oof_indices[rows] = indices
        fold_summaries.append(summary)

    top1_correct = oof_indices[:, 0].long().eq(y)
    topk_correct = oof_indices.long().eq(y[:, None]).any(dim=1)
    output_path = args.out_dir / "oof_topk.pt"
    torch.save(
        {
            "top_scores": oof_scores,
            "top_indices": oof_indices,
            "class_ids": y,
            "fold_ids": fold_ids,
            "labels": [classes[int(v)] for v in y.tolist()],
            "image_ids": image_ids,
            "classes": classes,
            "source_cache": str(args.train_cache),
            "strict_oof": True,
            "fixed_epochs": args.epochs,
        },
        output_path,
    )
    summary = {
        **assignment,
        "source_cache": str(args.train_cache),
        "output": str(output_path),
        "fixed_epochs": args.epochs,
        "top1_correct": int(top1_correct.sum()),
        "topk_correct": int(topk_correct.sum()),
        "top1": float(top1_correct.float().mean()),
        f"top{args.topk}": float(topk_correct.float().mean()),
        "oracle_complement": int((topk_correct & ~top1_correct).sum()),
        "fold_summaries": fold_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
