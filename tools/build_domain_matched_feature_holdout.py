from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_balanced_feature_holdout import (
    frequency_bucket,
    load_classes,
    merge_payloads,
    parse_paths,
    subset_payload,
)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def predict(
    model: nn.Module,
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits = model(features[batch_indices].to(device))
            chunks.append(logits.sigmoid().cpu())
    return torch.cat(chunks)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    points = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    result = torch.quantile(values.float(), points)
    return {str(float(point)): float(value) for point, value in zip(points, result)}


def train_oof_domain_scores(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    folds: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    source_count = source_features.shape[0]
    all_features = torch.cat([source_features, target_features], dim=0)
    all_targets = torch.cat(
        [torch.zeros(source_count), torch.ones(target_features.shape[0])], dim=0
    )
    generator = torch.Generator().manual_seed(seed)
    source_folds = torch.empty(source_count, dtype=torch.long)
    source_order = torch.randperm(source_count, generator=generator)
    source_folds[source_order] = torch.arange(source_count) % folds
    target_folds = torch.empty(target_features.shape[0], dtype=torch.long)
    target_order = torch.randperm(target_features.shape[0], generator=generator)
    target_folds[target_order] = torch.arange(target_features.shape[0]) % folds
    all_folds = torch.cat([source_folds, target_folds], dim=0)

    oof_scores = torch.empty(all_features.shape[0], dtype=torch.float32)
    fold_rows = []
    for fold in range(folds):
        torch.manual_seed(seed + fold)
        train_indices = (all_folds != fold).nonzero(as_tuple=False).squeeze(1)
        val_indices = (all_folds == fold).nonzero(as_tuple=False).squeeze(1)
        model = DomainClassifier(all_features.shape[1], hidden_dim, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_targets = all_targets[train_indices]
        positive_count = train_targets.sum().clamp_min(1.0)
        negative_count = (train_targets.numel() - positive_count).clamp_min(1.0)
        pos_weight = (negative_count / positive_count).to(device)
        loader = DataLoader(
            TensorDataset(all_features[train_indices], train_targets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        for _epoch in range(epochs):
            model.train()
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                logits = model(features)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        scores = predict(model, all_features, val_indices, batch_size, device)
        oof_scores[val_indices] = scores
        val_targets = all_targets[val_indices]
        auc = roc_auc_score(val_targets.numpy(), scores.numpy())
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(val_indices.numel()),
                "source_rows": int((val_targets == 0).sum().item()),
                "target_rows": int((val_targets == 1).sum().item()),
                "roc_auc": float(auc),
            }
        )
    return oof_scores[:source_count], oof_scores[source_count:], fold_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-caches", required=True, help="Comma-separated labeled caches")
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    source = merge_payloads(parse_paths(args.source_caches))
    target = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    source_features = F.normalize(source["features"].float(), dim=1)
    target_features = F.normalize(target["features"].float(), dim=1)
    if source_features.shape[1] != target_features.shape[1]:
        raise RuntimeError("Source and target feature dimensions differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_scores, target_scores, fold_rows = train_oof_domain_scores(
        source_features,
        target_features,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    classes = load_classes(args.classes, source["labels"])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    rows_by_class: dict[str, list[int]] = defaultdict(list)
    for row_idx, label in enumerate(source["labels"]):
        rows_by_class[label].append(row_idx)
    full_counts = Counter({name: len(indices) for name, indices in rows_by_class.items()})
    missing = [name for name in classes if len(rows_by_class.get(name, [])) < 2]
    if missing:
        raise RuntimeError(f"{len(missing)} classes have fewer than two source rows")
    val_indices = [
        max(rows_by_class[name], key=lambda idx: (float(source_scores[idx]), -idx))
        for name in classes
    ]
    val_set = set(val_indices)
    train_indices = [idx for idx in range(len(source["labels"])) if idx not in val_set]
    val_indices.sort()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = subset_payload(
        source, train_indices, classes, class_to_idx, full_counts, "domain_matched_train"
    )
    val_payload = subset_payload(
        source, val_indices, classes, class_to_idx, full_counts, "domain_matched_val"
    )
    val_payload["public_likeness"] = source_scores[torch.tensor(val_indices)]
    torch.save(train_payload, args.out_dir / "train.pt")
    torch.save(val_payload, args.out_dir / "val.pt")
    torch.save(
        {
            "source_image_ids": source["image_ids"],
            "source_scores": source_scores,
            "target_image_ids": list(target["image_ids"]),
            "target_scores": target_scores,
            "folds": fold_rows,
        },
        args.out_dir / "domain_scores.pt",
    )

    class_buckets = Counter(frequency_bucket(count) for count in full_counts.values())
    summary: dict[str, Any] = {
        "source_caches": [str(path) for path in parse_paths(args.source_caches)],
        "target_cache": str(args.target_cache),
        "source_rows": source_features.shape[0],
        "target_rows": target_features.shape[0],
        "feature_dim": source_features.shape[1],
        "domain_hidden_dim": args.hidden_dim,
        "folds": fold_rows,
        "mean_roc_auc": sum(row["roc_auc"] for row in fold_rows) / len(fold_rows),
        "source_score_quantiles": quantiles(source_scores),
        "target_score_quantiles": quantiles(target_scores),
        "selected_val_score_quantiles": quantiles(source_scores[torch.tensor(val_indices)]),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "classes": len(classes),
        "class_frequency_buckets": dict(sorted(class_buckets.items())),
        "train_cache": str(args.out_dir / "train.pt"),
        "val_cache": str(args.out_dir / "val.pt"),
        "domain_scores": str(args.out_dir / "domain_scores.pt"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
