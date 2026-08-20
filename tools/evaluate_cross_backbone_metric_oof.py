from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_pair(base: dict, alternate: dict) -> torch.Tensor:
    if base["image_ids"] != alternate["image_ids"]:
        raise RuntimeError("Feature caches have different image order")
    base_features = F.normalize(base["features"].float(), dim=1)
    alternate_features = F.normalize(alternate["features"].float(), dim=1)
    return torch.cat([base_features, alternate_features], dim=1) / (2.0**0.5)


def genus_folds(labels: list[str], fold_count: int) -> torch.Tensor:
    folds = []
    for label in labels:
        genus = label.split(maxsplit=1)[0]
        digest = hashlib.sha1(genus.encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:4], "little") % fold_count)
    return torch.tensor(folds, dtype=torch.long)


class ResidualMetric(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.clamp(-1.5, 1.5).exp()
        residual = self.up(F.gelu(self.down(self.dropout(x))))
        return F.normalize(x * scale + 0.2 * residual, dim=1)


def genus_batches(indices: torch.Tensor, labels: list[str], batch_size: int, rng: random.Random) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices.tolist():
        groups[labels[index].split(maxsplit=1)[0]].append(index)
    genera = list(groups)
    rng.shuffle(genera)
    batches: list[list[int]] = []
    current: list[int] = []
    for genus in genera:
        group = groups[genus]
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def train_metric(
    support: torch.Tensor,
    query: torch.Tensor,
    labels: list[str],
    train_indices: torch.Tensor,
    *,
    rank: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    seed: int,
    device: torch.device,
) -> tuple[ResidualMetric, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ResidualMetric(support.shape[1], rank, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in genus_batches(train_indices, labels, batch_size, rng):
            batch_idx = torch.tensor(batch, dtype=torch.long)
            support_batch = support[batch_idx].to(device)
            query_batch = query[batch_idx].to(device)
            support_embedding = model(support_batch)
            query_embedding = model(query_batch)
            logits = query_embedding @ support_embedding.T / temperature
            target = torch.arange(len(batch), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_rows += len(batch)
        losses.append(total_loss / max(1, total_rows))
    return model, losses


@torch.inference_mode()
def predict_topk(
    model: nn.Module,
    support: torch.Tensor,
    query: torch.Tensor,
    device: torch.device,
    topk: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    support_embedding = model(support.to(device))
    values = []
    indices = []
    for start in range(0, len(query), 512):
        query_embedding = model(query[start : start + 512].to(device))
        batch_values, batch_indices = (query_embedding @ support_embedding.T).topk(topk, dim=1)
        values.append(batch_values.cpu())
        indices.append(batch_indices.cpu())
    return torch.cat(values), torch.cat(indices)


def metrics(prediction: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> dict:
    correct = prediction.eq(target)
    base_correct = baseline.eq(target)
    return {
        "top1": float(correct.float().mean().item()),
        "changed": int(prediction.ne(baseline).sum().item()),
        "wins": int((correct & ~base_correct).sum().item()),
        "losses": int((~correct & base_correct).sum().item()),
        "net": int((correct.sum() - base_correct.sum()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-support", type=Path, required=True)
    parser.add_argument("--base-query", type=Path, required=True)
    parser.add_argument("--alt-support", type=Path, required=True)
    parser.add_argument("--alt-query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save-oof-topk", type=Path, default=None)
    args = parser.parse_args()

    base_support = load_cache(args.base_support)
    base_query = load_cache(args.base_query)
    alt_support = load_cache(args.alt_support)
    alt_query = load_cache(args.alt_query)
    if base_support["classes"] != alt_support["classes"]:
        raise RuntimeError("Class orders differ")
    support = prepare_pair(base_support, alt_support)
    query = prepare_pair(base_query, alt_query)
    labels = list(base_query["labels"])
    target = base_query["class_ids"].long()
    if not torch.equal(base_support["class_ids"].long(), target):
        raise RuntimeError("This gate requires one aligned support and query per class")
    support_order = base_support["class_ids"].long().argsort()
    expected_ids = torch.arange(len(base_support["classes"]), dtype=torch.long)
    if not torch.equal(base_support["class_ids"].long()[support_order], expected_ids):
        raise RuntimeError("This gate requires exactly one support row for every class")
    candidate_support = support[support_order]

    base_only_prediction = (
        F.normalize(base_query["features"].float(), dim=1)
        @ F.normalize(base_support["features"].float(), dim=1)[support_order].T
    ).argmax(dim=1)
    raw_prediction = (query @ candidate_support.T).argmax(dim=1)
    folds = genus_folds(labels, args.folds)
    oof_prediction = torch.empty_like(target)
    oof_topk_indices = torch.empty((len(target), 20), dtype=torch.long)
    oof_topk_values = torch.empty((len(target), 20), dtype=torch.float32)
    fold_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(args.folds):
        train_indices = torch.where(folds.ne(fold))[0]
        heldout_indices = torch.where(folds.eq(fold))[0]
        model, losses = train_metric(
            support,
            query,
            labels,
            train_indices,
            rank=args.rank,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            seed=args.seed + fold,
            device=device,
        )
        topk_values, topk_indices = predict_topk(model, candidate_support, query[heldout_indices], device)
        prediction = topk_indices[:, 0]
        oof_prediction[heldout_indices] = prediction
        oof_topk_indices[heldout_indices] = topk_indices
        oof_topk_values[heldout_indices] = topk_values
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "heldout_rows": int(len(heldout_indices)),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "heldout_top1": float(prediction.eq(target[heldout_indices]).float().mean().item()),
            }
        )

    summary = {
        "rows": len(target),
        "classes": len(base_support["classes"]),
        "feature_dim": int(support.shape[1]),
        "device": str(device),
        "config": {
            "folds": args.folds,
            "rank": args.rank,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "base_only": metrics(base_only_prediction, target, base_only_prediction),
        "raw_concat": metrics(raw_prediction, target, base_only_prediction),
        "metric_oof_vs_base": metrics(oof_prediction, target, base_only_prediction),
        "metric_oof_vs_raw_concat": metrics(oof_prediction, target, raw_prediction),
        "folds": fold_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_oof_topk is not None:
        args.save_oof_topk.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "topk_indices": oof_topk_indices,
                "topk_values": oof_topk_values,
                "class_ids": target,
                "image_ids": list(base_query["image_ids"]),
                "labels": labels,
                "classes": list(base_support["classes"]),
                "base_only_prediction": base_only_prediction,
                "raw_concat_prediction": raw_prediction,
                "source_summary": str(args.out),
            },
            args.save_oof_topk,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
