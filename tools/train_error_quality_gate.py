from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def softmax_entropy(scores: torch.Tensor) -> torch.Tensor:
    probs = scores.float().softmax(dim=1)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)


def class_count_features(top_indices: torch.Tensor, class_counts: torch.Tensor) -> torch.Tensor:
    gathered = class_counts[top_indices.long()].float()
    return torch.log1p(gathered)


def genus_from_class(name: str) -> str:
    return name.split()[0] if name.split() else name


def genus_features(top_indices: torch.Tensor, classes: list[str]) -> torch.Tensor:
    genus_ids: dict[str, int] = {}
    class_genus = []
    for name in classes:
        genus = genus_from_class(name)
        if genus not in genus_ids:
            genus_ids[genus] = len(genus_ids)
        class_genus.append(genus_ids[genus])
    class_genus_tensor = torch.tensor(class_genus, dtype=torch.long)
    top_genus = class_genus_tensor[top_indices.long()]
    same_as_top1 = top_genus.eq(top_genus[:, :1]).float()
    unique_counts = []
    for row in top_genus:
        unique_counts.append(len(set(int(x) for x in row.tolist())))
    return torch.cat(
        [
            same_as_top1[:, 1:],
            torch.tensor(unique_counts, dtype=torch.float32)[:, None] / top_indices.shape[1],
        ],
        dim=1,
    )


def read_flag_map(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            return result
        normalized = {name.strip(): name for name in reader.fieldnames}
        image_col = normalized.get("image_id")
        category_col = normalized.get("categories")
        if image_col is None or category_col is None:
            raise RuntimeError(f"flag CSV missing expected columns: {reader.fieldnames}")
        for row in reader:
            image_id = row[image_col]
            cats = {part.strip() for part in row[category_col].split("|") if part.strip()}
            result.setdefault(image_id, set()).update(cats)
    return result


def build_quality_targets(image_ids: list[str], flag_map: dict[str, set[str]]) -> tuple[list[str], torch.Tensor]:
    categories = sorted({cat for cats in flag_map.values() for cat in cats})
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    y = torch.zeros((len(image_ids), len(categories)), dtype=torch.float32)
    for row, image_id in enumerate(image_ids):
        for cat in flag_map.get(image_id, set()):
            y[row, cat_to_idx[cat]] = 1.0
    return categories, y


def quality_proto_features(
    train_features: torch.Tensor,
    query_features: torch.Tensor,
    quality_targets: torch.Tensor,
) -> torch.Tensor:
    if quality_targets.numel() == 0 or quality_targets.shape[1] == 0:
        return torch.zeros((query_features.shape[0], 1), dtype=torch.float32)
    train_norm = F.normalize(train_features.float(), dim=1)
    query_norm = F.normalize(query_features.float(), dim=1)
    negative = F.normalize(train_norm[quality_targets.sum(dim=1).eq(0)].mean(dim=0), dim=0)
    scores = []
    for col in range(quality_targets.shape[1]):
        mask = quality_targets[:, col].bool()
        if int(mask.sum()) == 0:
            proto = torch.zeros(train_norm.shape[1], dtype=torch.float32)
        else:
            proto = F.normalize(train_norm[mask].mean(dim=0), dim=0)
        scores.append(query_norm @ (proto - negative))
    per_cat = torch.stack(scores, dim=1)
    return torch.cat([per_cat, per_cat.max(dim=1).values[:, None], per_cat.mean(dim=1, keepdim=True)], dim=1)


def build_meta_features(
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    class_counts: torch.Tensor,
    classes: list[str],
    quality_features: torch.Tensor,
) -> torch.Tensor:
    scores = top_scores.float()
    margins = torch.stack(
        [
            scores[:, 0] - scores[:, 1],
            scores[:, 0] - scores[:, 2],
            scores[:, 0] - scores[:, -1],
            scores[:, 1] - scores[:, 2],
        ],
        dim=1,
    )
    probs = scores.softmax(dim=1)
    prob_features = torch.cat(
        [
            probs[:, :3],
            (probs[:, 0] - probs[:, 1])[:, None],
            softmax_entropy(scores)[:, None],
        ],
        dim=1,
    )
    pieces = [
        row_zscore(scores),
        scores[:, :1],
        margins,
        prob_features,
        class_count_features(top_indices, class_counts),
        genus_features(top_indices, classes),
        quality_features.float(),
    ]
    x = torch.cat(pieces, dim=1)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class GateModel(nn.Module):
    def __init__(self, in_dim: int, architecture: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(in_dim, 1)
        elif architecture == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(architecture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def standardize(train_x: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((x - mean) / std for x in (train_x, *others))


def train_gate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    architecture: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
) -> GateModel:
    seed_everything(seed)
    model = GateModel(train_x.shape[1], architecture, hidden_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    positives = train_y.sum().clamp_min(1.0)
    negatives = (1 - train_y).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(1.0, 20.0)
    generator = torch.Generator().manual_seed(seed + 17)
    for epoch in range(epochs):
        order = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for start in range(0, order.numel(), batch_size):
            rows = order[start : start + batch_size]
            logits = model(train_x[rows])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[rows], pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_gate(model: GateModel, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(model(x[start : start + batch_size]).sigmoid())
    return torch.cat(chunks)


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    pos_ranks = ranks[labels].sum()
    return float((pos_ranks - positives * (positives + 1) / 2) / (positives * negatives))


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, ks: list[int]) -> dict[str, Any]:
    order = scores.argsort(descending=True)
    result: dict[str, Any] = {}
    for k in ks:
        kk = min(k, len(scores))
        result[f"p@{k}"] = float(labels[order[:kk]].float().mean())
        result[f"errors@{k}"] = int(labels[order[:kk]].sum())
    return result


def paired_stats(base_prediction: torch.Tensor, prediction: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    base_correct = base_prediction.eq(labels)
    pred_correct = prediction.eq(labels)
    changed = mask & prediction.ne(base_prediction)
    wins = changed & ~base_correct & pred_correct
    losses = changed & base_correct & ~pred_correct
    return {
        "rows": int(mask.sum()),
        "base_correct": int((mask & base_correct).sum()),
        "candidate_correct": int((mask & pred_correct).sum()),
        "net": int(wins.sum() - losses.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "changed": int(changed.sum()),
    }


def sweep_gate_on_prediction(
    gate_scores: torch.Tensor,
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> dict[str, Any]:
    thresholds = torch.unique(
        torch.quantile(gate_scores, torch.linspace(0.00, 0.99, 100))
    ).tolist()
    trials = []
    for threshold in thresholds:
        pred = base_prediction.clone()
        change = candidate_prediction.ne(base_prediction) & gate_scores.ge(float(threshold))
        pred[change] = candidate_prediction[change]
        trials.append(
            {
                "threshold": float(threshold),
                "dev": paired_stats(base_prediction, pred, labels, dev),
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            row["dev"]["net"],
            -row["dev"]["changed"],
            row["threshold"],
        ),
    )
    pred = base_prediction.clone()
    change = candidate_prediction.ne(base_prediction) & gate_scores.ge(selected["threshold"])
    pred[change] = candidate_prediction[change]
    return {
        "selected": selected,
        "all": paired_stats(base_prediction, pred, labels, torch.ones_like(dev, dtype=torch.bool)),
        "dev": paired_stats(base_prediction, pred, labels, dev),
        "sealed": paired_stats(base_prediction, pred, labels, sealed),
        "prediction": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--oof-topk", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--crossfit-fusion", type=Path, required=True)
    parser.add_argument("--flags-train", type=Path, required=True)
    parser.add_argument("--flags-val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_cache = load_payload(args.train_cache)
    val_cache = load_payload(args.val_cache)
    oof = load_payload(args.oof_topk)
    bank = load_payload(args.candidate_bank)
    fusion = load_payload(args.crossfit_fusion)

    if list(train_cache["image_ids"]) != list(oof["image_ids"]):
        raise RuntimeError("train cache and OOF image order differ")
    if list(val_cache["image_ids"]) != list(bank["image_ids"]):
        raise RuntimeError("val cache and candidate bank image order differ")

    classes = list(train_cache["classes"])
    class_counts = torch.bincount(train_cache["class_ids"].long(), minlength=len(classes))

    train_flags = read_flag_map(args.flags_train)
    val_flags = read_flag_map(args.flags_val)
    quality_categories, train_quality_y = build_quality_targets(list(train_cache["image_ids"]), train_flags)
    _, val_quality_y = build_quality_targets(list(val_cache["image_ids"]), val_flags)
    if val_quality_y.shape[1] != train_quality_y.shape[1]:
        # Val flags are used only for stratified evaluation, so align manually.
        val_quality_y = torch.zeros((len(val_cache["image_ids"]), len(quality_categories)), dtype=torch.float32)
        cat_to_idx = {cat: idx for idx, cat in enumerate(quality_categories)}
        for row, image_id in enumerate(val_cache["image_ids"]):
            for cat in val_flags.get(image_id, set()):
                if cat in cat_to_idx:
                    val_quality_y[row, cat_to_idx[cat]] = 1.0

    train_quality_features = quality_proto_features(
        train_cache["features"], train_cache["features"], train_quality_y
    )
    val_quality_features = quality_proto_features(
        train_cache["features"], val_cache["features"], train_quality_y
    )

    train_x = build_meta_features(
        oof["top_scores"].float(),
        oof["top_indices"].long(),
        class_counts,
        classes,
        train_quality_features,
    )
    val_x = build_meta_features(
        bank["top_values"].float(),
        bank["top_indices"].long(),
        class_counts,
        classes,
        val_quality_features,
    )
    train_x, val_x = standardize(train_x, val_x)

    train_error = oof["top_indices"][:, 0].long().ne(oof["class_ids"].long()).float()
    val_labels = bank["labels"].long()
    base_prediction = bank["top_indices"][:, 0].long()
    val_error = base_prediction.ne(val_labels)
    dev = bank["dev"].bool()
    sealed = bank["sealed"].bool()

    models = []
    for architecture in ["linear", "mlp"]:
        for seed_offset in [0, 1, 2]:
            model = train_gate(
                train_x,
                train_error,
                architecture,
                args.seed + seed_offset * 1009,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.hidden_dim,
                args.dropout,
                args.batch_size,
            )
            train_scores = predict_gate(model, train_x, args.batch_size)
            val_scores = predict_gate(model, val_x, args.batch_size)
            models.append(
                {
                    "architecture": architecture,
                    "seed_offset": seed_offset,
                    "model": model,
                    "train_scores": train_scores,
                    "val_scores": val_scores,
                    "metrics": {
                        "train_auc": auc_score(train_scores, train_error.bool()),
                        "val_auc_all": auc_score(val_scores, val_error),
                        "val_auc_dev": auc_score(val_scores[dev], val_error[dev]),
                        "val_auc_sealed": auc_score(val_scores[sealed], val_error[sealed]),
                        **{
                            f"val_{k}": v
                            for k, v in precision_at_k(
                                val_scores, val_error, [100, 200, 500, 1000, 2000]
                            ).items()
                        },
                    },
                }
            )
            print(json.dumps({"stage": "gate", **models[-1]["metrics"], "architecture": architecture, "seed_offset": seed_offset}), flush=True)

    val_stack = torch.stack([row["val_scores"] for row in models])
    train_stack = torch.stack([row["train_scores"] for row in models])
    ensemble_val_scores = val_stack.mean(dim=0)
    ensemble_train_scores = train_stack.mean(dim=0)
    ensemble_metrics = {
        "train_auc": auc_score(ensemble_train_scores, train_error.bool()),
        "val_auc_all": auc_score(ensemble_val_scores, val_error),
        "val_auc_dev": auc_score(ensemble_val_scores[dev], val_error[dev]),
        "val_auc_sealed": auc_score(ensemble_val_scores[sealed], val_error[sealed]),
        **{
            f"val_{k}": v
            for k, v in precision_at_k(
                ensemble_val_scores, val_error, [100, 200, 500, 1000, 2000]
            ).items()
        },
    }

    crossfit_prediction = fusion["prediction"].long()
    crossfit_gated = sweep_gate_on_prediction(
        ensemble_val_scores,
        base_prediction,
        crossfit_prediction,
        val_labels,
        dev,
        sealed,
    )

    # A simple score-family consensus candidate from the previous scout:
    selected_names = [
        name
        for name in bank["scores"]
        if name.startswith("pair:") or name.startswith("contrast_family:")
    ]
    channels = torch.stack([row_zscore(bank["scores"][name].float()) for name in selected_names])
    aggregate = channels.mean(dim=0)
    consensus_prediction = bank["top_indices"].long().gather(
        1, aggregate.argmax(dim=1, keepdim=True)
    ).squeeze(1)
    consensus_gated = sweep_gate_on_prediction(
        ensemble_val_scores,
        base_prediction,
        consensus_prediction,
        val_labels,
        dev,
        sealed,
    )

    result = {
        "protocol": {
            "train_target": "strict train OOF top-1 error",
            "validation_target": "strong validation top-1 error; labels used only for evaluation and dev threshold selection",
            "manual_quality_flags": "train flags used to build automatic prototype features; val flags used only for stratified diagnostics",
            "test_seen_touched": False,
            "submission_created": False,
        },
        "rows": {
            "train": len(train_error),
            "train_errors": int(train_error.sum()),
            "val": len(val_error),
            "val_errors": int(val_error.sum()),
            "val_dev_errors": int((dev & val_error).sum()),
            "val_sealed_errors": int((sealed & val_error).sum()),
            "quality_categories": quality_categories,
            "flagged_train": int(train_quality_y.sum(dim=1).gt(0).sum()),
            "flagged_val": int(val_quality_y.sum(dim=1).gt(0).sum()),
        },
        "single_models": [
            {
                "architecture": row["architecture"],
                "seed_offset": row["seed_offset"],
                "metrics": row["metrics"],
            }
            for row in models
        ],
        "ensemble_metrics": ensemble_metrics,
        "crossfit_gated": {
            key: value
            for key, value in crossfit_gated.items()
            if key != "prediction"
        },
        "consensus_gated": {
            key: value
            for key, value in consensus_gated.items()
            if key != "prediction"
        },
    }

    torch.save(
        {
            "val_error_gate": ensemble_val_scores.half(),
            "train_error_gate": ensemble_train_scores.half(),
            "val_error": val_error,
            "train_error": train_error.bool(),
            "crossfit_gated_prediction": crossfit_gated["prediction"],
            "consensus_gated_prediction": consensus_gated["prediction"],
            "image_ids": bank["image_ids"],
            "dev": dev,
            "sealed": sealed,
        },
        args.out_dir / "gate_outputs.pt",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# Error/Quality Gate Scout

## Outcome

- Train OOF error target: `{int(train_error.sum())} / {len(train_error)}`.
- Validation strong-base errors: `{int(val_error.sum())} / {len(val_error)}`.
- Ensemble error-gate AUC: all `{ensemble_metrics['val_auc_all']:.4f}`, dev `{ensemble_metrics['val_auc_dev']:.4f}`, sealed `{ensemble_metrics['val_auc_sealed']:.4f}`.
- Precision in top 500 suspected errors: `{ensemble_metrics['val_p@500']:.4f}` with `{ensemble_metrics['val_errors@500']}` actual errors.
- Crossfit gated by error score: all `{crossfit_gated['all']['net']:+d}`, dev `{crossfit_gated['dev']['net']:+d}`, sealed `{crossfit_gated['sealed']['net']:+d}`.
- Consensus gated by error score: all `{consensus_gated['all']['net']:+d}`, dev `{consensus_gated['dev']['net']:+d}`, sealed `{consensus_gated['sealed']['net']:+d}`.

## Decision

No `test_seen` inference or submission was produced.
"""
    (args.out_dir / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "done", "ensemble": ensemble_metrics, "crossfit_gated": result["crossfit_gated"], "consensus_gated": result["consensus_gated"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
