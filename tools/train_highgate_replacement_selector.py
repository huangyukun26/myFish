from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def softmax_entropy(scores: torch.Tensor) -> torch.Tensor:
    probs = scores.float().softmax(dim=1)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)


def paired_stats(
    base_prediction: torch.Tensor,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
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


def genus_from_class(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def build_folds(image_ids: list[str], dev: torch.Tensor, folds: int) -> torch.Tensor:
    fold_ids = torch.full((len(image_ids),), -1, dtype=torch.long)
    for row, image_id in enumerate(image_ids):
        if bool(dev[row]):
            fold_ids[row] = stable_hash("highgate-selector:" + image_id) % folds
    return fold_ids


def family_name(channel: str) -> str:
    if channel.startswith("patch:"):
        return "patch"
    if channel.startswith("pair:"):
        return "pair"
    if channel.startswith("baseaware:"):
        return "baseaware"
    if channel.startswith("contrast_family:"):
        return "contrast"
    if channel.startswith("quality_lora:"):
        return "quality"
    if channel.startswith("old_crossfit:"):
        return "old_crossfit"
    return channel.split(":", 1)[0]


def build_slot_features(
    bank: dict[str, Any],
    gate_scores: torch.Tensor,
    train_cache: dict[str, Any] | None,
) -> tuple[torch.Tensor, list[str]]:
    top_values = bank["top_values"].float()
    top_indices = bank["top_indices"].long()
    scores = {name: value.float() for name, value in bank["scores"].items()}
    channel_names = list(scores)
    score_stack = torch.stack([row_zscore(scores[name]) for name in channel_names], dim=2)
    base_z = row_zscore(top_values)
    n, k = top_indices.shape
    features: list[torch.Tensor] = []
    names: list[str] = []

    # Candidate-local source scores.
    features.append(base_z[:, :, None])
    names.append("base_z")
    features.append(score_stack)
    names.extend([f"score:{name}" for name in channel_names])
    features.append(score_stack - base_z[:, :, None])
    names.extend([f"delta:{name}" for name in channel_names])

    # Source vote summaries.
    argmax = score_stack.argmax(dim=1)
    vote_count = torch.stack([(argmax == slot).sum(dim=1) for slot in range(k)], dim=1).float()
    features.append(vote_count[:, :, None] / max(1, len(channel_names)))
    names.append("vote_frac_all")

    for fam in sorted({family_name(name) for name in channel_names}):
        indices = [idx for idx, name in enumerate(channel_names) if family_name(name) == fam]
        fam_scores = score_stack[:, :, indices]
        fam_argmax = fam_scores.argmax(dim=1)
        fam_vote = torch.stack([(fam_argmax == slot).sum(dim=1) for slot in range(k)], dim=1).float()
        features.append(fam_scores.mean(dim=2, keepdim=True))
        names.append(f"family_mean:{fam}")
        features.append(fam_vote[:, :, None] / max(1, len(indices)))
        names.append(f"family_vote:{fam}")

    # Row-level score shape features, repeated over slots.
    margins = torch.stack(
        [
            top_values[:, 0] - top_values[:, 1],
            top_values[:, 0] - top_values[:, 2],
            top_values[:, 0] - top_values[:, -1],
            softmax_entropy(top_values),
            gate_scores.float(),
        ],
        dim=1,
    )
    features.append(margins[:, None, :].expand(-1, k, -1))
    names.extend(["margin01", "margin02", "margin0last", "base_entropy", "error_gate"])

    # Slot identity and base-slot flag.
    slot_rank = torch.arange(k, dtype=torch.float32)[None, :, None].expand(n, -1, -1) / max(1, k - 1)
    is_base_slot = torch.zeros((n, k, 1), dtype=torch.float32)
    is_base_slot[:, 0, 0] = 1.0
    features.extend([slot_rank, is_base_slot])
    names.extend(["slot_rank", "is_base_slot"])

    if train_cache is not None:
        class_counts = torch.bincount(train_cache["class_ids"].long(), minlength=len(train_cache["classes"]))
        count_feature = torch.log1p(class_counts[top_indices].float())
        features.append(count_feature[:, :, None])
        names.append("log_class_count")
        genus_to_idx: dict[str, int] = {}
        class_genus = []
        for cls in train_cache["classes"]:
            genus = genus_from_class(cls)
            if genus not in genus_to_idx:
                genus_to_idx[genus] = len(genus_to_idx)
            class_genus.append(genus_to_idx[genus])
        class_genus_tensor = torch.tensor(class_genus, dtype=torch.long)
        top_genus = class_genus_tensor[top_indices]
        same_as_top1 = top_genus.eq(top_genus[:, :1]).float()
        features.append(same_as_top1[:, :, None])
        names.append("same_genus_as_top1")

    x = torch.cat(features, dim=2)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


class SelectorModel(nn.Module):
    def __init__(self, in_dim: int, architecture: str, hidden_dim: int, dropout: float):
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
        n, k, d = x.shape
        return self.net(x.reshape(n * k, d)).reshape(n, k)


def standardize_fold(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    eval_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_flat = x[train_mask].reshape(-1, x.shape[2])
    mean = train_flat.mean(dim=0, keepdim=True)
    std = train_flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x[train_mask] - mean) / std
    x_eval = (x[eval_mask] - mean) / std
    return x_train, x_eval, mean, std


def row_weights(
    labels: torch.Tensor,
    top_indices: torch.Tensor,
    gate_scores: torch.Tensor,
    mask: torch.Tensor,
    error_weight: float,
    preserve_weight: float,
    unrecoverable_weight: float,
    gate_weight: float,
) -> torch.Tensor:
    row_labels = labels[mask]
    row_top = top_indices[mask]
    row_gate = gate_scores[mask].float()
    base_correct = row_top[:, 0].eq(row_labels)
    in_topk = row_top.eq(row_labels[:, None]).any(dim=1)
    weights = torch.full_like(row_gate, preserve_weight)
    weights[~base_correct & in_topk] = error_weight
    weights[~in_topk] = unrecoverable_weight
    return weights * (1.0 + gate_weight * row_gate)


def target_slots(labels: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    matches = top_indices.eq(labels[:, None])
    targets = matches.float().argmax(dim=1).long()
    targets[~matches.any(dim=1)] = 0
    return targets


def train_model(
    x: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    architecture: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> SelectorModel:
    seed_everything(seed)
    model = SelectorModel(x.shape[2], architecture, hidden_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 177)
    for _epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, order.numel(), batch_size):
            rows = order[start : start + batch_size]
            logits = model(x[rows])
            loss = F.cross_entropy(logits, targets[rows], reduction="none")
            loss = (loss * weights[rows]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_scores(model: SelectorModel, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(model(x[start : start + batch_size]).cpu())
    return torch.cat(chunks, dim=0)


def prediction_from_scores(
    top_indices: torch.Tensor,
    base_prediction: torch.Tensor,
    scores: torch.Tensor,
    gate_scores: torch.Tensor,
    gate_threshold: float,
    confidence_threshold: float,
) -> torch.Tensor:
    probs = scores.softmax(dim=1)
    pred_slot = scores.argmax(dim=1)
    selected_score = scores.gather(1, pred_slot[:, None]).squeeze(1)
    base_score = scores[:, 0]
    confidence = selected_score - base_score
    change = pred_slot.ne(0) & gate_scores.ge(gate_threshold) & confidence.ge(confidence_threshold)
    prediction = base_prediction.clone()
    prediction[change] = top_indices.gather(1, pred_slot[:, None]).squeeze(1)[change]
    # Softmax probability is computed to make score scaling less opaque in saved diagnostics.
    _ = probs
    return prediction


def threshold_grid(gate_scores: torch.Tensor, score_margin: torch.Tensor) -> list[tuple[float, float]]:
    gate_thresholds = torch.unique(torch.quantile(gate_scores.float(), torch.linspace(0.0, 0.995, 80))).tolist()
    positive_margin = score_margin[score_margin > 0]
    if positive_margin.numel() == 0:
        confidence_thresholds = [0.0]
    else:
        confidence_thresholds = torch.unique(
            torch.quantile(positive_margin.float(), torch.linspace(0.0, 0.95, 40))
        ).tolist()
        confidence_thresholds = [-10.0, -1.0, -0.25, 0.0] + confidence_thresholds
    return [(float(g), float(c)) for g in gate_thresholds for c in confidence_thresholds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--gate-outputs", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--architectures", nargs="+", default=["linear", "mlp"])
    parser.add_argument("--error-weights", nargs="+", type=float, default=[2.0, 4.0, 8.0])
    parser.add_argument("--preserve-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--gate-weights", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.candidate_bank, map_location="cpu", weights_only=False)
    gate_payload = torch.load(args.gate_outputs, map_location="cpu", weights_only=False)
    gate_scores = gate_payload["val_error_gate"].float()
    train_cache = (
        torch.load(args.train_cache, map_location="cpu", weights_only=False)
        if args.train_cache
        else None
    )

    x, feature_names = build_slot_features(bank, gate_scores, train_cache)
    labels = bank["labels"].long()
    top_indices = bank["top_indices"].long()
    targets = target_slots(labels, top_indices)
    base_prediction = top_indices[:, 0]
    dev = bank["dev"].bool()
    sealed = bank["sealed"].bool()
    fold_ids = build_folds(list(bank["image_ids"]), dev, args.folds)

    configs = [
        (architecture, error_weight, preserve_weight, gate_weight)
        for architecture in args.architectures
        for error_weight in args.error_weights
        for preserve_weight in args.preserve_weights
        for gate_weight in args.gate_weights
    ]
    trials: list[dict[str, Any]] = []
    oof_scores_by_config: dict[tuple[str, float, float, float], torch.Tensor] = {}
    for config_index, (architecture, error_weight, preserve_weight, gate_weight) in enumerate(configs):
        oof_scores = torch.zeros((len(labels), top_indices.shape[1]), dtype=torch.float32)
        for fold in range(args.folds):
            train_mask = dev & fold_ids.ne(fold)
            holdout_mask = dev & fold_ids.eq(fold)
            x_train, x_holdout, _mean, _std = standardize_fold(x, train_mask, holdout_mask)
            weights = row_weights(
                labels,
                top_indices,
                gate_scores,
                train_mask,
                error_weight,
                preserve_weight,
                unrecoverable_weight=preserve_weight,
                gate_weight=gate_weight,
            )
            model = train_model(
                x_train,
                targets[train_mask],
                weights,
                architecture,
                args.hidden_dim,
                args.dropout,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.batch_size,
                args.seed + config_index * 101 + fold,
            )
            oof_scores[holdout_mask] = predict_scores(model, x_holdout, args.batch_size)
        pred_slots = oof_scores.argmax(dim=1)
        score_margin = oof_scores.gather(1, pred_slots[:, None]).squeeze(1) - oof_scores[:, 0]
        for gate_threshold, confidence_threshold in threshold_grid(gate_scores[dev], score_margin[dev]):
            prediction = prediction_from_scores(
                top_indices,
                base_prediction,
                oof_scores,
                gate_scores,
                gate_threshold,
                confidence_threshold,
            )
            stats = paired_stats(base_prediction, prediction, labels, dev)
            trials.append(
                {
                    "architecture": architecture,
                    "error_weight": error_weight,
                    "preserve_weight": preserve_weight,
                    "gate_weight": gate_weight,
                    "gate_threshold": gate_threshold,
                    "confidence_threshold": confidence_threshold,
                    "dev_oof": stats,
                }
            )
        oof_scores_by_config[(architecture, error_weight, preserve_weight, gate_weight)] = oof_scores
        print(
            json.dumps(
                {
                    "stage": "config",
                    "config": config_index + 1,
                    "configs": len(configs),
                    "architecture": architecture,
                    "error_weight": error_weight,
                    "preserve_weight": preserve_weight,
                    "gate_weight": gate_weight,
                    "best_net": max(
                        row["dev_oof"]["net"]
                        for row in trials
                        if row["architecture"] == architecture
                        and row["error_weight"] == error_weight
                        and row["preserve_weight"] == preserve_weight
                        and row["gate_weight"] == gate_weight
                    ),
                }
            ),
            flush=True,
        )

    selected = max(
        trials,
        key=lambda row: (
            row["dev_oof"]["net"],
            -row["dev_oof"]["changed"],
            row["gate_threshold"],
            row["confidence_threshold"],
        ),
    )
    key = (
        selected["architecture"],
        selected["error_weight"],
        selected["preserve_weight"],
        selected["gate_weight"],
    )
    final_train_mask = dev
    sealed_mask = bank["sealed"].bool()
    x_train, x_sealed, mean, std = standardize_fold(x, final_train_mask, sealed_mask)
    final_weights = row_weights(
        labels,
        top_indices,
        gate_scores,
        final_train_mask,
        selected["error_weight"],
        selected["preserve_weight"],
        unrecoverable_weight=selected["preserve_weight"],
        gate_weight=selected["gate_weight"],
    )
    final_model = train_model(
        x_train,
        targets[final_train_mask],
        final_weights,
        selected["architecture"],
        args.hidden_dim,
        args.dropout,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.seed + 99991,
    )
    final_scores = torch.zeros((len(labels), top_indices.shape[1]), dtype=torch.float32)
    final_scores[dev] = oof_scores_by_config[key][dev]
    final_scores[sealed_mask] = predict_scores(final_model, x_sealed, args.batch_size)
    final_prediction = prediction_from_scores(
        top_indices,
        base_prediction,
        final_scores,
        gate_scores,
        selected["gate_threshold"],
        selected["confidence_threshold"],
    )
    masks = {
        "all_crossfit_plus_sealed": torch.ones(len(labels), dtype=torch.bool),
        "dev_oof": dev,
        "sealed_once": sealed_mask,
    }
    final_stats = {
        name: paired_stats(base_prediction, final_prediction, labels, mask)
        for name, mask in masks.items()
    }
    result = {
        "protocol": {
            "training_scope": "dev-only five-fold OOF replacement selector; sealed read once after dev selection",
            "features": feature_names,
            "test_seen_touched": False,
            "submission_created": False,
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "base": {
            "rows": len(labels),
            "correct": int(base_prediction.eq(labels).sum()),
            "dev_correct": int((dev & base_prediction.eq(labels)).sum()),
            "sealed_correct": int((sealed_mask & base_prediction.eq(labels)).sum()),
        },
        "selected": selected,
        "final": final_stats,
        "trials": trials,
    }
    torch.save(
        {
            "selected": selected,
            "feature_names": feature_names,
            "final_scores": final_scores.half(),
            "prediction": final_prediction,
            "dev": dev,
            "sealed": sealed_mask,
            "mean": mean,
            "std": std,
            "state_dict": final_model.state_dict(),
        },
        args.out_dir / "selector.pt",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# High-Gate Replacement Selector

## Outcome

- Selected on dev OOF: `{selected['architecture']}`, error weight `{selected['error_weight']}`, preserve weight `{selected['preserve_weight']}`, gate weight `{selected['gate_weight']}`.
- Thresholds: gate `{selected['gate_threshold']:.6f}`, confidence `{selected['confidence_threshold']:.6f}`.
- Locked dev OOF: net {final_stats['dev_oof']['net']:+d} ({final_stats['dev_oof']['wins']} wins / {final_stats['dev_oof']['losses']} losses; {final_stats['dev_oof']['changed']} changes).
- Sealed once: net {final_stats['sealed_once']['net']:+d} ({final_stats['sealed_once']['wins']} wins / {final_stats['sealed_once']['losses']} losses; {final_stats['sealed_once']['changed']} changes).
- Combined: net {final_stats['all_crossfit_plus_sealed']['net']:+d}.

No `test_seen` inference or submission was produced.
"""
    (args.out_dir / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "done", "selected": selected, "final": final_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
