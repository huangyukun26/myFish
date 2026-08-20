from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_text_to_visual_adapter import train_model as train_residual_text_adapter


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def genus(label: str) -> str:
    return label.split(maxsplit=1)[0]


class TextVisualMapper(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, visual_dim, bias=False),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_dim, visual_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def train_mapper(
    text: torch.Tensor,
    visual: torch.Tensor,
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    contrastive_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[TextVisualMapper, list[float]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = TextVisualMapper(text.shape[1], visual.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    text = text.to(device)
    visual = visual.to(device)
    losses = []
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(text), generator=generator)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].to(device)
            predicted = model(text[indices])
            target = visual[indices]
            cosine_loss = 1.0 - (predicted * target).sum(dim=1).mean()
            logits = predicted @ target.T / temperature
            labels = torch.arange(len(indices), device=device)
            contrastive_loss = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            loss = cosine_loss + contrastive_weight * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_rows += len(indices)
        losses.append(total_loss / max(1, total_rows))
    return model, losses


def candidate_indices(
    heldout_classes: list[str], all_classes: list[str], candidate_count: int, seed: int
) -> tuple[list[str], torch.Tensor]:
    heldout = set(heldout_classes)
    distractors = [name for name in all_classes if name not in heldout]
    needed = candidate_count - len(heldout_classes)
    if needed < 0 or needed > len(distractors):
        raise RuntimeError("Cannot construct requested candidate pool")
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(distractors), generator=generator)[:needed].tolist()
    classes = heldout_classes + [distractors[index] for index in selected]
    all_to_idx = {name: index for index, name in enumerate(all_classes)}
    return classes, torch.tensor([all_to_idx[name] for name in classes], dtype=torch.long)


@dataclass
class Accumulator:
    rows: int = 0
    top1: int = 0
    top5: int = 0
    top20: int = 0
    changed: int = 0
    wins: int = 0
    losses: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> None:
        top1 = prediction[:, 0]
        base_correct = base.eq(target)
        correct = top1.eq(target)
        self.rows += len(target)
        self.top1 += int(correct.sum().item())
        self.top5 += int(prediction[:, :5].eq(target[:, None]).any(dim=1).sum().item())
        self.top20 += int(prediction.eq(target[:, None]).any(dim=1).sum().item())
        self.changed += int(top1.ne(base).sum().item())
        self.wins += int((correct & ~base_correct).sum().item())
        self.losses += int((~correct & base_correct).sum().item())

    def summary(self) -> dict:
        return {
            "rows": self.rows,
            "top1": self.top1 / max(1, self.rows),
            "top5": self.top5 / max(1, self.rows),
            "top20": self.top20 / max(1, self.rows),
            "changed": self.changed,
            "wins": self.wins,
            "losses": self.losses,
            "net": self.wins - self.losses,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioclip-support", type=Path, required=True)
    parser.add_argument("--bioclip-query", type=Path, required=True)
    parser.add_argument("--dino-support", type=Path, required=True)
    parser.add_argument("--dino-query", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-modes", default="species,genus")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=11598)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--bioclip-adapter-blend", type=float, default=0.75)
    parser.add_argument("--bioclip-residual-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    bio_support = torch.load(args.bioclip_support, map_location="cpu", weights_only=False)
    bio_query = torch.load(args.bioclip_query, map_location="cpu", weights_only=False)
    dino_support = torch.load(args.dino_support, map_location="cpu", weights_only=False)
    dino_query = torch.load(args.dino_query, map_location="cpu", weights_only=False)
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    if bio_query["image_ids"] != dino_query["image_ids"]:
        raise RuntimeError("BioCLIP and DINO query rows differ")
    if bio_support["labels"] != bio_query["labels"]:
        raise RuntimeError("BioCLIP support and query class rows differ")
    if dino_support["labels"] != dino_query["labels"]:
        raise RuntimeError("DINO support and query class rows differ")

    all_classes = list(text_payload["classes"])
    all_to_idx = {name: index for index, name in enumerate(all_classes)}
    text = normalize(text_payload["features"])
    bio_support_features = normalize(bio_support["features"])
    bio_features = normalize(bio_query["features"])
    dino_support_features = normalize(dino_support["features"])
    dino_query_features = normalize(dino_query["features"])
    visual_targets = normalize(dino_support_features + dino_query_features)
    bio_visual_targets = normalize(bio_support_features + bio_features)
    seen_classes = list(dino_query["labels"])
    seen_to_row = {name: index for index, name in enumerate(seen_classes)}
    missing = [name for name in seen_classes if name not in all_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} seen classes missing from text cache")

    weights = [round(index / 10, 1) for index in range(11)]
    modes = [value.strip() for value in args.holdout_modes.split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_modes = {}
    for mode in modes:
        totals = {weight: Accumulator() for weight in weights}
        original_total = Accumulator()
        adapted_vs_original_total = Accumulator()
        fold_rows = []
        for fold in range(args.folds):
            fold_of = {
                name: stable_fold(name if mode == "species" else genus(name), args.folds, args.seed)
                for name in seen_classes
            }
            heldout_classes = [name for name in seen_classes if fold_of[name] == fold]
            train_classes = [name for name in seen_classes if fold_of[name] != fold]
            train_rows = torch.tensor([seen_to_row[name] for name in train_classes], dtype=torch.long)
            train_text_rows = torch.tensor([all_to_idx[name] for name in train_classes], dtype=torch.long)
            model, losses = train_mapper(
                text[train_text_rows],
                visual_targets[train_rows],
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                contrastive_weight=args.contrastive_weight,
                seed=args.seed + fold,
                device=device,
            )
            torch.manual_seed(args.seed + 10000 + fold)
            bio_model, bio_losses = train_residual_text_adapter(
                text[train_text_rows],
                bio_visual_targets[train_rows],
                hidden_dim=0,
                residual_scale=args.bioclip_residual_scale,
                dropout=0.0,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                contrastive_weight=0.05,
                device=device,
            )
            model.eval()
            bio_model.eval()
            with torch.inference_mode():
                mapped_text = []
                adapted_bio_text = []
                for start in range(0, len(text), 1024):
                    mapped_text.append(model(text[start : start + 1024].to(device)).cpu())
                    adapted_bio_text.append(bio_model(text[start : start + 1024].to(device)).cpu())
                mapped_text = torch.cat(mapped_text)
                adapted_bio_text = torch.cat(adapted_bio_text)

            candidates, candidate_rows = candidate_indices(
                heldout_classes, all_classes, args.candidate_count, args.seed + 1009 * fold
            )
            candidate_to_idx = {name: index for index, name in enumerate(candidates)}
            query_rows = torch.tensor([seen_to_row[name] for name in heldout_classes], dtype=torch.long)
            target = torch.tensor([candidate_to_idx[name] for name in heldout_classes], device=device)
            original_candidates = text[candidate_rows]
            adapted_candidates = normalize(
                args.bioclip_adapter_blend * original_candidates
                + (1.0 - args.bioclip_adapter_blend) * adapted_bio_text[candidate_rows]
            )
            original_scores = bio_features[query_rows].to(device) @ original_candidates.to(device).T
            base_scores = bio_features[query_rows].to(device) @ adapted_candidates.to(device).T
            alt_scores = dino_query_features[query_rows].to(device) @ mapped_text[candidate_rows].to(device).T
            base_z = (base_scores - base_scores.mean(dim=1, keepdim=True)) / base_scores.std(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            alt_z = (alt_scores - alt_scores.mean(dim=1, keepdim=True)) / alt_scores.std(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            base_top1 = base_scores.argmax(dim=1)
            original_prediction = original_scores.topk(20, dim=1).indices
            original_top1 = original_prediction[:, 0]
            adapted_prediction = base_scores.topk(20, dim=1).indices
            original_total.update(original_prediction, target, original_top1)
            adapted_vs_original_total.update(adapted_prediction, target, original_top1)
            fold_metrics = {}
            for weight in weights:
                scores = (1.0 - weight) * base_z + weight * alt_z
                prediction = scores.topk(20, dim=1).indices
                totals[weight].update(prediction, target, base_top1)
                local = Accumulator()
                local.update(prediction, target, base_top1)
                fold_metrics[str(weight)] = local.summary()
            fold_rows.append(
                {
                    "fold": fold,
                    "heldout_classes": len(heldout_classes),
                    "train_classes": len(train_classes),
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "bio_initial_loss": bio_losses[0],
                    "bio_final_loss": bio_losses[-1],
                    "metrics": fold_metrics,
                }
            )
            original_local = Accumulator()
            original_local.update(original_prediction, target, original_top1)
            adapted_local = Accumulator()
            adapted_local.update(adapted_prediction, target, original_top1)
            fold_rows[-1]["original"] = original_local.summary()
            fold_rows[-1]["adapted_vs_original"] = adapted_local.summary()

        weight_rows = []
        for weight in weights:
            row = {"dino_weight": weight, **totals[weight].summary()}
            row["worst_fold_net"] = min(fold["metrics"][str(weight)]["net"] for fold in fold_rows)
            weight_rows.append(row)
        result_modes[mode] = {
            "original_base": original_total.summary(),
            "adapted_vs_original": adapted_vs_original_total.summary(),
            "base": next(row for row in weight_rows if row["dino_weight"] == 0.0),
            "fixed_0.3": next(row for row in weight_rows if row["dino_weight"] == 0.3),
            "fixed_0.5": next(row for row in weight_rows if row["dino_weight"] == 0.5),
            "gate_best": max(weight_rows, key=lambda row: (row["top1"], -abs(row["dino_weight"] - 0.3))),
            "weights": weight_rows,
            "folds": fold_rows,
        }

    summary = {
        "device": str(device),
        "seen_classes": len(seen_classes),
        "all_text_classes": len(all_classes),
        "candidate_count": args.candidate_count,
        "config": vars(args),
        "modes": result_modes,
    }
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["config"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
