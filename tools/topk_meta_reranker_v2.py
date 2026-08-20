from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_topk_meta_reranker import (
    genus,
    load_split,
    metrics,
    parse_grid,
    parse_split,
    row_zscore,
    target_indices,
    trigger_mask,
)


def rank_features(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = scores.shape
    order = scores.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(k).view(1, k).expand(n, k))
    rank_f = rank.float() / max(1, k - 1)
    return rank_f, 1.0 - rank_f, rank


def enhanced_features(split: dict[str, Any]) -> torch.Tensor:
    base = torch.tensor(split["base_scores"], dtype=torch.float32)
    if base.ndim != 2:
        base = split["base_scores"].float()
    n, k = base.shape
    rank = torch.linspace(0.0, 1.0, steps=k).view(1, k).expand(n, k)
    inv_rank = 1.0 - rank
    base_z = row_zscore(base)
    base_center = base - base.mean(dim=1, keepdim=True)
    base_delta_top = base - base[:, :1]
    base_top12_margin = (base[:, :1] - base[:, 1:2]).expand(n, k)
    base_top15_margin = (base[:, :1] - base[:, min(4, k - 1) : min(5, k)]).expand(n, k)
    base_top120_margin = (base[:, :1] - base[:, -1:]).expand(n, k)

    pieces = [
        base,
        base_z,
        base_center,
        base_delta_top,
        rank,
        inv_rank,
        base_top12_margin,
        base_top15_margin,
        base_top120_margin,
    ]

    all_z = [base_z]
    top1_votes = []
    top3_votes = []
    top5_votes = []
    inv_rank_values = [inv_rank]
    branch_top1_genus_matches = []
    branch_top3_genus_matches = []
    branch_winner_same_as_base = []
    base_genera = [genus(preds[0]) for preds in split["predictions"]]

    for scores in split["adapter_scores"]:
        scores = scores.float()
        z = row_zscore(scores)
        centered = scores - scores.mean(dim=1, keepdim=True)
        delta_top = scores - scores.max(dim=1, keepdim=True).values
        rank_f, inv_rank_f, rank_i = rank_features(scores)
        top1 = (rank_i == 0).float()
        top3 = (rank_i < 3).float()
        top5 = (rank_i < 5).float()
        pieces.extend([scores, z, centered, delta_top, rank_f, inv_rank_f, top1, top3, top5])
        all_z.append(z)
        top1_votes.append(top1)
        top3_votes.append(top3)
        top5_votes.append(top5)
        inv_rank_values.append(inv_rank_f)

        order = scores.argsort(dim=1, descending=True)
        same_top1_genus = torch.zeros((n, k), dtype=torch.float32)
        same_top3_genus = torch.zeros((n, k), dtype=torch.float32)
        same_top1_class = torch.zeros((n, k), dtype=torch.float32)
        for row_idx, preds in enumerate(split["predictions"]):
            cand_genera = [genus(pred) for pred in preds]
            top1_idx = int(order[row_idx, 0].item())
            top3_idx = [int(v) for v in order[row_idx, : min(3, k)].tolist()]
            top1_g = cand_genera[top1_idx]
            top3_g = {cand_genera[idx] for idx in top3_idx}
            for col_idx, cand_g in enumerate(cand_genera):
                same_top1_genus[row_idx, col_idx] = float(cand_g == top1_g)
                same_top3_genus[row_idx, col_idx] = float(cand_g in top3_g)
                same_top1_class[row_idx, col_idx] = float(col_idx == top1_idx)
        branch_top1_genus_matches.append(same_top1_genus)
        branch_top3_genus_matches.append(same_top3_genus)
        branch_winner_same_as_base.append((order[:, :1] == 0).float().expand(n, k))
        pieces.extend([same_top1_genus, same_top3_genus, same_top1_class, branch_winner_same_as_base[-1]])

    score_stack = torch.stack(all_z, dim=2)
    inv_rank_stack = torch.stack(inv_rank_values, dim=2)
    pieces.extend(
        [
            score_stack.mean(dim=2),
            score_stack.std(dim=2),
            score_stack.max(dim=2).values,
            score_stack.min(dim=2).values,
            inv_rank_stack.mean(dim=2),
            inv_rank_stack.std(dim=2),
            torch.stack(top1_votes, dim=2).sum(dim=2) if top1_votes else torch.zeros((n, k)),
            torch.stack(top3_votes, dim=2).sum(dim=2) if top3_votes else torch.zeros((n, k)),
            torch.stack(top5_votes, dim=2).sum(dim=2) if top5_votes else torch.zeros((n, k)),
            torch.stack(branch_top1_genus_matches, dim=2).sum(dim=2)
            if branch_top1_genus_matches
            else torch.zeros((n, k)),
            torch.stack(branch_top3_genus_matches, dim=2).sum(dim=2)
            if branch_top3_genus_matches
            else torch.zeros((n, k)),
            torch.stack(branch_winner_same_as_base, dim=2).mean(dim=2)
            if branch_winner_same_as_base
            else torch.zeros((n, k)),
        ]
    )

    genus_count_frac = torch.zeros((n, k), dtype=torch.float32)
    same_as_top1_genus = torch.zeros((n, k), dtype=torch.float32)
    same_as_base_genus = torch.zeros((n, k), dtype=torch.float32)
    dominant_genus_flag = torch.zeros((n, k), dtype=torch.float32)
    top1_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    dominant_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    distinct_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    for row_idx, preds in enumerate(split["predictions"]):
        genera = [genus(pred) for pred in preds]
        counts = Counter(genera)
        top_g = genera[0] if genera else ""
        max_count = max(counts.values()) if counts else 0
        top_frac = counts.get(top_g, 0) / max(1, k)
        dom_frac = max_count / max(1, k)
        distinct = len(counts) / max(1, k)
        for col, g in enumerate(genera):
            genus_count_frac[row_idx, col] = counts[g] / max(1, k)
            same_as_top1_genus[row_idx, col] = float(g == top_g)
            same_as_base_genus[row_idx, col] = float(g == base_genera[row_idx])
            dominant_genus_flag[row_idx, col] = float(counts[g] == max_count)
        top1_genus_frac[row_idx] = top_frac
        dominant_genus_frac[row_idx] = dom_frac
        distinct_genus_frac[row_idx] = distinct
    pieces.extend(
        [
            genus_count_frac,
            same_as_top1_genus,
            same_as_base_genus,
            dominant_genus_flag,
            top1_genus_frac,
            dominant_genus_frac,
            distinct_genus_frac,
        ]
    )
    return torch.stack(pieces, dim=2)


def standardize(train_x: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = train_x.flatten(0, 1)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)
    return (train_x - mean) / std, (x - mean) / std, mean, std


class CandidateScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(feature_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    x: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    hidden_dim: int,
    dropout: float,
) -> tuple[CandidateScorer, list[float]]:
    torch.manual_seed(seed)
    keep = targets >= 0
    x = x[keep]
    targets = targets[keep]
    if not len(targets):
        raise RuntimeError("No train rows have the true label in topK")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateScorer(x.shape[-1], hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x, targets), batch_size=batch_size, shuffle=True, num_workers=0)
    losses = []
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        losses.append(total / max(1, count))
    model.eval()
    return model, losses


def evaluate_sweep(
    split: dict[str, Any],
    meta_scores: torch.Tensor,
    *,
    weight_grid: list[float],
    margin_grid: list[float],
    genus_frac_grid: list[float],
    trigger_modes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor]:
    base = torch.tensor(split["base_scores"], dtype=torch.float32)
    if base.ndim != 2:
        base = split["base_scores"].float()
    meta_z = row_zscore(meta_scores)
    rows = []
    best_row = None
    best_indices = None
    for weight in weight_grid:
        for margin_threshold in margin_grid:
            for genus_frac_threshold in genus_frac_grid:
                for mode in trigger_modes:
                    trigger = trigger_mask(
                        split,
                        mode=mode,
                        margin_threshold=margin_threshold,
                        genus_frac_threshold=genus_frac_threshold,
                    )
                    final = base + weight * meta_z
                    final = torch.where(trigger[:, None], final, base)
                    indices = final.argsort(dim=1, descending=True)
                    setattr(indices, "triggered", int(trigger.sum().item()))
                    row = {
                        "weight": weight,
                        "margin_threshold": margin_threshold,
                        "genus_frac_threshold": genus_frac_threshold,
                        "trigger_mode": mode,
                        **metrics(indices, split),
                    }
                    rows.append(row)
                    key = (row["top1"], row["net_wins"], -row["losses"], -abs(weight))
                    if best_row is None or key > best_row[0]:
                        best_row = (key, row)
                        best_indices = indices
    assert best_row is not None and best_indices is not None
    return rows, best_row[1], best_indices


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


def run_cv(args: argparse.Namespace) -> None:
    splits = [load_split(*parse_split(value)) for value in args.split]
    features = {split["name"]: enhanced_features(split) for split in splits}
    targets = {split["name"]: target_indices(split) for split in splits}
    weight_grid = parse_grid(args.rerank_weight_grid)
    margin_grid = parse_grid(args.margin_grid)
    genus_frac_grid = parse_grid(args.genus_frac_grid)
    trigger_modes = [part.strip() for part in args.trigger_modes.split(",") if part.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv_rows = []
    fold_summaries = []
    for fold_idx, val_split in enumerate(splits):
        train_names = [split["name"] for split in splits if split["name"] != val_split["name"]]
        x_train = torch.cat([features[name] for name in train_names], dim=0)
        y_train = torch.cat([targets[name] for name in train_names], dim=0)
        x_val = features[val_split["name"]]
        x_train_std, x_val_std, _, _ = standardize(x_train, x_val)
        model, losses = train_model(
            x_train_std,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_idx,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        device = next(model.parameters()).device
        with torch.inference_mode():
            meta_scores = model(x_val_std.to(device)).cpu()
        sweep_rows, best, _ = evaluate_sweep(
            val_split,
            meta_scores,
            weight_grid=weight_grid,
            margin_grid=margin_grid,
            genus_frac_grid=genus_frac_grid,
            trigger_modes=trigger_modes,
        )
        for row in sweep_rows:
            cv_rows.append({"fold": val_split["name"], **row})
        fold_summaries.append(
            {
                "fold": val_split["name"],
                "train_folds": train_names,
                "train_rows": int((y_train >= 0).sum().item()),
                "val_rows": len(val_split["labels"]),
                "losses": losses,
                "best": best,
            }
        )

    with (args.out_dir / "cv_sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(cv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cv_rows)
    base_top1 = {
        split["name"]: metrics(torch.tensor(split["base_scores"], dtype=torch.float32).argsort(dim=1, descending=True), split)[
            "top1"
        ]
        for split in splits
    }
    best_by_fold = {item["fold"]: item["best"] for item in fold_summaries}
    gains = {name: best_by_fold[name]["top1"] - base_top1[name] for name in best_by_fold}
    summary = {
        "splits": [{"name": split["name"], "paths": split["paths"], "rows": len(split["labels"])} for split in splits],
        "feature_dim": int(next(iter(features.values())).shape[-1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "base_top1": base_top1,
        "best_by_fold": best_by_fold,
        "gains": gains,
        "avg_gain": sum(gains.values()) / len(gains),
        "min_gain": min(gains.values()),
        "fold_summaries": fold_summaries,
        "cv_sweep_csv": str(args.out_dir / "cv_sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def run_apply(args: argparse.Namespace) -> None:
    train_splits = [load_split(*parse_split(value)) for value in args.train_split]
    apply_split = load_split(*parse_split(args.apply_split))
    train_features = {split["name"]: enhanced_features(split) for split in train_splits}
    train_targets = {split["name"]: target_indices(split) for split in train_splits}
    x_train = torch.cat([train_features[split["name"]] for split in train_splits], dim=0)
    y_train = torch.cat([train_targets[split["name"]] for split in train_splits], dim=0)
    x_apply = enhanced_features(apply_split)
    x_train_std, x_apply_std, _, _ = standardize(x_train, x_apply)
    model, losses = train_model(
        x_train_std,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    device = next(model.parameters()).device
    with torch.inference_mode():
        meta_scores = model(x_apply_std.to(device)).cpu()
    base = torch.tensor(apply_split["base_scores"], dtype=torch.float32)
    trigger = trigger_mask(
        apply_split,
        mode=args.apply_trigger_mode,
        margin_threshold=args.apply_margin_threshold,
        genus_frac_threshold=args.apply_genus_frac_threshold,
    )
    final_scores = base + args.apply_rerank_weight * row_zscore(meta_scores)
    final_scores = torch.where(trigger[:, None], final_scores, base)
    indices = final_scores.argsort(dim=1, descending=True)
    setattr(indices, "triggered", int(trigger.sum().item()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / "predictions.csv", apply_split, indices)
    torch.save(
        {
            "image_ids": apply_split["image_ids"],
            "predictions": apply_split["predictions"],
            "base_scores": base,
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
        "feature_dim": int(x_train.shape[-1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "rerank_weight": args.apply_rerank_weight,
        "trigger_mode": args.apply_trigger_mode,
        "margin_threshold": args.apply_margin_threshold,
        "genus_frac_threshold": args.apply_genus_frac_threshold,
        "losses": losses,
        "changed": int((indices[:, 0] != 0).sum().item()),
        "triggered": int(trigger.sum().item()),
        "metrics_if_labeled": metrics(indices, apply_split) if any(apply_split["labels"]) else {},
        "predictions_csv": str(args.out_dir / "predictions.csv"),
        "score_file": str(args.out_dir / "meta_topk_scores.pt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", help="CV split: name=score1.pt|score2.pt")
    parser.add_argument("--train-split", action="append", help="Apply train split: name=score1.pt|score2.pt")
    parser.add_argument("--apply-split", help="Apply target split: name=score1.pt|score2.pt")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rerank-weight-grid", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--margin-grid", default="0.002,0.005,0.01,0.02,1.0")
    parser.add_argument("--genus-frac-grid", default="0.25,0.30,0.40,0.50,1.01")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered",
    )
    parser.add_argument("--apply-rerank-weight", type=float, default=0.05)
    parser.add_argument("--apply-trigger-mode", default="all")
    parser.add_argument("--apply-margin-threshold", type=float, default=0.01)
    parser.add_argument("--apply-genus-frac-threshold", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.apply_split:
        if not args.train_split:
            raise SystemExit("--train-split is required with --apply-split")
        run_apply(args)
    else:
        if not args.split:
            raise SystemExit("--split is required for CV mode")
        run_cv(args)


if __name__ == "__main__":
    main()
