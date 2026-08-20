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


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_split(value: str) -> tuple[str, list[Path]]:
    name, sep, rest = value.partition("=")
    if not sep or not name.strip() or not rest.strip():
        raise ValueError(f"Bad --split value: {value!r}. Use name=path1|path2")
    return name.strip(), [Path(part.strip()) for part in rest.split("|") if part.strip()]


def load_split(name: str, paths: list[Path]) -> dict[str, Any]:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["image_ids"] != base["image_ids"]:
            raise RuntimeError(f"image_ids differ in {name}: {path}")
        if payload["predictions"] != base["predictions"]:
            raise RuntimeError(f"predictions differ in {name}: {path}")
    return {
        "name": name,
        "paths": [str(path) for path in paths],
        "image_ids": base["image_ids"],
        "predictions": base["predictions"],
        "base_scores": torch.tensor(base["base_scores"], dtype=torch.float32),
        "labels": base["labels"],
        "adapter_scores": [payload["adapter_scores"].float() for payload in payloads],
    }


def target_indices(split: dict[str, Any]) -> torch.Tensor:
    targets = []
    for label, preds in zip(split["labels"], split["predictions"]):
        try:
            targets.append(preds.index(label))
        except ValueError:
            targets.append(-1)
    return torch.tensor(targets, dtype=torch.long)


def build_features(split: dict[str, Any]) -> torch.Tensor:
    base = split["base_scores"]
    n, k = base.shape
    rank = torch.linspace(0.0, 1.0, steps=k).view(1, k).expand(n, k)
    inv_rank = 1.0 - rank
    base_z = row_zscore(base)
    base_center = base - base.mean(dim=1, keepdim=True)
    base_delta_top = base - base[:, :1]
    top12_margin = (base[:, :1] - base[:, 1:2]).expand(n, k)
    top15_margin = (base[:, :1] - base[:, 4:5]).expand(n, k) if k >= 5 else top12_margin

    adapter_features = []
    for scores in split["adapter_scores"]:
        adapter_features.append(row_zscore(scores))
        adapter_features.append(scores - scores.mean(dim=1, keepdim=True))

    genus_count_frac = torch.zeros((n, k), dtype=torch.float32)
    same_as_top1_genus = torch.zeros((n, k), dtype=torch.float32)
    top1_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    dominant_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    distinct_genus_frac = torch.zeros((n, k), dtype=torch.float32)
    for row_idx, preds in enumerate(split["predictions"]):
        genera = [genus(pred) for pred in preds]
        counts = Counter(genera)
        top_g = genera[0] if genera else ""
        top_frac = counts.get(top_g, 0) / max(1, k)
        dom_frac = max(counts.values()) / max(1, k) if counts else 0.0
        distinct = len(counts) / max(1, k)
        for col, g in enumerate(genera):
            genus_count_frac[row_idx, col] = counts[g] / max(1, k)
            same_as_top1_genus[row_idx, col] = float(g == top_g)
        top1_genus_frac[row_idx] = top_frac
        dominant_genus_frac[row_idx] = dom_frac
        distinct_genus_frac[row_idx] = distinct

    pieces = [
        base,
        base_z,
        base_center,
        base_delta_top,
        rank,
        inv_rank,
        top12_margin,
        top15_margin,
        genus_count_frac,
        same_as_top1_genus,
        top1_genus_frac,
        dominant_genus_frac,
        distinct_genus_frac,
        *adapter_features,
    ]
    return torch.stack(pieces, dim=2)


def standardize(train_x: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = train_x.flatten(0, 1)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)
    return (train_x - mean) / std, (x - mean) / std, mean


class CandidateScorer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_model(
    x: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[CandidateScorer, list[float]]:
    torch.manual_seed(seed)
    keep = targets >= 0
    x = x[keep]
    targets = targets[keep]
    if not len(targets):
        raise RuntimeError("No train rows have the true label in topK")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateScorer(x.shape[-1]).to(device)
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


def trigger_mask(
    split: dict[str, Any],
    *,
    mode: str,
    margin_threshold: float,
    genus_frac_threshold: float,
) -> torch.Tensor:
    base = split["base_scores"]
    values = []
    for row_idx, preds in enumerate(split["predictions"]):
        margin = float(base[row_idx, 0] - base[row_idx, 1]) if base.shape[1] > 1 else 0.0
        genera = [genus(pred) for pred in preds]
        counts = Counter(genera)
        top1_frac = counts.get(genera[0], 0) / max(1, len(genera)) if genera else 0.0
        low_margin = margin <= margin_threshold
        clustered = top1_frac >= genus_frac_threshold
        if mode == "all":
            values.append(True)
        elif mode == "low_margin":
            values.append(low_margin)
        elif mode == "clustered":
            values.append(clustered)
        elif mode == "low_margin_or_clustered":
            values.append(low_margin or clustered)
        elif mode == "low_margin_and_clustered":
            values.append(low_margin and clustered)
        else:
            raise ValueError(f"Unknown trigger mode: {mode}")
    return torch.tensor(values, dtype=torch.bool)


def metrics(indices: torch.Tensor, split: dict[str, Any]) -> dict[str, Any]:
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    for row_idx, label in enumerate(split["labels"]):
        preds = split["predictions"][row_idx]
        base_pred = preds[0]
        final_pred = preds[int(indices[row_idx, 0].item())]
        base_ok = base_pred == label
        final_ok = final_pred == label
        changed += int(base_pred != final_pred)
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [preds[int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks)
    return {
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "triggered": int(getattr(indices, "triggered", 0)),
    }


def evaluate_sweep(
    split: dict[str, Any],
    meta_scores: torch.Tensor,
    *,
    weight_grid: list[float],
    margin_grid: list[float],
    genus_frac_grid: list[float],
    trigger_modes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor]:
    base = split["base_scores"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, help="name=score1.pt|score2.pt")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rerank-weight-grid", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--margin-grid", default="0.002,0.005,0.01,0.02,1.0")
    parser.add_argument("--genus-frac-grid", default="0.25,0.30,0.40,1.01")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = [load_split(*parse_split(value)) for value in args.split]
    features = {split["name"]: build_features(split) for split in splits}
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
        x_train_std, x_val_std, _mean = standardize(x_train, x_val)
        model, losses = train_model(
            x_train_std,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_idx,
        )
        device = next(model.parameters()).device
        with torch.inference_mode():
            meta_scores = model(x_val_std.to(device)).cpu()
        sweep_rows, best, _best_indices = evaluate_sweep(
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
    base_top1 = {split["name"]: metrics(split["base_scores"].argsort(dim=1, descending=True), split)["top1"] for split in splits}
    best_by_fold = {item["fold"]: item["best"] for item in fold_summaries}
    gains = {name: best_by_fold[name]["top1"] - base_top1[name] for name in best_by_fold}
    summary = {
        "splits": [{"name": split["name"], "paths": split["paths"], "rows": len(split["labels"])} for split in splits],
        "feature_dim": int(next(iter(features.values())).shape[-1]),
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


if __name__ == "__main__":
    main()
