from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import List

import torch


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_named_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected name=value, got {value!r}")
    name, raw = value.split("=", 1)
    name = name.strip()
    raw = raw.strip()
    if not name or not raw:
        raise argparse.ArgumentTypeError(f"Expected name=value, got {value!r}")
    return name, raw


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_topk(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def evaluate_predictions(rows: list[dict], predictions: list[str]) -> dict:
    known = [idx for idx, row in enumerate(rows) if row.get("true_label")]
    if not known:
        return {}
    base_correct = []
    new_correct = []
    topk_hit = []
    changed_flags = []
    for idx in known:
        row = rows[idx]
        true_label = row["true_label"]
        top_classes = row["top_classes"].split("|")
        base_correct.append(top_classes[0] == true_label)
        new_correct.append(predictions[idx] == true_label)
        topk_hit.append(true_label in top_classes)
        changed_flags.append(top_classes[0] != predictions[idx])
    base = torch.tensor(base_correct, dtype=torch.bool)
    new = torch.tensor(new_correct, dtype=torch.bool)
    topk = torch.tensor(topk_hit, dtype=torch.bool)
    changed = torch.tensor(changed_flags, dtype=torch.bool)
    wins = ~base & new
    losses = base & ~new
    return {
        "known": len(known),
        "base_top1": float(base.float().mean().item()),
        "new_top1": float(new.float().mean().item()),
        "topk_recall": float(topk.float().mean().item()),
        "changed": int(changed.sum().item()),
        "changed_frac": float(changed.float().mean().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net_wins": int(wins.sum().item() - losses.sum().item()),
        "win_loss_ratio": float(wins.sum().item() / max(1, losses.sum().item())),
    }


def load_score_cache(path: Path, expected_rows: int, expected_k: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    scores = payload["scores"].float()
    if scores.shape != (expected_rows, expected_k):
        raise RuntimeError(
            f"{path} scores shape {tuple(scores.shape)} does not match expected {(expected_rows, expected_k)}"
        )
    return row_zscore(scores)


def rerank_rows(
    rows: list[dict],
    normalized_score_caches: list[torch.Tensor],
    weights: list[float],
    margin_threshold: float,
) -> list[str]:
    predictions: list[str] = []
    for idx, row in enumerate(rows):
        top_classes = row["top_classes"].split("|")
        base_scores = torch.tensor([float(value) for value in row["top_scores"].split("|")], dtype=torch.float32)
        margin = float(row["margin_top1_top2"])
        if margin > margin_threshold or all(weight == 0 for weight in weights):
            predictions.append(top_classes[0])
            continue
        final_scores = base_scores.clone()
        for cache, weight in zip(normalized_score_caches, weights):
            if weight:
                final_scores += weight * cache[idx]
        predictions.append(top_classes[int(final_scores.argmax().item())])
    return predictions


def write_predictions(path: Path, rows: list[dict], predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"image_id": row["image_id"], "prediction": prediction})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-csv", type=Path, required=True)
    parser.add_argument(
        "--score-cache",
        action="append",
        required=True,
        help="Repeatable name=path entry pointing to neighbor_candidate_scores.pt.",
    )
    parser.add_argument(
        "--weight-grid",
        action="append",
        required=True,
        help="Repeatable name=v1,v2,... entry. Names must match --score-cache.",
    )
    parser.add_argument("--margin-threshold-grid", default="0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_topk(args.topk_csv)
    if not rows:
        raise RuntimeError(f"No rows in {args.topk_csv}")
    expected_k = len(rows[0]["top_classes"].split("|"))

    cache_entries = [parse_named_value(value) for value in args.score_cache]
    grid_entries = dict(parse_named_value(value) for value in args.weight_grid)
    cache_names = [name for name, _path in cache_entries]
    missing_grids = [name for name in cache_names if name not in grid_entries]
    if missing_grids:
        raise RuntimeError(f"Missing --weight-grid entries for: {missing_grids}")

    score_caches = [
        load_score_cache(Path(raw_path), expected_rows=len(rows), expected_k=expected_k) for _name, raw_path in cache_entries
    ]
    weight_grids = [parse_grid(grid_entries[name]) for name in cache_names]
    margin_grid = parse_grid(args.margin_threshold_grid)

    sweep_rows = []
    best_predictions: list[str] = []
    best_row = None
    best_key = None
    for margin_threshold in margin_grid:
        for weights in itertools.product(*weight_grids):
            predictions = rerank_rows(
                rows=rows,
                normalized_score_caches=score_caches,
                weights=list(weights),
                margin_threshold=margin_threshold,
            )
            metrics = evaluate_predictions(rows, predictions)
            row = {
                "margin_threshold": margin_threshold,
                **{f"weight_{name}": weight for name, weight in zip(cache_names, weights)},
                **metrics,
            }
            sweep_rows.append(row)
            if metrics:
                key = (metrics["new_top1"], metrics["net_wins"], -metrics["losses"], -metrics["changed"])
            else:
                key = (-sum(abs(weight) for weight in weights), -margin_threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
                best_predictions = predictions

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    write_predictions(args.out_dir / "best_predictions.csv", rows, best_predictions)
    summary = {
        "topk_csv": str(args.topk_csv),
        "score_caches": {name: raw_path for name, raw_path in cache_entries},
        "weight_grids": {name: grid_entries[name] for name in cache_names},
        "margin_threshold_grid": args.margin_threshold_grid,
        "rows": len(rows),
        "best": best_row,
        "out_predictions": str(args.out_dir / "best_predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
