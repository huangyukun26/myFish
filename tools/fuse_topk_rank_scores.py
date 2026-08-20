from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def load_topk(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()).div(x.std().clamp_min(1e-6))


def parse_weight_tuples(value: str, count: int) -> list[tuple[float, ...]]:
    tuples: list[tuple[float, ...]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        weights = tuple(float(part.strip()) for part in item.split(":") if part.strip())
        if len(weights) != count:
            raise ValueError(f"Weight tuple {item!r} has {len(weights)} values, expected {count}")
        tuples.append(weights)
    return tuples


def default_weight_tuples(count: int) -> list[tuple[float, ...]]:
    if count == 2:
        return [(1.0, w) for w in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]]
    if count == 3:
        values = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        return [(1.0, a, b) for a in values for b in values]
    raise ValueError("Provide --weight-tuples for more than three topK files")


def score_branch(preds: list[str], scores: list[float], method: str, rrf_k: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if method == "score_z":
        values = row_zscore(torch.tensor(scores, dtype=torch.float32)).tolist()
        for cls, score in zip(preds, values):
            out[cls] = float(score)
        return out
    if method == "rank":
        for rank, cls in enumerate(preds):
            out[cls] = 1.0 / float(rank + 1)
        return out
    if method == "rrf":
        for rank, cls in enumerate(preds):
            out[cls] = 1.0 / (rrf_k + float(rank + 1))
        return out
    raise ValueError(f"Unknown method: {method}")


def metrics(predictions: list[str], labels: list[str], base_predictions: list[str]) -> dict[str, Any]:
    known = 0
    correct = 0
    base_correct = 0
    changed = 0
    wins = 0
    losses = 0
    for pred, label, base_pred in zip(predictions, labels, base_predictions):
        if not label:
            continue
        known += 1
        ok = pred == label
        base_ok = base_pred == label
        correct += int(ok)
        base_correct += int(base_ok)
        changed += int(pred != base_pred)
        wins += int((not base_ok) and ok)
        losses += int(base_ok and (not ok))
    return {
        "known": known,
        "top1": correct / known if known else 0.0,
        "base_top1": base_correct / known if known else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def fuse(
    branch_rows: list[list[dict[str, Any]]],
    weights: tuple[float, ...],
    *,
    method: str,
    rrf_k: float,
    keep_base_on_tie: bool,
) -> list[str]:
    out: list[str] = []
    for row_idx in range(len(branch_rows[0])):
        totals: dict[str, float] = {}
        order_bonus: dict[str, int] = {}
        base_pred = branch_rows[0][row_idx]["predictions"][0]
        for branch_idx, rows in enumerate(branch_rows):
            row = rows[row_idx]
            branch_scores = score_branch(
                list(row["predictions"]),
                [float(v) for v in row["scores"]],
                method=method,
                rrf_k=rrf_k,
            )
            for rank, cls in enumerate(row["predictions"]):
                totals[cls] = totals.get(cls, 0.0) + weights[branch_idx] * branch_scores[cls]
                order_bonus.setdefault(cls, rank)
        if keep_base_on_tie:
            totals[base_pred] = totals.get(base_pred, 0.0) + 1e-8
        pred = max(totals, key=lambda cls: (totals[cls], -order_bonus.get(cls, 10_000)))
        out.append(pred)
    return out


def write_predictions(path: Path, rows: list[dict[str, Any]], predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "label", "base_prediction", "changed"])
        writer.writeheader()
        for row, pred in zip(rows, predictions):
            base_pred = row["predictions"][0]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "prediction": pred,
                    "label": row.get("label", ""),
                    "base_prediction": base_pred,
                    "changed": pred != base_pred,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-files", required=True, help="Comma-separated topK jsonl files. First one is the base.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method-grid", default="score_z,rank,rrf")
    parser.add_argument("--rrf-k-grid", default="10,30,60")
    parser.add_argument("--weight-tuples", default="")
    parser.add_argument("--apply-method", default="")
    parser.add_argument("--apply-rrf-k", type=float, default=None)
    parser.add_argument("--apply-weights", default="")
    args = parser.parse_args()

    paths = [Path(part.strip()) for part in args.topk_files.split(",") if part.strip()]
    branch_rows = [load_topk(path) for path in paths]
    base_ids = [row["image_id"] for row in branch_rows[0]]
    for path, rows in zip(paths[1:], branch_rows[1:]):
        ids = [row["image_id"] for row in rows]
        if ids != base_ids:
            raise RuntimeError(f"image_id order differs in {path}")

    labels = [row.get("label", "") for row in branch_rows[0]]
    base_predictions = [row["predictions"][0] for row in branch_rows[0]]

    if args.apply_method:
        methods = [args.apply_method]
        rrf_ks = [args.apply_rrf_k if args.apply_rrf_k is not None else 60.0]
        weight_tuples = parse_weight_tuples(args.apply_weights, len(paths))
    else:
        methods = [part.strip() for part in args.method_grid.split(",") if part.strip()]
        rrf_ks = [float(part.strip()) for part in args.rrf_k_grid.split(",") if part.strip()]
        weight_tuples = parse_weight_tuples(args.weight_tuples, len(paths)) if args.weight_tuples else default_weight_tuples(len(paths))

    rows = []
    best_key = None
    best_row = None
    best_predictions = None
    for method in methods:
        for rrf_k in rrf_ks:
            if method != "rrf" and rrf_k != rrf_ks[0]:
                continue
            for weights in weight_tuples:
                preds = fuse(branch_rows, weights, method=method, rrf_k=rrf_k, keep_base_on_tie=True)
                row = {
                    "method": method,
                    "rrf_k": rrf_k,
                    **{f"weight_{idx}": weight for idx, weight in enumerate(weights)},
                    **metrics(preds, labels, base_predictions),
                }
                rows.append(row)
                key = (row["top1"], row["net"], -row["losses"], -row["changed"])
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
                    best_predictions = preds

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assert best_row is not None and best_predictions is not None
    write_predictions(args.out_dir / "predictions.csv", branch_rows[0], best_predictions)
    summary = {
        "topk_files": [str(path) for path in paths],
        "rows": len(branch_rows[0]),
        "best": best_row,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
