from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def zscore(values: list[float]) -> list[float]:
    mean = sum(values) / max(1, len(values))
    var = sum((value - mean) ** 2 for value in values) / max(1, len(values))
    std = math.sqrt(var) or 1.0
    return [(value - mean) / std for value in values]


def logsumexp(values: list[float]) -> float:
    if not values:
        return 0.0
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def genus_scores(predictions: list[str], scores: list[float], mode: str) -> list[float]:
    by_genus: dict[str, list[float]] = defaultdict(list)
    for pred, score in zip(predictions, scores):
        by_genus[genus(pred)].append(score)
    out_by_genus: dict[str, float] = {}
    for key, values in by_genus.items():
        if mode == "count":
            out_by_genus[key] = float(len(values))
        elif mode == "sum":
            out_by_genus[key] = sum(values)
        elif mode == "mean":
            out_by_genus[key] = sum(values) / len(values)
        elif mode == "max":
            out_by_genus[key] = max(values)
        elif mode == "logsumexp":
            out_by_genus[key] = logsumexp(values)
        else:
            raise ValueError(f"Unknown genus score mode: {mode}")
    return [out_by_genus[genus(pred)] for pred in predictions]


def should_trigger(
    *,
    mode: str,
    margin: float,
    margin_threshold: float,
    genus_frac: float,
    genus_frac_threshold: float,
) -> bool:
    low_margin = margin <= margin_threshold
    clustered = genus_frac >= genus_frac_threshold
    if mode == "all":
        return True
    if mode == "low_margin":
        return low_margin
    if mode == "clustered":
        return clustered
    if mode == "low_margin_or_clustered":
        return low_margin or clustered
    if mode == "low_margin_and_clustered":
        return low_margin and clustered
    raise ValueError(f"Unknown trigger mode: {mode}")


def rerank_row(
    row: dict[str, Any],
    *,
    topk: int,
    weight: float,
    genus_mode: str,
    trigger_mode: str,
    margin_threshold: float,
    genus_frac_threshold: float,
) -> tuple[str, list[str], bool]:
    predictions = list(row.get("predictions", []))[:topk]
    scores = [float(v) for v in row.get("scores", [])[: len(predictions)]]
    if not predictions:
        return "", [], False
    margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
    counts = Counter(genus(pred) for pred in predictions)
    genus_frac = counts.most_common(1)[0][1] / len(predictions)
    trigger = should_trigger(
        mode=trigger_mode,
        margin=margin,
        margin_threshold=margin_threshold,
        genus_frac=genus_frac,
        genus_frac_threshold=genus_frac_threshold,
    )
    if not trigger or weight == 0:
        return predictions[0], predictions, False
    base_z = zscore(scores)
    genus_z = zscore(genus_scores(predictions, scores, genus_mode))
    final = [base + weight * gen for base, gen in zip(base_z, genus_z)]
    order = sorted(range(len(predictions)), key=lambda idx: final[idx], reverse=True)
    ranked = [predictions[idx] for idx in order]
    return ranked[0], ranked, ranked[0] != predictions[0]


def evaluate(rows: list[dict[str, Any]], final_predictions: list[str], ranked_predictions: list[list[str]]) -> dict[str, Any]:
    known = [idx for idx, row in enumerate(rows) if row.get("label")]
    if not known:
        return {}
    ranks = []
    wins = losses = changed = 0
    for idx in known:
        row = rows[idx]
        label = row["label"]
        base = row["predictions"][0]
        final = final_predictions[idx]
        base_ok = base == label
        final_ok = final == label
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        changed += int(base != final)
        try:
            ranks.append(ranked_predictions[idx].index(label) + 1)
        except ValueError:
            ranks.append(len(ranked_predictions[idx]) + 1)
    return {
        "known": len(known),
        "base_top1": sum(rows[idx]["predictions"][0] == rows[idx]["label"] for idx in known) / len(known),
        "new_top1": sum(final_predictions[idx] == rows[idx]["label"] for idx in known) / len(known),
        "top5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "top20": sum(rank <= 20 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "win_loss_ratio": wins / max(1, losses),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]], predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for row, pred in zip(rows, predictions):
            writer.writerow({"image_id": row["image_id"], "prediction": pred})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--weight-grid", default="0,0.05,0.1,0.2,0.3,0.5,0.8,1.0")
    parser.add_argument("--genus-mode", choices=["count", "sum", "mean", "max", "logsumexp"], default="count")
    parser.add_argument("--trigger-mode", choices=["all", "low_margin", "clustered", "low_margin_or_clustered", "low_margin_and_clustered"], default="all")
    parser.add_argument("--margin-threshold-grid", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--genus-frac-threshold-grid", default="0.25,0.35,0.5")
    args = parser.parse_args()

    rows = read_jsonl(args.topk_jsonl)
    sweep = []
    best = None
    best_predictions: list[str] | None = None
    for weight in parse_grid(args.weight_grid):
        for margin_threshold in parse_grid(args.margin_threshold_grid):
            for genus_frac_threshold in parse_grid(args.genus_frac_threshold_grid):
                predictions = []
                ranked_rows = []
                changed_by_trigger = 0
                for row in rows:
                    pred, ranked, changed = rerank_row(
                        row,
                        topk=args.topk,
                        weight=weight,
                        genus_mode=args.genus_mode,
                        trigger_mode=args.trigger_mode,
                        margin_threshold=margin_threshold,
                        genus_frac_threshold=genus_frac_threshold,
                    )
                    predictions.append(pred)
                    ranked_rows.append(ranked)
                    changed_by_trigger += int(changed)
                row = {
                    "weight": weight,
                    "margin_threshold": margin_threshold,
                    "genus_frac_threshold": genus_frac_threshold,
                    "genus_mode": args.genus_mode,
                    "trigger_mode": args.trigger_mode,
                    "trigger_changed": changed_by_trigger,
                    **evaluate(rows, predictions, ranked_rows),
                }
                sweep.append(row)
                key = (
                    row.get("new_top1", 0.0),
                    row.get("net_wins", 0),
                    row.get("mrr", 0.0),
                    -row.get("losses", 0),
                    -row.get("changed", 0),
                )
                if best is None or key > best[0]:
                    best = (key, row)
                    best_predictions = predictions

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    if best_predictions is not None:
        write_predictions(args.out_dir / "best_predictions.csv", rows, best_predictions)
    summary = {
        "topk_jsonl": str(args.topk_jsonl),
        "rows": len(rows),
        "topk": args.topk,
        "best": best[1] if best else None,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "best_predictions": str(args.out_dir / "best_predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
