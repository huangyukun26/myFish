from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_predictions(path: Path) -> Dict[str, str]:
    return {row["image_id"]: row["prediction"] for row in read_csv_rows(path)}


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def margin_bin(value: float) -> str:
    if value <= 0.02:
        return "<=0.02"
    if value <= 0.05:
        return "0.02-0.05"
    if value <= 0.10:
        return "0.05-0.10"
    if value <= 0.20:
        return "0.10-0.20"
    if value <= 0.30:
        return "0.20-0.30"
    if value <= 0.50:
        return "0.30-0.50"
    return ">0.50"


def class_counts(manifest: Path | None) -> Counter:
    counts: Counter = Counter()
    if manifest is None:
        return counts
    for row in read_csv_rows(manifest):
        label = row.get("label", "")
        if label:
            counts[label] += 1
    return counts


def summarize_group(rows: Iterable[dict], key: str) -> List[dict]:
    grouped = defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0, "neutral": 0, "net": 0})
    for row in rows:
        item = grouped[row[key]]
        item["count"] += 1
        item[row["outcome"]] += 1
        if row["outcome"] == "wins":
            item["net"] += 1
        elif row["outcome"] == "losses":
            item["net"] -= 1
    out = []
    for value, item in grouped.items():
        ratio = item["wins"] / max(1, item["losses"])
        out.append({"group": value, **item, "win_loss_ratio": ratio})
    return sorted(out, key=lambda item: (item["net"], item["wins"], -item["losses"]), reverse=True)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-csv", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=80)
    args = parser.parse_args()

    topk_rows = read_csv_rows(args.topk_csv)
    base = read_predictions(args.base_csv)
    new = read_predictions(args.new_csv)
    counts = class_counts(args.train_manifest)

    audited = []
    for row in topk_rows:
        image_id = row["image_id"]
        true_label = row.get("true_label", "")
        base_pred = base.get(image_id, row["prediction"])
        new_pred = new.get(image_id, base_pred)
        if base_pred == new_pred:
            continue
        base_correct = bool(true_label and base_pred == true_label)
        new_correct = bool(true_label and new_pred == true_label)
        if (not base_correct) and new_correct:
            outcome = "wins"
        elif base_correct and (not new_correct):
            outcome = "losses"
        else:
            outcome = "neutral"
        margin = float(row["margin_top1_top2"])
        top_classes = row["top_classes"].split("|")
        audited.append(
            {
                "image_id": image_id,
                "true_label": true_label,
                "base_prediction": base_pred,
                "new_prediction": new_pred,
                "outcome": outcome,
                "margin": margin,
                "margin_bin": margin_bin(margin),
                "base_genus": genus(base_pred),
                "new_genus": genus(new_pred),
                "true_genus": genus(true_label),
                "base_true_same_genus": str(bool(true_label and genus(base_pred) == genus(true_label))),
                "new_true_same_genus": str(bool(true_label and genus(new_pred) == genus(true_label))),
                "base_new_same_genus": str(genus(base_pred) == genus(new_pred)),
                "true_in_top20": str(true_label in top_classes if true_label else ""),
                "true_rank_in_top20": top_classes.index(true_label) + 1 if true_label in top_classes else "",
                "base_train_count": counts.get(base_pred, 0),
                "new_train_count": counts.get(new_pred, 0),
                "true_train_count": counts.get(true_label, 0),
            }
        )

    total = len(topk_rows)
    changed = len(audited)
    wins = sum(row["outcome"] == "wins" for row in audited)
    losses = sum(row["outcome"] == "losses" for row in audited)
    neutral = changed - wins - losses
    summary = {
        "topk_csv": str(args.topk_csv),
        "base_csv": str(args.base_csv),
        "new_csv": str(args.new_csv),
        "rows": total,
        "changed": changed,
        "changed_frac": changed / total if total else 0,
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "net_wins": wins - losses,
        "win_loss_ratio": wins / max(1, losses),
        "by_margin": summarize_group(audited, "margin_bin"),
        "by_base_new_same_genus": summarize_group(audited, "base_new_same_genus"),
        "by_new_true_same_genus": summarize_group(audited, "new_true_same_genus"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "changed_examples.csv", audited)
    examples = []
    for outcome in ["wins", "losses", "neutral"]:
        examples.extend([row for row in audited if row["outcome"] == outcome][: args.max_examples])
    write_csv(args.out_dir / "sampled_examples.csv", examples)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
