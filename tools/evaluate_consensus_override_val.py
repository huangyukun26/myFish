#!/usr/bin/env python
"""Evaluate consensus overrides on labeled validation prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_pred(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if not rows:
        return {}
    fields = rows[0].keys()
    key = "prediction" if "prediction" in fields else "label" if "label" in fields else None
    if key is None:
        raise ValueError(f"{path} has no prediction/label column")
    return {row["image_id"]: row[key] for row in rows if row.get("image_id") and row.get(key)}


def read_labels(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        labels: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = row.get("label") or row.get("true_label")
                image_id = row.get("image_id")
                if image_id and label:
                    labels[image_id] = label
        return labels
    labels: dict[str, str] = {}
    for row in read_rows(path):
        label = row.get("true_label") or row.get("label")
        if label:
            labels[row["image_id"]] = label
    return labels


def genus(label: str) -> str:
    parts = str(label or "").split()
    return parts[0] if parts else ""


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_bool_grid(value: str) -> list[bool]:
    out = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        out.append(key in {"1", "true", "yes"})
    return out or [False]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(labels: dict[str, str], base: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = losses = neutral = 0
    for row in rows:
        image_id = row["image_id"]
        label = labels[image_id]
        before = base[image_id] == label
        after = row["candidate_prediction"] == label
        if after and not before:
            wins += 1
        elif before and not after:
            losses += 1
        else:
            neutral += 1
    changed = len(rows)
    return {
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "net": wins - losses,
        "efficiency": (wins - losses) / max(1, changed),
        "win_loss_ratio": wins / max(1, losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-votes-grid", default="2,3,4,5,6")
    parser.add_argument("--vote-margin-grid", default="1,2,3")
    parser.add_argument("--same-genus-grid", default="false,true")
    parser.add_argument("--max-changed-grid", default="25,50,100,200,400,800,1200")
    args = parser.parse_args()

    base = read_pred(args.base_csv)
    labels = read_labels(args.label_csv)
    image_ids = [image_id for image_id in base if image_id in labels]
    sources = []
    for path in args.source:
        pred = read_pred(path)
        covered = sum(1 for image_id in image_ids if image_id in pred)
        changed = sum(1 for image_id in image_ids if pred.get(image_id) and pred.get(image_id) != base[image_id])
        sources.append({"path": str(path), "pred": pred, "covered": covered, "changed_vs_base": changed})

    candidate_rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        counts: Counter[str] = Counter()
        coverage = 0
        for src in sources:
            label = src["pred"].get(image_id)
            if not label:
                continue
            coverage += 1
            if label != base[image_id]:
                counts[label] += 1
        if not counts:
            continue
        label, votes = counts.most_common(1)[0]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        candidate_rows.append(
            {
                "image_id": image_id,
                "true_label": labels[image_id],
                "base_prediction": base[image_id],
                "candidate_prediction": label,
                "candidate_votes": votes,
                "runner_up_votes": runner_up,
                "vote_margin": votes - runner_up,
                "source_coverage": coverage,
                "same_genus": genus(base[image_id]) == genus(label),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            int(row["candidate_votes"]),
            int(row["vote_margin"]),
            int(row["source_coverage"]),
            int(row["same_genus"]),
        ),
        reverse=True,
    )

    sweep: list[dict[str, Any]] = []
    best_by_net: list[dict[str, Any]] = []
    for min_votes in parse_ints(args.min_votes_grid):
        for margin in parse_ints(args.vote_margin_grid):
            for same_genus_only in parse_bool_grid(args.same_genus_grid):
                eligible = [
                    row
                    for row in candidate_rows
                    if int(row["candidate_votes"]) >= min_votes
                    and int(row["vote_margin"]) >= margin
                    and (not same_genus_only or bool(row["same_genus"]))
                ]
                for cap in parse_ints(args.max_changed_grid):
                    selected = eligible[:cap]
                    if not selected:
                        continue
                    row = {
                        "min_votes": min_votes,
                        "vote_margin": margin,
                        "same_genus_only": same_genus_only,
                        "cap": cap,
                        "eligible": len(eligible),
                        **evaluate(labels, base, selected),
                    }
                    sweep.append(row)
                    best_by_net.append(row)

    best_by_net.sort(key=lambda row: (row["net"], row["efficiency"], -row["losses"], row["changed"]), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "candidate_rows.csv", candidate_rows)
    write_csv(args.out_dir / "sweep.csv", sweep)
    summary = {
        "base_csv": str(args.base_csv),
        "label_csv": str(args.label_csv),
        "rows": len(image_ids),
        "base_correct": sum(1 for image_id in image_ids if base[image_id] == labels[image_id]),
        "sources": [{"path": src["path"], "covered": src["covered"], "changed_vs_base": src["changed_vs_base"]} for src in sources],
        "candidate_rows": len(candidate_rows),
        "best_by_net": best_by_net[:30],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
