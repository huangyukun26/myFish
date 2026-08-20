#!/usr/bin/env python
"""Build a public-data target species list from validation-dev top-k failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def read_topk_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "image_id": row["image_id"],
                    "label": row.get("label"),
                    "predictions": list(row["predictions"]),
                    "scores": [float(x) for x in row["scores"]],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-species", type=int, default=300)
    parser.add_argument("--split-prefix", default="external-support:")
    parser.add_argument("--include-dev-only", action="store_true")
    args = parser.parse_args()

    rows = read_topk_jsonl(args.topk)
    true_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    hard_rows = []
    for row in rows:
        if args.include_dev_only and stable_hash(args.split_prefix + row["image_id"]) % 2 != 0:
            continue
        label = row["label"]
        preds = row["predictions"][: args.k]
        if not label or not preds:
            continue
        base = preds[0]
        if base == label:
            continue
        if label not in preds:
            continue
        true_rank = preds.index(label) + 1
        margin = row["scores"][0] - row["scores"][1] if len(row["scores"]) > 1 else 999.0
        true_counter[label] += 1
        for pred in preds[: min(args.k, len(preds))]:
            pred_counter[pred] += 1
            cooc[label][pred] += 1
        hard_rows.append(
            {
                "image_id": row["image_id"],
                "label": label,
                "base_pred": base,
                "true_rank": true_rank,
                "base_margin": margin,
                "topk": json.dumps(preds, ensure_ascii=False),
            }
        )

    selected: list[dict[str, Any]] = []
    seen_species: set[str] = set()
    for label, count in true_counter.most_common():
        if label in seen_species:
            continue
        selected.append(
            {
                "species": label,
                "role": "true_label",
                "true_error_count": count,
                "pred_count": pred_counter[label],
                "score": count * 1000 + pred_counter[label],
            }
        )
        seen_species.add(label)
        if len(selected) >= args.max_species:
            break
    for label, count in pred_counter.most_common():
        if len(selected) >= args.max_species:
            break
        if label in seen_species:
            continue
        selected.append(
            {
                "species": label,
                "role": "topk_candidate",
                "true_error_count": true_counter[label],
                "pred_count": count,
                "score": true_counter[label] * 1000 + count,
            }
        )
        seen_species.add(label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    species = [row["species"] for row in selected]
    (args.out_dir / "species_targets.json").write_text(json.dumps(species, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "species_targets.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["species", "role", "true_error_count", "pred_count", "score"])
        writer.writeheader()
        writer.writerows(selected)
    with (args.out_dir / "hard_rows.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "label", "base_pred", "true_rank", "base_margin", "topk"])
        writer.writeheader()
        writer.writerows(hard_rows)
    summary = {
        "topk": str(args.topk),
        "k": args.k,
        "dev_only": args.include_dev_only,
        "hard_rows": len(hard_rows),
        "unique_true_labels": len(true_counter),
        "unique_topk_candidates": len(pred_counter),
        "selected_species": len(selected),
        "top_true_labels": true_counter.most_common(30),
        "out_species_json": str(args.out_dir / "species_targets.json"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
