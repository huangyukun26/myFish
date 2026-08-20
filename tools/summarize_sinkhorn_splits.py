from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONFIG_FIELDS = [
    "active_count",
    "active_mode",
    "union_topk",
    "tau",
    "blend",
    "prior_mode",
    "prior_alpha",
    "prior_uniform_mix",
    "row_weight_mode",
    "row_weight_floor",
    "row_weight_power",
    "row_weight_clean_fraction",
    "row_weight_scope",
]


def config_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in CONFIG_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--splits", default="species43,species44,genus43,genus44")
    args = parser.parse_args()

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    by_split: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    baseline_by_split: dict[str, dict[str, str]] = {}
    for split in splits:
        with (args.run_root / split / "sweep.csv").open(encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        by_split[split] = {config_key(row): row for row in rows}
        baseline = [row for row in rows if row["row_weight_mode"] == "none"]
        if len(baseline) != 1:
            raise RuntimeError(f"Expected one unweighted baseline for {split}, got {len(baseline)}")
        baseline_by_split[split] = baseline[0]

    shared_keys = set.intersection(*(set(rows) for rows in by_split.values()))
    summaries: list[dict[str, Any]] = []
    for key in shared_keys:
        first = by_split[splits[0]][key]
        summary: dict[str, Any] = {field: first[field] for field in CONFIG_FIELDS}
        delta_nets = []
        total_correct = 0.0
        total_known = 0
        for split in splits:
            row = by_split[split][key]
            baseline = baseline_by_split[split]
            known = int(row["known"])
            top1 = float(row["top1"])
            delta_net = int(row["net"]) - int(baseline["net"])
            summary[f"{split}_top1"] = top1
            summary[f"{split}_delta_net"] = delta_net
            summary[f"{split}_net_vs_independent"] = int(row["net"])
            delta_nets.append(delta_net)
            total_correct += top1 * known
            total_known += known
        summary["worst_delta_net"] = min(delta_nets)
        summary["total_delta_net"] = sum(delta_nets)
        summary["nonnegative_splits"] = sum(delta >= 0 for delta in delta_nets)
        summary["weighted_top1"] = total_correct / max(1, total_known)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            int(row["worst_delta_net"]),
            int(row["total_delta_net"]),
            float(row["weighted_top1"]),
        ),
        reverse=True,
    )
    output_csv = args.run_root / "robust_summary.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "splits": splits,
        "baseline": {
            split: {
                "top1": float(row["top1"]),
                "net_vs_independent": int(row["net"]),
            }
            for split, row in baseline_by_split.items()
        },
        "shared_configs": len(summaries),
        "top_robust_configs": summaries[:10],
        "robust_summary_csv": str(output_csv),
    }
    output_json = args.run_root / "robust_summary.json"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
