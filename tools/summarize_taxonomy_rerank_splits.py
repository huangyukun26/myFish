from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--splits", default="species43,species44,genus43,genus44")
    args = parser.parse_args()

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    by_split: dict[str, dict[str, dict[str, str]]] = {}
    for split in splits:
        with (args.run_root / split / "sweep.csv").open(encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        by_split[split] = {row["config_id"]: row for row in rows}

    shared = set.intersection(*(set(rows) for rows in by_split.values()))
    summaries: list[dict[str, Any]] = []
    for config_id in shared:
        first = by_split[splits[0]][config_id]
        summary: dict[str, Any] = {
            "config_id": config_id,
            "topk": int(first["topk"]),
            "branch_source": first["branch_source"],
            "species_weight": float(first["species_weight"]),
            "mode": first["mode"],
            "support_weight": float(first["support_weight"]),
            "support_temperature": float(first["support_temperature"]),
        }
        if "novelty_gate" in first:
            summary["novelty_gate"] = first["novelty_gate"]
        nets = []
        for split in splits:
            row = by_split[split][config_id]
            net = int(row["net"])
            nets.append(net)
            summary[f"{split}_top1"] = float(row["top1"])
            summary[f"{split}_changed"] = int(row["changed"])
            summary[f"{split}_wins"] = int(row["wins"])
            summary[f"{split}_losses"] = int(row["losses"])
            summary[f"{split}_net"] = net
        summary["worst_net"] = min(nets)
        summary["total_net"] = sum(nets)
        summary["nonnegative_splits"] = sum(net >= 0 for net in nets)
        summary["total_changed"] = sum(summary[f"{split}_changed"] for split in splits)
        summary["total_wins"] = sum(summary[f"{split}_wins"] for split in splits)
        summary["total_losses"] = sum(summary[f"{split}_losses"] for split in splits)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["worst_net"],
            row["total_net"],
            -row["total_losses"],
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
        "shared_configs": len(summaries),
        "top_robust_configs": summaries[:10],
        "robust_summary_csv": str(output_csv),
    }
    (args.run_root / "robust_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
