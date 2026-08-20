from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.topk_jsonl.open("r", encoding="utf-8") as in_fp, args.out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_fp:
        writer = csv.DictWriter(
            out_fp,
            fieldnames=["image_id", "true_label", "prediction", "margin_top1_top2", "top_classes", "top_scores"],
        )
        writer.writeheader()
        for line in in_fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds = list(row.get("predictions", []))
            scores = [float(v) for v in row.get("scores", [])]
            margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "true_label": row.get("label", ""),
                    "prediction": preds[0] if preds else "",
                    "margin_top1_top2": margin,
                    "top_classes": "|".join(preds),
                    "top_scores": "|".join(str(v) for v in scores),
                }
            )


if __name__ == "__main__":
    main()
