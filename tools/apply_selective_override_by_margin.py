from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_predictions(path: Path) -> dict[str, str]:
    return {row["image_id"]: row["prediction"] for row in read_rows(path)}


def read_margins(path: Path) -> dict[str, float]:
    margins = {}
    for row in read_rows(path):
        try:
            margins[row["image_id"]] = float(row.get("margin_top1_top2", 0.0))
        except ValueError:
            margins[row["image_id"]] = 0.0
    return margins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--base-topk-csv", type=Path, required=True)
    parser.add_argument("--candidate-topk-csv", type=Path, required=True)
    parser.add_argument("--base-margin-max", type=float, required=True)
    parser.add_argument("--candidate-margin-min", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.base_csv)
    base = {row["image_id"]: row["prediction"] for row in rows}
    candidate = read_predictions(args.candidate_csv)
    base_margin = read_margins(args.base_topk_csv)
    candidate_margin = read_margins(args.candidate_topk_csv)

    final: dict[str, str] = {}
    changed = 0
    triggered = 0
    for row in rows:
        image_id = row["image_id"]
        if image_id not in candidate:
            raise RuntimeError(f"{image_id} missing from candidate predictions")
        use_candidate = (
            base[image_id] != candidate[image_id]
            and base_margin.get(image_id, 0.0) <= args.base_margin_max
            and candidate_margin.get(image_id, 0.0) >= args.candidate_margin_min
        )
        triggered += int(use_candidate)
        pred = candidate[image_id] if use_candidate else base[image_id]
        changed += int(pred != base[image_id])
        final[image_id] = pred

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for row in rows:
            image_id = row["image_id"]
            writer.writerow({"image_id": image_id, "prediction": final[image_id]})

    summary = {
        "base_csv": str(args.base_csv),
        "candidate_csv": str(args.candidate_csv),
        "base_topk_csv": str(args.base_topk_csv),
        "candidate_topk_csv": str(args.candidate_topk_csv),
        "base_margin_max": args.base_margin_max,
        "candidate_margin_min": args.candidate_margin_min,
        "rows": len(rows),
        "triggered": triggered,
        "changed_vs_base": changed,
        "changed_frac": changed / len(rows) if rows else 0.0,
        "predictions_csv": str(pred_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
