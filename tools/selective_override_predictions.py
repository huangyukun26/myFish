from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_predictions(path: Path) -> dict[str, str]:
    return {row["image_id"]: row["prediction"] for row in read_rows(path)}


def read_margins(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_rows(path):
        try:
            out[row["image_id"]] = float(row["margin_top1_top2"])
        except (KeyError, TypeError, ValueError):
            out[row["image_id"]] = 0.0
    return out


def read_labels(path: Path) -> dict[str, str]:
    labels = {}
    for row in read_rows(path):
        label = row.get("true_label") or row.get("label") or ""
        if label:
            labels[row["image_id"]] = label
    return labels


def evaluate(image_ids: list[str], labels: dict[str, str], base: dict[str, str], final: dict[str, str]) -> dict:
    known = [image_id for image_id in image_ids if labels.get(image_id)]
    if not known:
        return {}
    base_ok = [base.get(image_id) == labels[image_id] for image_id in known]
    final_ok = [final.get(image_id) == labels[image_id] for image_id in known]
    changed = [base.get(image_id) != final.get(image_id) for image_id in known]
    wins = sum((not b) and f for b, f in zip(base_ok, final_ok))
    losses = sum(b and (not f) for b, f in zip(base_ok, final_ok))
    return {
        "known": len(known),
        "base_top1": sum(base_ok) / len(known),
        "new_top1": sum(final_ok) / len(known),
        "changed": sum(changed),
        "changed_frac": sum(changed) / len(known),
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "win_loss_ratio": wins / max(1, losses),
    }


def write_predictions(path: Path, image_ids: list[str], final: dict[str, str], labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fields = ["image_id", "prediction"]
        if labels:
            fields.append("label")
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for image_id in image_ids:
            row = {"image_id": image_id, "prediction": final[image_id]}
            if labels:
                row["label"] = labels.get(image_id, "")
            writer.writerow(row)


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--base-topk-csv", type=Path, required=True)
    parser.add_argument("--candidate-topk-csv", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, default=None)
    parser.add_argument("--base-margin-max-grid", default="0.05,0.1,0.2,0.3,0.4,0.5,1.0")
    parser.add_argument("--candidate-margin-min-grid", default="0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = read_predictions(args.base_csv)
    candidate = read_predictions(args.candidate_csv)
    base_margins = read_margins(args.base_topk_csv)
    candidate_margins = read_margins(args.candidate_topk_csv)
    labels = read_labels(args.label_csv) if args.label_csv else read_labels(args.base_topk_csv)
    image_ids = [row["image_id"] for row in read_rows(args.base_csv)]
    missing = [image_id for image_id in image_ids if image_id not in candidate]
    if missing:
        raise RuntimeError(f"{len(missing)} image_ids missing from candidate; first={missing[:5]}")

    rows = []
    best = None
    best_final: dict[str, str] | None = None
    for base_margin_max, candidate_margin_min in itertools.product(
        parse_grid(args.base_margin_max_grid),
        parse_grid(args.candidate_margin_min_grid),
    ):
        final = {}
        triggered = 0
        for image_id in image_ids:
            use_candidate = (
                base[image_id] != candidate[image_id]
                and base_margins.get(image_id, 0.0) <= base_margin_max
                and candidate_margins.get(image_id, 0.0) >= candidate_margin_min
            )
            final[image_id] = candidate[image_id] if use_candidate else base[image_id]
            triggered += int(use_candidate)
        row = {
            "base_margin_max": base_margin_max,
            "candidate_margin_min": candidate_margin_min,
            "triggered": triggered,
            **evaluate(image_ids, labels, base, final),
        }
        rows.append(row)
        key = (
            row.get("new_top1", 0),
            row.get("net_wins", 0),
            -row.get("losses", 0),
            -row.get("changed", triggered),
        )
        if best is None or key > best[0]:
            best = (key, row)
            best_final = final

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best_final is not None:
        write_predictions(args.out_dir / "best_predictions.csv", image_ids, best_final, labels)
    summary = {
        "base_csv": str(args.base_csv),
        "candidate_csv": str(args.candidate_csv),
        "base_topk_csv": str(args.base_topk_csv),
        "candidate_topk_csv": str(args.candidate_topk_csv),
        "label_csv": str(args.label_csv) if args.label_csv else None,
        "rows": len(image_ids),
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
