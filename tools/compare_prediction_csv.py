from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return {row["image_id"]: row.get("label", "") for row in csv.DictReader(fp)}


def read_predictions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return {row["image_id"]: row["prediction"] for row in csv.DictReader(fp)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    labels = read_manifest(args.manifest)
    base = read_predictions(args.base)
    candidate = read_predictions(args.candidate)
    image_ids = [image_id for image_id, label in labels.items() if label]
    missing_base = [image_id for image_id in image_ids if image_id not in base]
    missing_candidate = [image_id for image_id in image_ids if image_id not in candidate]
    if missing_base or missing_candidate:
        raise RuntimeError(
            f"missing_base={len(missing_base)} first={missing_base[:5]} "
            f"missing_candidate={len(missing_candidate)} first={missing_candidate[:5]}"
        )

    changed = wins = losses = base_correct = candidate_correct = both_correct = both_wrong = 0
    for image_id in image_ids:
        label = labels[image_id]
        base_ok = base[image_id] == label
        candidate_ok = candidate[image_id] == label
        base_correct += int(base_ok)
        candidate_correct += int(candidate_ok)
        both_correct += int(base_ok and candidate_ok)
        both_wrong += int((not base_ok) and (not candidate_ok))
        changed += int(base[image_id] != candidate[image_id])
        wins += int((not base_ok) and candidate_ok)
        losses += int(base_ok and (not candidate_ok))

    summary = {
        "manifest": str(args.manifest),
        "base": str(args.base),
        "candidate": str(args.candidate),
        "rows": len(image_ids),
        "base_top1": base_correct / len(image_ids),
        "candidate_top1": candidate_correct / len(image_ids),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
