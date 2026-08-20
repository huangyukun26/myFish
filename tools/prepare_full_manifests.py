from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("work/full_manifests"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    zip_path = dataset_root / "images.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite to replace manifest files.")
    args.output.mkdir(parents=True, exist_ok=True)

    labels = json.loads((dataset_root / "label_train.json").read_text(encoding="utf-8"))
    all_classes = load_pickle(dataset_root / "all_classes.pkl")
    train_split = load_pickle(dataset_root / "splits" / "train.pkl")
    test_split = load_pickle(dataset_root / "splits" / "test.pkl")
    unseen_split = load_pickle(dataset_root / "splits" / "unseen.pkl")

    try:
        with zipfile.ZipFile(zip_path) as zf:
            image_members = [
                info.filename
                for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{zip_path} is not a complete readable ZIP yet. "
            "Run tools\\verify_images_zip.py first and wait until it reports status=complete_readable."
        ) from exc
    member_by_base = {Path(member).name: member for member in image_members}

    missing_train = [name for name in train_split if name not in member_by_base]
    missing_test = [name for name in test_split if name not in member_by_base]
    missing_unseen = [name for name in unseen_split if name not in member_by_base]
    if missing_train or missing_test or missing_unseen:
        raise RuntimeError(
            "ZIP does not contain all split images. "
            f"missing train={len(missing_train)} test={len(missing_test)} unseen={len(missing_unseen)}"
        )

    seen_classes = sorted(set(labels.values()))
    seen_class_to_idx = {name: idx for idx, name in enumerate(seen_classes)}
    all_class_set = set(all_classes)
    unseen_candidate_classes = sorted(all_class_set - set(seen_classes))

    train_rows = (
        {
            "image_id": image_id,
            "zip_member": member_by_base[image_id],
            "label": labels[image_id],
            "class_id": seen_class_to_idx[labels[image_id]],
            "split": "train",
        }
        for image_id in train_split
    )
    test_rows = (
        {"image_id": image_id, "zip_member": member_by_base[image_id], "label": "", "class_id": "", "split": "test_seen"}
        for image_id in test_split
    )
    unseen_rows = (
        {"image_id": image_id, "zip_member": member_by_base[image_id], "label": "", "class_id": "", "split": "test_unseen"}
        for image_id in unseen_split
    )

    train_count = write_csv(
        args.output / "train.csv",
        train_rows,
        ["image_id", "zip_member", "label", "class_id", "split"],
    )
    test_count = write_csv(
        args.output / "test_seen.csv",
        test_rows,
        ["image_id", "zip_member", "label", "class_id", "split"],
    )
    unseen_count = write_csv(
        args.output / "test_unseen.csv",
        unseen_rows,
        ["image_id", "zip_member", "label", "class_id", "split"],
    )

    submission_rows = (
        {"image_id": image_id, "source_split": split}
        for split, names in [("test_seen", test_split), ("test_unseen", unseen_split)]
        for image_id in names
    )
    submission_count = write_csv(args.output / "submission_keys.csv", submission_rows, ["image_id", "source_split"])

    (args.output / "seen_class_to_idx.json").write_text(
        json.dumps(seen_class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "all_classes.json").write_text(json.dumps(all_classes, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "unseen_candidate_classes.json").write_text(
        json.dumps(unseen_candidate_classes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "zip_path": str(zip_path),
        "image_members": len(image_members),
        "train_rows": train_count,
        "test_seen_rows": test_count,
        "test_unseen_rows": unseen_count,
        "submission_rows": submission_count,
        "seen_classes": len(seen_classes),
        "all_classes": len(all_classes),
        "unseen_candidate_classes": len(unseen_candidate_classes),
        "output": str(args.output),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
