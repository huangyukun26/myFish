from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from fishnet.zip_local import iter_local_entries, read_entry_data


EXPECTED_IMAGES_ZIP_SIZE = 40_845_415_697


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def split_counts_by_basename(names, train_set, test_set, unseen_set) -> Dict[str, int]:
    counts = {"train": 0, "test_seen": 0, "test_unseen": 0, "other": 0}
    for name in names:
        base = Path(name).name
        if not base:
            continue
        if base in train_set:
            counts["train"] += 1
        elif base in test_set:
            counts["test_seen"] += 1
        elif base in unseen_set:
            counts["test_unseen"] += 1
        else:
            counts["other"] += 1
    return counts


def verify_full_zip(zip_path: Path, train_set, test_set, unseen_set, sample_images: int) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        image_names = [name for name in names if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        by_base = {Path(name).name: name for name in image_names}
        expected = train_set | test_set | unseen_set
        missing = sorted(expected - set(by_base))
        extra = sorted(set(by_base) - expected)

        corrupt = []
        for base in sorted(expected & set(by_base))[:sample_images]:
            try:
                with zf.open(by_base[base]) as fp:
                    with Image.open(fp) as image:
                        image.verify()
            except Exception as exc:
                corrupt.append({"image": base, "error": repr(exc)})

    return {
        "central_directory_ok": True,
        "file_entries": len(infos),
        "image_entries": len(image_names),
        "split_counts": split_counts_by_basename(image_names, train_set, test_set, unseen_set),
        "missing_expected_images": len(missing),
        "missing_examples": missing[:20],
        "extra_images": len(extra),
        "extra_examples": extra[:20],
        "sample_checked_images": min(sample_images, len(expected & set(by_base))),
        "sample_corrupt_images": corrupt,
    }


def verify_partial_zip(zip_path: Path, train_set, test_set, unseen_set, sample_images: int) -> Dict[str, Any]:
    entries = list(iter_local_entries(zip_path))
    image_entries = [entry for entry in entries if entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    corrupt = []
    checked = 0
    with zip_path.open("rb") as fp:
        for entry in image_entries[:sample_images]:
            try:
                raw = read_entry_data(fp, entry)
                from io import BytesIO

                with Image.open(BytesIO(raw)) as image:
                    image.verify()
                checked += 1
            except Exception as exc:
                corrupt.append({"image": Path(entry.name).name, "error": repr(exc)})
    return {
        "central_directory_ok": False,
        "complete_local_entries": len(entries),
        "complete_local_image_entries": len(image_entries),
        "split_counts_in_complete_local_entries": split_counts_by_basename(
            [entry.name for entry in image_entries], train_set, test_set, unseen_set
        ),
        "sample_checked_images": checked,
        "sample_corrupt_images": corrupt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--expected-size", type=int, default=EXPECTED_IMAGES_ZIP_SIZE)
    parser.add_argument("--sample-images", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    zip_path = dataset_root / "images.zip"
    train_set = set(load_pickle(dataset_root / "splits" / "train.pkl"))
    test_set = set(load_pickle(dataset_root / "splits" / "test.pkl"))
    unseen_set = set(load_pickle(dataset_root / "splits" / "unseen.pkl"))

    result: Dict[str, Any] = {
        "zip_path": str(zip_path),
        "exists": zip_path.exists(),
        "expected_size_bytes": args.expected_size,
        "actual_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "expected_split_images": {
            "train": len(train_set),
            "test_seen": len(test_set),
            "test_unseen": len(unseen_set),
            "total": len(train_set | test_set | unseen_set),
        },
    }

    if not zip_path.exists():
        result["status"] = "missing"
    else:
        result["size_matches_expected"] = result["actual_size_bytes"] == args.expected_size
        result["size_is_at_least_expected"] = result["actual_size_bytes"] >= args.expected_size
        try:
            result.update(verify_full_zip(zip_path, train_set, test_set, unseen_set, args.sample_images))
            result["status"] = "complete_readable" if result["missing_expected_images"] == 0 else "readable_with_missing_images"
        except zipfile.BadZipFile as exc:
            result["zipfile_error"] = repr(exc)
            result.update(verify_partial_zip(zip_path, train_set, test_set, unseen_set, args.sample_images))
            result["status"] = "partial_or_corrupt"

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    if args.require_complete and result.get("status") != "complete_readable":
        sys.exit(2)


if __name__ == "__main__":
    main()

