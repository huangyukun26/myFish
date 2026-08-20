from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


SAFE_LABELS = {
    "Building",
    "House",
    "Room",
    "Office building",
    "Road",
    "Street",
    "Wall",
    "Floor",
    "Furniture",
    "Chair",
    "Table",
    "Desk",
    "Shelf",
    "Kitchen",
    "Bedroom",
    "Living room",
    "Vehicle",
    "Car",
    "Train",
    "Airplane",
    "Tower",
    "City",
    "Skyscraper",
    "Door",
    "Window",
    "Stairs",
    "Sidewalk",
    "Cabinetry",
    "Countertop",
}

BLOCKED_LABELS = {
    "Animal",
    "Fish",
    "Aquarium",
    "Fishing",
    "Lake",
    "Ocean",
    "River",
    "Sea",
    "Seafood",
    "Underwater",
    "Water",
    "Person",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cattle",
    "Elephant",
    "Bear",
    "Zebra",
    "Giraffe",
    "Insect",
    "Reptile",
    "Amphibian",
    "Marine mammal",
}


def read_name_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    mid_to_name: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            mid_to_name[row["LabelName"]] = row["DisplayName"]
    return mid_to_name, {name: mid for mid, name in mid_to_name.items()}


def stable_key(image_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    mid_to_name, name_to_mid = read_name_maps(args.class_descriptions)
    missing_safe = sorted(SAFE_LABELS - set(name_to_mid))
    missing_blocked = sorted(BLOCKED_LABELS - set(name_to_mid))
    if missing_safe or missing_blocked:
        raise RuntimeError(
            f"Missing Open Images names; safe={missing_safe}, blocked={missing_blocked}"
        )

    safe_mids = {name_to_mid[name] for name in SAFE_LABELS}
    blocked_mids = {name_to_mid[name] for name in BLOCKED_LABELS}
    positive_safe: dict[str, set[str]] = defaultdict(set)
    positive_blocked: set[str] = set()
    with args.human_labels.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if float(row["Confidence"]) != 1.0:
                continue
            image_id = row["ImageID"]
            mid = row["LabelName"]
            if mid in safe_mids:
                positive_safe[image_id].add(mid_to_name[mid])
            if mid in blocked_mids:
                positive_blocked.add(image_id)

    candidates = set(positive_safe) - positive_blocked
    metadata: dict[str, dict[str, str]] = {}
    with args.image_metadata.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            image_id = row["ImageID"]
            if image_id not in candidates:
                continue
            if row.get("Subset") != "validation":
                continue
            license_url = row.get("License", "")
            if "creativecommons.org/licenses/by/2.0" not in license_url:
                continue
            metadata[image_id] = row

    ordered = sorted(metadata, key=lambda image_id: stable_key(image_id, args.seed))
    selected = ordered[: args.limit]
    if len(selected) < args.limit:
        raise RuntimeError(f"Only {len(selected)} eligible rows for requested {args.limit}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "split",
        "safe_positive_labels",
        "license",
        "author",
        "author_profile_url",
        "original_landing_url",
        "original_url",
        "original_md5",
        "original_size",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in selected:
            row = metadata[image_id]
            writer.writerow(
                {
                    "image_id": image_id,
                    "split": "validation",
                    "safe_positive_labels": "|".join(sorted(positive_safe[image_id])),
                    "license": row.get("License", ""),
                    "author": row.get("Author", ""),
                    "author_profile_url": row.get("AuthorProfileURL", ""),
                    "original_landing_url": row.get("OriginalLandingURL", ""),
                    "original_url": row.get("OriginalURL", ""),
                    "original_md5": row.get("OriginalMD5", ""),
                    "original_size": row.get("OriginalSize", ""),
                }
            )

    args.id_list.parent.mkdir(parents=True, exist_ok=True)
    args.id_list.write_text(
        "".join(f"validation/{image_id}\n" for image_id in selected),
        encoding="utf-8",
    )
    print(
        {
            "eligible_before_license": len(candidates),
            "eligible_cc_by_2": len(metadata),
            "selected": len(selected),
            "seed": args.seed,
            "manifest": str(args.out),
            "id_list": str(args.id_list),
        }
    )


if __name__ == "__main__":
    main()
