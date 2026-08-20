"""Build a leakage-audited pseudo-unseen manifest over the full class universe.

Only frozen official image features and the official 17,393 class text universe
are used.  No image prototype, classifier head, or adaptation is fitted for a
held-out class.  Species folds hold out classes; genus folds hold out all
classes belonging to a set of genera.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stable_int(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_complement_train.pt"))
    parser.add_argument("--val-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_val.pt"))
    parser.add_argument("--text-cache", type=Path, default=Path("work/clip_text_features/all_bioclip25_taxon.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/full_universe"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load(args.train_feature)
    val = load(args.val_feature)
    text = load(args.text_cache)
    full_classes = [str(x) for x in text["classes"]]
    full_to_id = {name: i for i, name in enumerate(full_classes)}
    rows: list[dict[str, Any]] = []
    for source_name, payload in (("train", train), ("val", val)):
        labels = [str(x) for x in payload["labels"]]
        for local_i, (image_id, label) in enumerate(zip(payload["image_ids"], labels)):
            if label not in full_to_id:
                raise RuntimeError(f"label absent from official full universe: {label}")
            genus = label.split()[0] if label else ""
            class_id = full_to_id[label]
            # Hash assignment is fixed before evaluation and independent of
            # predictions; the seed is recorded in the manifest.
            species_fold = (stable_int(f"species:{args.seed}:{label}") % args.folds)
            genus_fold = (stable_int(f"genus:{args.seed}:{genus}") % args.folds)
            rows.append({
                "row_id": len(rows),
                "source": source_name,
                "source_row": local_i,
                "image_id": Path(str(image_id)).name,
                "label": label,
                "class_id": class_id,
                "genus": genus,
                "species_fold": species_fold,
                "genus_fold": genus_fold,
            })

    fields = list(rows[0])
    with (args.out_dir / "query_manifest.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    source_counts = Counter(r["source"] for r in rows)
    class_counts = Counter(r["label"] for r in rows)
    genera = sorted({r["genus"] for r in rows})
    summary = {
        "candidate_classes": len(full_classes),
        "candidate_classes_path": str(args.text_cache),
        "query_rows": len(rows),
        "source_counts": dict(source_counts),
        "query_classes": len(class_counts),
        "query_genera": len(genera),
        "folds": args.folds,
        "seed": args.seed,
        "species_fold_rows": {str(f): sum(r["species_fold"] == f for r in rows) for f in range(args.folds)},
        "genus_fold_rows": {str(f): sum(r["genus_fold"] == f for r in rows) for f in range(args.folds)},
        "species_fold_classes": {str(f): len({r["label"] for r in rows if r["species_fold"] == f}) for f in range(args.folds)},
        "genus_fold_genera": {str(f): len({r["genus"] for r in rows if r["genus_fold"] == f}) for f in range(args.folds)},
        "protocol": {
            "no_heldout_image_prototypes": True,
            "no_classifier_head_or_adaptation": True,
            "candidate_pool_is_full_official_classes": True,
            "query_features_are_frozen_bioclip25": True,
        },
    }
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
