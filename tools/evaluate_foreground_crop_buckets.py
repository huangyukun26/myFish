from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_prototype_feature_fusion import (
    cosine_logits,
    genus_folds,
    load_pair,
    row_standardize,
)


def comparison(prediction: torch.Tensor, target: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> dict:
    prediction = prediction[mask]
    target = target[mask]
    base = base[mask]
    correct = prediction.eq(target)
    base_correct = base.eq(target)
    return {
        "rows": int(mask.sum()),
        "base_top1": float(base_correct.float().mean()) if len(target) else 0.0,
        "top1": float(correct.float().mean()) if len(target) else 0.0,
        "changed": int(prediction.ne(base).sum()),
        "wins": int((correct & ~base_correct).sum()),
        "losses": int((~correct & base_correct).sum()),
        "net": int(correct.sum() - base_correct.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-query", type=Path, required=True)
    parser.add_argument("--crop-train", type=Path, required=True)
    parser.add_argument("--crop-query", type=Path, required=True)
    parser.add_argument("--crop-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    base_train, base_query = load_pair(args.base_train, args.base_query)
    crop_train, crop_query = load_pair(args.crop_train, args.crop_query)
    if base_query["image_ids"] != crop_query["image_ids"]:
        raise RuntimeError("Base and crop query image order differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = base_query["class_ids"].long().to(device)
    base_scores = cosine_logits(base_train, base_query, device)
    crop_scores = cosine_logits(crop_train, crop_query, device)
    base_prediction = base_scores.argmax(dim=1)
    crop_prediction = crop_scores.argmax(dim=1)
    base_z = row_standardize(base_scores)
    crop_z = row_standardize(crop_scores)

    weights = [round(index / 10, 1) for index in range(11)]
    predictions = [((1.0 - weight) * base_z + weight * crop_z).argmax(dim=1) for weight in weights]
    folds = genus_folds(list(base_query["labels"]), args.folds).to(device)
    oof_prediction = base_prediction.clone()
    selected_weights = []
    for fold in range(args.folds):
        train_mask = folds.ne(fold)
        heldout = ~train_mask
        accuracies = [float(pred[train_mask].eq(target[train_mask]).float().mean()) for pred in predictions]
        index = max(range(len(weights)), key=lambda item: (accuracies[item], -abs(weights[item] - 0.5)))
        selected_weights.append(weights[index])
        oof_prediction[heldout] = predictions[index][heldout]

    metadata_by_id = {}
    with args.crop_metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata_by_id[row["image_id"]] = row
    missing = [image_id for image_id in base_query["image_ids"] if image_id not in metadata_by_id]
    if missing:
        raise RuntimeError(f"Missing crop metadata for {len(missing)} images; first={missing[:5]}")
    metadata = [metadata_by_id[image_id] for image_id in base_query["image_ids"]]
    area = torch.tensor([float(row["crop_area_fraction"]) for row in metadata], device=device)
    aspect = torch.tensor([float(row["aspect_ratio"]) for row in metadata], device=device)
    density = torch.tensor(
        [float(row["component_patches"]) / max(1, int(row["valid_patches"])) for row in metadata],
        device=device,
    )

    buckets = {
        "crop_area_lt025": area.lt(0.25),
        "crop_area_025_050": area.ge(0.25) & area.lt(0.50),
        "crop_area_050_075": area.ge(0.50) & area.lt(0.75),
        "crop_area_ge075": area.ge(0.75),
        "aspect_tall_lt075": aspect.lt(0.75),
        "aspect_normal_075_150": aspect.ge(0.75) & aspect.lt(1.50),
        "aspect_wide_150_225": aspect.ge(1.50) & aspect.lt(2.25),
        "aspect_extreme_ge225": aspect.ge(2.25),
        "component_density_lt005": density.lt(0.05),
        "component_density_005_015": density.ge(0.05) & density.lt(0.15),
        "component_density_ge015": density.ge(0.15),
    }
    branches = {"crop": crop_prediction, "fusion_oof": oof_prediction}
    output = {
        "rows": len(metadata),
        "selected_weights": selected_weights,
        "overall": {
            name: comparison(prediction, target, base_prediction, torch.ones_like(target, dtype=torch.bool))
            for name, prediction in branches.items()
        },
        "buckets": {
            bucket: {
                name: comparison(prediction, target, base_prediction, mask)
                for name, prediction in branches.items()
            }
            for bucket, mask in buckets.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
