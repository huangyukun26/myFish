from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_boxes(path: Path):
    boxes = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            boxes[row["image_id"]] = {
                "crop_box": row["crop_box"],
                "fallback": bool(row.get("fallback", False)),
                "crop_area_fraction": float(row.get("crop_area_fraction", 1.0)),
            }
    return boxes


class RoiDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        boxes_path: Path,
        image_size: int,
        expand: float,
        max_samples: int,
    ):
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.image_root = image_root
        self.boxes = load_boxes(boxes_path)
        self.expand = expand
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_id = row["image_id"]
        box_row = self.boxes.get(image_id)
        with Image.open(self.image_root / image_id) as source:
            image = source.convert("RGB")
            width, height = image.size
            if box_row is None:
                crop_box = (0, 0, width, height)
                fallback = True
                area = 1.0
            else:
                x0, y0, x1, y1 = [float(value) for value in box_row["crop_box"]]
                pad_x = (x1 - x0) * self.expand
                pad_y = (y1 - y0) * self.expand
                x0 = max(0, int(math.floor(x0 - pad_x)))
                y0 = max(0, int(math.floor(y0 - pad_y)))
                x1 = min(width, int(math.ceil(x1 + pad_x)))
                y1 = min(height, int(math.ceil(y1 + pad_y)))
                if x1 <= x0 or y1 <= y0:
                    crop_box = (0, 0, width, height)
                    fallback = True
                    area = 1.0
                else:
                    crop_box = (x0, y0, x1, y1)
                    fallback = box_row["fallback"]
                    area = box_row["crop_area_fraction"]
            tensor = self.transform(image.crop(crop_box))
        class_text = row.get("class_id", "")
        class_id = int(class_text) if class_text not in {"", None} else -1
        return (
            tensor,
            image_id,
            row.get("label", ""),
            class_id,
            bool(fallback),
            float(area),
        )


def build_model(checkpoint):
    args = checkpoint.get("args", {})
    model_name = args.get("model", "convnext_small")
    if model_name != "convnext_small":
        raise ValueError("This scout expects a convnext_small checkpoint, got %r" % model_name)
    num_classes = int(args.get("num_classes", 5795))
    model = models.convnext_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features,
        num_classes,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def normalize(value):
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    model = build_model(checkpoint).to(device).eval()
    dataset = RoiDataset(
        args.manifest,
        args.image_root,
        args.boxes,
        args.image_size,
        args.expand,
        args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    use_amp = (not args.no_amp) and device.type == "cuda"
    globals_out = []
    parts_out = []
    mid_parts_out = []
    image_ids = []
    labels = []
    class_ids = []
    fallbacks = []
    crop_areas = []
    started = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            images, ids, batch_labels, batch_classes, batch_fallbacks, batch_areas = batch
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                mid_feature_map = model.features[:6](images)
                feature_map = model.features[6:](mid_feature_map)
                mid_feature_map = model.features[6][0](mid_feature_map)
                feature_map = model.classifier[0](feature_map)
                pooled_mid_parts = F.adaptive_avg_pool2d(mid_feature_map, (2, 3))
                pooled_mid_parts = pooled_mid_parts.flatten(2).transpose(1, 2)
                pooled_parts = F.adaptive_avg_pool2d(feature_map, (2, 3))
                pooled_parts = pooled_parts.flatten(2).transpose(1, 2)
                pooled_global = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            mid_parts_out.append(normalize(pooled_mid_parts).half().cpu())
            parts_out.append(normalize(pooled_parts).half().cpu())
            globals_out.append(normalize(pooled_global).half().cpu())
            image_ids.extend(list(ids))
            labels.extend(list(batch_labels))
            class_ids.extend(int(value) for value in batch_classes.tolist())
            fallbacks.extend(bool(value) for value in batch_fallbacks.tolist())
            crop_areas.extend(float(value) for value in batch_areas.tolist())
            if args.log_every and (step % args.log_every == 0 or step == len(loader)):
                elapsed = time.time() - started
                done = min(step * args.batch_size, len(dataset))
                rate = done / max(elapsed, 1e-6)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "steps": len(loader),
                            "rows": done,
                            "rows_per_sec": round(rate, 2),
                            "elapsed_sec": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )

    payload = {
        "global": torch.cat(globals_out, dim=0),
        "mid_parts": torch.cat(mid_parts_out, dim=0),
        "parts": torch.cat(parts_out, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "fallback": torch.tensor(fallbacks, dtype=torch.bool),
        "crop_area_fraction": torch.tensor(crop_areas, dtype=torch.float32),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "boxes": str(args.boxes),
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "expand": args.expand,
        "pool": "convnext_final_norm_2x3_parts",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out))
    elapsed = time.time() - started
    summary = {
        "out": str(args.out),
        "rows": len(image_ids),
        "global_shape": list(payload["global"].shape),
        "mid_parts_shape": list(payload["mid_parts"].shape),
        "parts_shape": list(payload["parts"].shape),
        "fallback_rows": int(payload["fallback"].sum().item()),
        "elapsed_sec": elapsed,
        "rows_per_sec": len(image_ids) / max(elapsed, 1e-6),
        "device": str(device),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
