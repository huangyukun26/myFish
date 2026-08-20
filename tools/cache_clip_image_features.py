from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_pil_variants(image: Image.Image, mode: str) -> List[Image.Image]:
    if mode == "none":
        return [image.copy()]
    if mode == "hflip":
        return [image.copy(), ImageOps.mirror(image)]
    if mode in {"fivecrop", "tencrop"}:
        width, height = image.size
        crop_size = min(width, height)
        boxes = [
            (0, 0, crop_size, crop_size),
            (width - crop_size, 0, width, crop_size),
            (0, height - crop_size, crop_size, height),
            (width - crop_size, height - crop_size, width, height),
            ((width - crop_size) // 2, (height - crop_size) // 2, (width + crop_size) // 2, (height + crop_size) // 2),
        ]
        crops = [image.crop(box).copy() for box in boxes]
        if mode == "fivecrop":
            return crops
        return crops + [ImageOps.mirror(crop) for crop in crops]
    raise ValueError(f"Unknown tta mode: {mode}")


class ImageFeatureDataset(Dataset):
    def __init__(self, rows: List[dict], image_root: Path, preprocess, tta_crops: str):
        self.rows = rows
        self.image_root = image_root
        self.preprocess = preprocess
        self.tta_crops = tta_crops

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        label = row.get("label", "")
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            x = torch.stack([self.preprocess(variant) for variant in variants])
        return x, image_id, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="local-dir:work/hf_models/bioclip-2.5-vith14")
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--clip-precision", default="fp16")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import open_clip

    rows = read_manifest(args.manifest)
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")

    pretrained = None if str(args.pretrained).lower() in {"none", "null", ""} else args.pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        precision=args.clip_precision,
        device=device,
    )
    model = model.eval()
    amp = (not args.no_amp) and device.type == "cuda"
    ds = ImageFeatureDataset(rows, args.image_root, preprocess_val, args.tta_crops)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    all_features = []
    image_ids: List[str] = []
    labels: List[str] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels in tqdm(loader, desc="encode_images"):
            current_batch = len(batch_image_ids)
            crop_count = x.shape[1]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                features = model.encode_image(x)
            features = normalize_features(features.float()).view(current_batch, crop_count, -1)
            features = normalize_features(features.mean(dim=1)).cpu()
            all_features.append(features)
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)

    output = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "image_ids": image_ids,
        "labels": labels,
        "features": torch.cat(all_features, dim=0),
        "model": args.model,
        "pretrained": args.pretrained,
        "clip_precision": args.clip_precision,
        "tta_crops": args.tta_crops,
        "amp": amp,
        "env": environment_report(),
        "gpu": gpu_snapshot(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(image_ids),
                "dim": int(output["features"].shape[1]),
                "tta_crops": args.tta_crops,
                "device": str(device),
                "gpu": gpu_snapshot(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
