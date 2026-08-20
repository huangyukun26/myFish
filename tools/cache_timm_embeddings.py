from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def make_pil_variants(image: Image.Image, mode: str) -> List[Image.Image]:
    if mode == "none":
        return [image.copy()]
    if mode == "hflip":
        return [image.copy(), ImageOps.mirror(image)]
    raise ValueError(f"Unsupported tta_crops for timm embeddings: {mode}")


class ImageEmbeddingDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        transform,
        tta_crops: str,
        max_samples: int = 0,
    ):
        self.image_root = image_root
        self.transform = transform
        self.tta_crops = tta_crops
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError(f"No rows in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.image_root / row["image_id"]) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            x = torch.stack([self.transform(variant) for variant in variants])
        class_id_text = row.get("class_id", "")
        class_id = int(class_id_text) if class_id_text not in {"", None} else -1
        return x, row["image_id"], row.get("label", ""), class_id


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip"], default="none")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import timm
    from timm.data import resolve_model_data_config

    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model,
        pretrained=not args.no_pretrained,
        num_classes=0,
        img_size=args.image_size,
    )
    model = model.to(device).eval()
    data_config = resolve_model_data_config(model)
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    transform = transforms.Compose(
        [
            transforms.Resize(int(args.image_size * 1.15)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    ds = ImageEmbeddingDataset(args.manifest, args.image_root, transform, args.tta_crops, args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    amp = (not args.no_amp) and device.type == "cuda"
    features = []
    image_ids: List[str] = []
    labels: List[str] = []
    class_ids: List[int] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(loader, desc="timm_embed_images"):
            batch_size, crop_count = x.shape[:2]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model(x).float().view(batch_size, crop_count, -1).mean(dim=1)
            features.append(normalize_features(encoded).cpu())
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)
            class_ids.extend(int(value) for value in batch_class_ids.tolist())

    feature_tensor = torch.cat(features, dim=0)
    payload = {
        "features": feature_tensor,
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "classes": classes,
        "model": args.model,
        "manifest": str(args.manifest),
        "image_size": args.image_size,
        "tta_crops": args.tta_crops,
        "data_config": data_config,
        "env": environment_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    summary = {
        "out": str(args.out),
        "rows": len(image_ids),
        "dim": int(feature_tensor.shape[1]),
        "model": args.model,
        "image_size": args.image_size,
        "tta_crops": args.tta_crops,
        "gpu": gpu_snapshot(),
    }
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
