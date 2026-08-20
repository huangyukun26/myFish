from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot
from train_supervised import build_model

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
    raise ValueError(f"Unsupported tta_crops for embeddings: {mode}")


class EmbeddingDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, image_size: int, tta_crops: str, max_samples: int = 0):
        self.image_root = image_root
        self.tta_crops = tta_crops
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples:
            self.rows = self.rows[:max_samples]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

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
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def feature_model_from_checkpoint(checkpoint: dict, class_count: int) -> nn.Module:
    args = checkpoint.get("args", {})
    model_name = args.get("model", "resnet50")
    model = build_model(model_name, int(args.get("num_classes", class_count)), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    if model_name in {"resnet18", "resnet50"}:
        return nn.Sequential(*(list(model.children())[:-1]), nn.Flatten())
    if model_name in {
        "convnext_tiny",
        "convnext_small",
        "efficientnet_b0",
        "efficientnet_v2_s",
        "mobilenet_v3_small",
    }:
        return nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    raise ValueError(f"Prototype feature extraction does not support {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prototypes-out", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip"], default="none")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = feature_model_from_checkpoint(checkpoint, len(classes)).to(device).eval()
    ds = EmbeddingDataset(args.manifest, args.image_root, args.image_size, args.tta_crops, args.max_samples)
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
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(loader, desc="embed_images"):
            batch_size, crop_count = x.shape[:2]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model(x).float().view(batch_size, crop_count, -1).mean(dim=1)
            encoded = normalize_features(encoded)
            features.append(encoded.cpu())
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
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "tta_crops": args.tta_crops,
        "env": environment_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    summary = {"out": str(args.out), "rows": len(image_ids), "dim": int(feature_tensor.shape[1])}
    if args.prototypes_out:
        valid = payload["class_ids"] >= 0
        if not bool(valid.any()):
            raise RuntimeError("Cannot build prototypes: manifest has no class_id column")
        sums = torch.zeros((len(classes), feature_tensor.shape[1]), dtype=torch.float32)
        counts = torch.zeros(len(classes), dtype=torch.long)
        for feature, class_id in zip(feature_tensor[valid], payload["class_ids"][valid]):
            sums[int(class_id.item())] += feature
            counts[int(class_id.item())] += 1
        prototypes = normalize_features(sums / counts[:, None].clamp_min(1).float())
        proto_payload = {
            "prototypes": prototypes,
            "counts": counts,
            "classes": classes,
            "source_features": str(args.out),
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
        }
        args.prototypes_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(proto_payload, args.prototypes_out)
        summary["prototypes_out"] = str(args.prototypes_out)
        summary["prototype_classes_with_samples"] = int((counts > 0).sum().item())

    summary["gpu"] = gpu_snapshot()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
