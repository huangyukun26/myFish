from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


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


class EmbeddingRowsDataset(Dataset):
    def __init__(self, rows: list[dict], image_root: Path, image_size: int, tta_crops: str):
        self.rows = rows
        self.image_root = image_root
        self.tta_crops = tta_crops
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


def shard_paths(shard_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard_{shard_index:05d}"
    return shard_dir / f"{stem}.pt", shard_dir / f"{stem}.summary.json"


def valid_completed_shard(summary_path: Path, expected_rows: int, args_hash: dict) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("rows") == expected_rows and summary.get("args_hash") == args_hash


def encode_rows(
    *,
    rows: list[dict],
    shard_index: int,
    shard_dir: Path,
    image_root: Path,
    image_size: int,
    tta_crops: str,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    args_hash: dict,
) -> dict:
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    ds = EmbeddingRowsDataset(rows, image_root, image_size, tta_crops)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = []
    image_ids: List[str] = []
    labels: List[str] = []
    class_ids: List[int] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(loader, desc=f"supervised_shard_{shard_index:05d}", leave=False):
            current_batch, crop_count = x.shape[:2]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model(x).float().view(current_batch, crop_count, -1).mean(dim=1)
            features.append(normalize_features(encoded).cpu())
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)
            class_ids.extend(int(value) for value in batch_class_ids.tolist())
    payload = {
        "features": torch.cat(features, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "args_hash": args_hash,
    }
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, shard_path)
    summary = {
        "shard_index": shard_index,
        "rows": len(image_ids),
        "dim": int(payload["features"].shape[1]),
        "shard": str(shard_path),
        "args_hash": args_hash,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def merge_shards(
    *,
    shard_dir: Path,
    shard_count: int,
    out: Path,
    classes: list[str],
    checkpoint: Path,
    manifest: Path,
    image_root: Path,
    image_size: int,
    tta_crops: str,
) -> dict:
    features = []
    image_ids: List[str] = []
    labels: List[str] = []
    class_ids = []
    for shard_index in range(shard_count):
        shard_path, _summary_path = shard_paths(shard_dir, shard_index)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        features.append(payload["features"].float())
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
        class_ids.append(payload["class_ids"].long())
    feature_tensor = torch.cat(features, dim=0)
    output = {
        "features": feature_tensor,
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.cat(class_ids, dim=0),
        "classes": classes,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "image_root": str(image_root),
        "image_size": image_size,
        "tta_crops": tta_crops,
        "shard_dir": str(shard_dir),
        "env": environment_report(),
        "gpu": gpu_snapshot(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    return {
        "out": str(out),
        "rows": len(image_ids),
        "dim": int(feature_tensor.shape[1]),
        "shards": shard_count,
        "checkpoint": str(checkpoint),
        "image_size": image_size,
        "tta_crops": tta_crops,
        "gpu": gpu_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip"], default="none")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = feature_model_from_checkpoint(checkpoint, len(classes)).to(device).eval()
    amp = (not args.no_amp) and device.type == "cuda"
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "tta_crops": args.tta_crops,
        "max_samples": args.max_samples,
    }

    shard_count = math.ceil(len(rows) / args.shard_size)
    summaries = []
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        _shard_path, summary_path = shard_paths(args.shard_dir, shard_index)
        if args.resume and valid_completed_shard(summary_path, len(shard_rows), args_hash):
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        summaries.append(
            encode_rows(
                rows=shard_rows,
                shard_index=shard_index,
                shard_dir=args.shard_dir,
                image_root=args.image_root,
                image_size=args.image_size,
                tta_crops=args.tta_crops,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp=amp,
                args_hash=args_hash,
            )
        )

    summary = merge_shards(
        shard_dir=args.shard_dir,
        shard_count=shard_count,
        out=args.out,
        classes=classes,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        image_root=args.image_root,
        image_size=args.image_size,
        tta_crops=args.tta_crops,
    )
    summary["shard_summaries"] = summaries
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
