from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from functools import partial
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


def letterbox_to_square(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
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
    raise ValueError(f"Unsupported tta_crops for timm embeddings: {mode}")


class ImageRowsDataset(Dataset):
    def __init__(self, rows: list[dict], image_root: Path, transform, tta_crops: str):
        self.rows = rows
        self.image_root = image_root
        self.transform = transform
        self.tta_crops = tta_crops

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
    transform,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_crops: str,
    amp: bool,
    args_hash: dict,
    feature_pool: str,
) -> dict:
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    ds = ImageRowsDataset(rows, image_root, transform, tta_crops)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids: list[int] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(loader, desc=f"timm_shard_{shard_index:05d}", leave=False):
            current_batch, crop_count = x.shape[:2]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                if feature_pool == "model":
                    encoded = model(x)
                elif feature_pool == "cls":
                    tokens = model.forward_features(x)
                    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                        raise RuntimeError("cls requires a ViT forward_features tensor shaped [B, tokens, dim]")
                    encoded = tokens[:, 0]
                elif feature_pool == "cls_patch_mean":
                    tokens = model.forward_features(x)
                    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                        raise RuntimeError(
                            "cls_patch_mean requires a ViT forward_features tensor shaped [B, tokens, dim]"
                        )
                    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
                    if tokens.shape[1] <= prefix_tokens:
                        raise RuntimeError("Model output has no patch tokens")
                    encoded = torch.cat(
                        [tokens[:, 0], tokens[:, prefix_tokens:].mean(dim=1)],
                        dim=1,
                    )
                else:
                    raise ValueError(f"Unsupported feature_pool: {feature_pool}")
                encoded = encoded.float().view(current_batch, crop_count, -1).mean(dim=1)
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
    manifest: Path,
    image_root: Path,
    model_name: str,
    image_size: int,
    tta_crops: str,
    data_config: dict,
    preprocess_mode: str,
    feature_pool: str,
    preserve_head: bool,
) -> dict:
    features = []
    image_ids: list[str] = []
    labels: list[str] = []
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
        "model": model_name,
        "manifest": str(manifest),
        "image_root": str(image_root),
        "image_size": image_size,
        "tta_crops": tta_crops,
        "preprocess_mode": preprocess_mode,
        "feature_pool": feature_pool,
        "preserve_head": preserve_head,
        "data_config": data_config,
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
        "model": model_name,
        "image_size": image_size,
        "tta_crops": tta_crops,
        "preprocess_mode": preprocess_mode,
        "feature_pool": feature_pool,
        "preserve_head": preserve_head,
        "gpu": gpu_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument(
        "--preprocess-mode",
        choices=["model", "official", "letterbox"],
        default="model",
        help=(
            "'official' uses timm's resolved pretrained transform exactly; "
            "'model' retains the legacy 1.15x resize + center crop behavior."
        ),
    )
    parser.add_argument("--feature-pool", choices=["model", "cls", "cls_patch_mean"], default="model")
    parser.add_argument(
        "--preserve-head",
        action="store_true",
        help="Keep the pretrained model head/projection (required for official CLIP image embeddings).",
    )
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pretrained-file", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import timm
    from timm.data import create_transform, resolve_model_data_config

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_kwargs = {
        "pretrained": not args.no_pretrained,
        "img_size": args.image_size,
    }
    if not args.preserve_head:
        create_kwargs["num_classes"] = 0
    if args.pretrained_file is not None:
        create_kwargs["pretrained_cfg_overlay"] = {"file": str(args.pretrained_file)}
    model = timm.create_model(args.model, **create_kwargs).to(device).eval()
    data_config = resolve_model_data_config(model)
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    if args.preprocess_mode == "official":
        transform = create_transform(**data_config, is_training=False)
    elif args.preprocess_mode == "model":
        geometry = [
            transforms.Resize(int(args.image_size * 1.15)),
            transforms.CenterCrop(args.image_size),
        ]
        transform = transforms.Compose(
            [
                *geometry,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        fill = tuple(round(float(value) * 255) for value in mean)
        geometry = [
            transforms.Lambda(partial(letterbox_to_square, fill=fill)),
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform = transforms.Compose(
            [
                *geometry,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    amp = (not args.no_amp) and device.type == "cuda"
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "model": args.model,
        "image_size": args.image_size,
        "tta_crops": args.tta_crops,
        "preprocess_mode": args.preprocess_mode,
        "feature_pool": args.feature_pool,
        "preserve_head": args.preserve_head,
        "max_samples": args.max_samples,
        "no_pretrained": args.no_pretrained,
        "pretrained_file": str(args.pretrained_file) if args.pretrained_file else None,
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
                transform=transform,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tta_crops=args.tta_crops,
                amp=amp,
                args_hash=args_hash,
                feature_pool=args.feature_pool,
            )
        )
    summary = merge_shards(
        shard_dir=args.shard_dir,
        shard_count=shard_count,
        out=args.out,
        classes=classes,
        manifest=args.manifest,
        image_root=args.image_root,
        model_name=args.model,
        image_size=args.image_size,
        tta_crops=args.tta_crops,
        data_config=data_config,
        preprocess_mode=args.preprocess_mode,
        feature_pool=args.feature_pool,
        preserve_head=args.preserve_head,
    )
    summary["shard_summaries"] = summaries
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
