from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def letterbox_to_square(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def extract_normalize(preprocess) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    for tr in getattr(preprocess, "transforms", []):
        if isinstance(tr, transforms.Normalize):
            return tuple(float(x) for x in tr.mean), tuple(float(x) for x in tr.std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class ImageRowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path, transform):
        self.rows = rows
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.image_root / row["image_id"]) as image:
            x = self.transform(image.convert("RGB"))
        class_id_text = row.get("class_id", "")
        class_id = int(class_id_text) if class_id_text not in {"", None} else -1
        return x, row["image_id"], row.get("label", ""), class_id


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def shard_paths(shard_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard_{shard_index:05d}"
    return shard_dir / f"{stem}.pt", shard_dir / f"{stem}.summary.json"


def valid_completed_shard(summary_path: Path, expected_rows: int, args_hash: dict[str, Any]) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("rows") == expected_rows and summary.get("args_hash") == args_hash


def encode_rows(
    *,
    rows: list[dict[str, str]],
    shard_index: int,
    shard_dir: Path,
    image_root: Path,
    transform,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    args_hash: dict[str, Any],
) -> dict[str, Any]:
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    ds = ImageRowsDataset(rows, image_root, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features: list[torch.Tensor] = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids: list[int] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(loader, desc=f"openclip_shard_{shard_index:05d}", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model.encode_image(x, normalize=True)
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
    preprocess_mode: str,
    preprocess_repr: str,
) -> dict[str, Any]:
    features: list[torch.Tensor] = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids: list[torch.Tensor] = []
    for shard_index in range(shard_count):
        shard_path, _summary_path = shard_paths(shard_dir, shard_index)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        features.append(payload["features"].float())
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
        class_ids.append(payload["class_ids"].long())
    output = {
        "features": torch.cat(features, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.cat(class_ids, dim=0),
        "classes": classes,
        "model": model_name,
        "manifest": str(manifest),
        "image_root": str(image_root),
        "image_size": image_size,
        "preprocess_mode": preprocess_mode,
        "preprocess": preprocess_repr,
        "shard_dir": str(shard_dir),
        "env": environment_report(),
        "gpu": gpu_snapshot(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    return {
        "out": str(out),
        "rows": len(image_ids),
        "dim": int(output["features"].shape[1]),
        "shards": shard_count,
        "model": model_name,
        "image_size": image_size,
        "preprocess_mode": preprocess_mode,
        "gpu": gpu_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--model", default="hf-hub:timm/PE-Core-L-14-336")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocess-mode", choices=["model", "letterbox"], default="model")
    parser.add_argument("--shard-size", type=int, default=1200)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import open_clip

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")
    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model)
    model = model.to(device).eval()
    if args.preprocess_mode == "model":
        transform = preprocess_val
        preprocess_repr = repr(preprocess_val)
    else:
        mean, std = extract_normalize(preprocess_val)
        fill = tuple(round(float(value) * 255) for value in mean)
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: letterbox_to_square(image, fill)),
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        preprocess_repr = repr(transform)
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "model": args.model,
        "image_size": args.image_size,
        "preprocess_mode": args.preprocess_mode,
        "max_samples": args.max_samples,
    }
    shard_count = math.ceil(len(rows) / args.shard_size)
    summaries = []
    amp = (not args.no_amp) and device.type == "cuda"
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
                amp=amp,
                args_hash=args_hash,
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
        preprocess_mode=args.preprocess_mode,
        preprocess_repr=preprocess_repr,
    )
    summary["shard_summaries"] = summaries
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
