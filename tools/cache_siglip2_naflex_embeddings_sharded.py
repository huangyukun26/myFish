from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> list[dict[str, str]]:
    # utf-8-sig also accepts plain UTF-8 and strips the BOM emitted by Excel/PowerShell.
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


class ImagePathDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path) -> None:
        self.rows = rows
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        return {
            "path": str(self.image_root / row["image_id"]),
            "image_id": row["image_id"],
            "label": row.get("label", ""),
            "class_id": row.get("class_id", ""),
        }


def collate_images(
    rows: list[dict[str, str]],
    *,
    processor: Any,
    max_num_patches: int,
    hflip: bool,
) -> dict[str, Any]:
    images = []
    for row in rows:
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            images.append(image.copy())
            if hflip:
                images.append(ImageOps.mirror(image))
    processor_kwargs = {"images": images, "return_tensors": "pt"}
    if max_num_patches > 0:
        processor_kwargs["max_num_patches"] = max_num_patches
    inputs = processor(**processor_kwargs)
    return {"inputs": inputs, "rows": rows}


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


def encode_shard(
    *,
    rows: list[dict[str, str]],
    shard_index: int,
    shard_dir: Path,
    image_root: Path,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_num_patches: int,
    hflip: bool,
    args_hash: dict,
) -> dict:
    dataset = ImagePathDataset(rows, image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=partial(
            collate_images,
            processor=processor,
            max_num_patches=max_num_patches,
            hflip=hflip,
        ),
    )
    features = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids: list[int] = []
    crop_count = 2 if hflip else 1
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"siglip2_shard_{shard_index:05d}", leave=False):
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["inputs"].items()
                if isinstance(value, torch.Tensor)
            }
            encoded = model.get_image_features(**inputs)
            encoded = encoded.float().view(len(batch["rows"]), crop_count, -1).mean(dim=1)
            features.append(F.normalize(encoded, dim=1).cpu())
            for row in batch["rows"]:
                image_ids.append(row["image_id"])
                labels.append(row["label"])
                class_id = row["class_id"]
                class_ids.append(int(class_id) if class_id not in {"", None} else -1)

    payload = {
        "features": torch.cat(features, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "args_hash": args_hash,
    }
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, shard_path)
    summary = {
        "shard_index": shard_index,
        "rows": len(image_ids),
        "dim": int(payload["features"].shape[1]),
        "shard": str(shard_path),
        "args_hash": args_hash,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def merge_shards(
    *,
    shard_dir: Path,
    shard_count: int,
    out: Path,
    classes: list[str],
    args: argparse.Namespace,
) -> dict:
    feature_chunks = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids = []
    for shard_index in range(shard_count):
        shard_path, _summary_path = shard_paths(shard_dir, shard_index)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        feature_chunks.append(payload["features"].float())
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
        class_ids.append(payload["class_ids"].long())
    features = torch.cat(feature_chunks, dim=0)
    output = {
        "features": features,
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.cat(class_ids, dim=0),
        "classes": classes,
        "model": args.model,
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "max_num_patches": args.max_num_patches,
        "hflip": args.hflip,
        "shard_dir": str(shard_dir),
        "env": environment_report(),
        "gpu": gpu_snapshot(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    return {
        "out": str(out),
        "rows": len(image_ids),
        "dim": int(features.shape[1]),
        "shards": shard_count,
        "model": args.model,
        "max_num_patches": args.max_num_patches,
        "hflip": args.hflip,
        "gpu": gpu_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--max-num-patches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")
    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    # FG-CLIP publishes its custom CLIP class through the causal-LM auto map.
    # Standard SigLIP/SigLIP2 checkpoints continue to use AutoModel.
    config_path = Path(args.model) / "config.json"
    use_causal_auto = False
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        use_causal_auto = "AutoModelForCausalLM" in config_data.get("auto_map", {})
    model_auto = AutoModelForCausalLM if use_causal_auto else AutoModel
    model = model_auto.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "model": args.model,
        "max_num_patches": args.max_num_patches,
        "max_samples": args.max_samples,
        "hflip": args.hflip,
    }

    shard_count = math.ceil(len(rows) / args.shard_size)
    shard_summaries = []
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        _shard_path, summary_path = shard_paths(args.shard_dir, shard_index)
        if args.resume and valid_completed_shard(summary_path, len(shard_rows), args_hash):
            shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        shard_summaries.append(
            encode_shard(
                rows=shard_rows,
                shard_index=shard_index,
                shard_dir=args.shard_dir,
                image_root=args.image_root,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_num_patches=args.max_num_patches,
                hflip=args.hflip,
                args_hash=args_hash,
            )
        )
    summary = merge_shards(
        shard_dir=args.shard_dir,
        shard_count=shard_count,
        out=args.out,
        classes=classes,
        args=args,
    )
    summary["shard_summaries"] = shard_summaries
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
