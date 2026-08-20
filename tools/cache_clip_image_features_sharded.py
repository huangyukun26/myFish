from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot
from fishnet.image_preprocess import apply_preprocess_mode

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def parse_class_id(value: str | None) -> int:
    if value is None or value == "":
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def build_classes(labels: list[str], class_ids: list[int]) -> list[str]:
    valid_ids = [class_id for class_id in class_ids if class_id >= 0]
    if not valid_ids:
        return []
    classes = [""] * (max(valid_ids) + 1)
    for label, class_id in zip(labels, class_ids):
        if class_id >= 0 and label and not classes[class_id]:
            classes[class_id] = label
    return classes


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
    def __init__(self, rows: List[dict], image_root: Path, preprocess, tta_crops: str, preprocess_mode: str):
        self.rows = rows
        self.image_root = image_root
        self.preprocess = preprocess
        self.tta_crops = tta_crops
        self.preprocess_mode = preprocess_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        label = row.get("label", "")
        class_id = parse_class_id(row.get("class_id"))
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            variants = [apply_preprocess_mode(variant, self.preprocess_mode) for variant in variants]
            x = torch.stack([self.preprocess(variant) for variant in variants])
        return x, image_id, label, class_id


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
    rows: List[dict],
    shard_index: int,
    shard_dir: Path,
    image_root: Path,
    preprocess,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_crops: str,
    preprocess_mode: str,
    amp: bool,
    args_hash: dict,
) -> dict:
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    ds = ImageFeatureDataset(rows, image_root, preprocess, tta_crops, preprocess_mode)
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
        for x, batch_image_ids, batch_labels, batch_class_ids in tqdm(
            loader,
            desc=f"clip_shard_{shard_index:05d}",
            leave=False,
        ):
            current_batch = len(batch_image_ids)
            crop_count = x.shape[1]
            x = x.to(device, non_blocking=True).flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model.encode_image(x)
            encoded = normalize_features(encoded.float()).view(current_batch, crop_count, -1)
            encoded = normalize_features(encoded.mean(dim=1)).cpu()
            features.append(encoded)
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)
            if torch.is_tensor(batch_class_ids):
                class_ids.extend(int(x) for x in batch_class_ids.tolist())
            else:
                class_ids.extend(int(x) for x in batch_class_ids)

    payload = {
        "features": torch.cat(features, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "classes": build_classes(labels, class_ids),
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
    manifest: Path,
    image_root: Path,
    model_name: str,
    pretrained: str,
    clip_precision: str,
    tta_crops: str,
    preprocess_mode: str,
    amp: bool,
) -> dict:
    all_features = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids: list[int] = []
    for shard_index in range(shard_count):
        shard_path, _summary_path = shard_paths(shard_dir, shard_index)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_features.append(payload["features"].float())
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
        if "class_ids" in payload:
            class_ids.extend(int(x) for x in payload["class_ids"].tolist())
        else:
            class_ids.extend([-1] * len(payload["image_ids"]))
    feature_tensor = torch.cat(all_features, dim=0)
    output = {
        "manifest": str(manifest),
        "image_root": str(image_root),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "classes": build_classes(labels, class_ids),
        "features": feature_tensor,
        "model": model_name,
        "pretrained": pretrained,
        "clip_precision": clip_precision,
        "tta_crops": tta_crops,
        "preprocess_mode": preprocess_mode,
        "amp": amp,
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
        "tta_crops": tta_crops,
        "preprocess_mode": preprocess_mode,
        "shards": shard_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--model", default="local-dir:work/hf_models/bioclip-2.5-vith14")
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--preprocess-mode", choices=["model", "letterbox"], default="model")
    parser.add_argument("--clip-precision", default="fp16")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--load-on-cpu", action="store_true", help="Load checkpoint on CPU before moving model to CUDA.")
    parser.add_argument(
        "--load-safetensors-bytes",
        action="store_true",
        help="Opt-in workaround for Windows pagefile mmap failures when loading safetensors.",
    )
    parser.add_argument(
        "--preload-safetensors-state",
        action="store_true",
        help="Load local-dir safetensors state dict before model construction, then create model with load_weights=False.",
    )
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    import open_clip

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")

    pretrained = None if str(args.pretrained).lower() in {"none", "null", ""} else args.pretrained
    if args.load_safetensors_bytes:
        from fishnet.safetensors_compat import patch_open_clip_safetensors_load_file

        patch_open_clip_safetensors_load_file()
    preloaded_state = None
    if args.preload_safetensors_state:
        if not str(args.model).startswith("local-dir:"):
            raise ValueError("--preload-safetensors-state currently supports only local-dir: models")
        from fishnet.safetensors_compat import load_state_dict_without_mmap

        model_dir = Path(str(args.model).split("local-dir:", 1)[1])
        checkpoint_path = model_dir / "open_clip_model.safetensors"
        preloaded_state = load_state_dict_without_mmap(checkpoint_path)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_device = torch.device("cpu") if args.load_on_cpu else device
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        precision=args.clip_precision,
        device=load_device,
        load_weights=not args.preload_safetensors_state,
    )
    if load_device != device:
        model = model.to(device)
    if preloaded_state is not None:
        model.load_state_dict(preloaded_state, strict=True)
        del preloaded_state
    model = model.eval()
    amp = (not args.no_amp) and device.type == "cuda"
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "model": args.model,
        "pretrained": args.pretrained,
        "device": args.device,
        "clip_precision": args.clip_precision,
        "tta_crops": args.tta_crops,
        "preprocess_mode": args.preprocess_mode,
        "max_samples": args.max_samples,
        "load_on_cpu": args.load_on_cpu,
        "load_safetensors_bytes": args.load_safetensors_bytes,
        "preload_safetensors_state": args.preload_safetensors_state,
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
            encode_rows(
                rows=shard_rows,
                shard_index=shard_index,
                shard_dir=args.shard_dir,
                image_root=args.image_root,
                preprocess=preprocess_val,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tta_crops=args.tta_crops,
                preprocess_mode=args.preprocess_mode,
                amp=amp,
                args_hash=args_hash,
            )
        )

    merged = merge_shards(
        shard_dir=args.shard_dir,
        shard_count=shard_count,
        out=args.out,
        manifest=args.manifest,
        image_root=args.image_root,
        model_name=args.model,
        pretrained=args.pretrained,
        clip_precision=args.clip_precision,
        tta_crops=args.tta_crops,
        preprocess_mode=args.preprocess_mode,
        amp=amp,
    )
    summary = {
        **merged,
        "device": str(device),
        "gpu": gpu_snapshot(),
        "shard_summaries": shard_summaries,
    }
    (args.out.parent / f"{args.out.stem}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
