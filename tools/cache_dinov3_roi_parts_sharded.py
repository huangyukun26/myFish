from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_classes(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _index in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def load_boxes(path: Path):
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output[row["image_id"]] = row
    return output


def letterbox(image: Image.Image, side: int, fill):
    width, height = image.size
    scale = side / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (side - resized_width) // 2
    top = (side - resized_height) // 2
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized_width, top + resized_height)


class RoiDataset(Dataset):
    def __init__(
        self,
        rows,
        image_root: Path,
        boxes,
        image_size: int,
        fill,
        mean,
        std,
        expand: float,
    ):
        self.rows = rows
        self.image_root = image_root
        self.boxes = boxes
        self.image_size = image_size
        self.fill = fill
        self.expand = expand
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
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
                crop_area = 1.0
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
                    crop_area = 1.0
                else:
                    crop_box = (x0, y0, x1, y1)
                    fallback = bool(box_row.get("fallback", False))
                    crop_area = float(box_row.get("crop_area_fraction", 1.0))
            square, content_box = letterbox(
                image.crop(crop_box),
                self.image_size,
                self.fill,
            )
            tensor = self.to_tensor(square)
        class_text = row.get("class_id", "")
        class_id = int(class_text) if class_text not in {"", None} else -1
        return (
            tensor,
            torch.tensor(content_box, dtype=torch.long),
            image_id,
            row.get("label", ""),
            class_id,
            bool(fallback),
            float(crop_area),
        )


def normalize(value):
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def pool_parts(patch_tokens, content_boxes, side, grid_rows, grid_cols):
    batch, patch_count, dim = patch_tokens.shape
    grid = int(math.isqrt(patch_count))
    if grid * grid != patch_count:
        raise RuntimeError("Patch-token count %d is not square" % patch_count)
    patch_grid = patch_tokens.view(batch, grid, grid, dim)
    centers = (torch.arange(grid).float() + 0.5) * side / grid
    part_count = grid_rows * grid_cols
    part_masks = torch.zeros((batch, part_count, grid, grid), dtype=torch.float32)
    valid_masks = torch.zeros((batch, grid, grid), dtype=torch.float32)
    for index in range(batch):
        left, top, right, bottom = [int(value) for value in content_boxes[index].tolist()]
        valid_rows = torch.where(centers.ge(top) & centers.lt(bottom))[0]
        valid_cols = torch.where(centers.ge(left) & centers.lt(right))[0]
        if len(valid_rows) < grid_rows or len(valid_cols) < grid_cols:
            valid_rows = torch.arange(grid)
            valid_cols = torch.arange(grid)
        row_groups = torch.tensor_split(valid_rows, grid_rows)
        col_groups = torch.tensor_split(valid_cols, grid_cols)
        valid_masks[index][valid_rows[:, None], valid_cols[None, :]] = 1.0
        part_index = 0
        for row_group in row_groups:
            for col_group in col_groups:
                part_masks[index, part_index][
                    row_group[:, None], col_group[None, :]
                ] = 1.0
                part_index += 1
    part_masks = part_masks.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    valid_masks = valid_masks.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    part_masks = part_masks / part_masks.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    valid_masks = valid_masks / valid_masks.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    parts = torch.einsum("bphw,bhwd->bpd", part_masks, patch_grid)
    valid_means = torch.einsum("bhw,bhwd->bd", valid_masks, patch_grid)
    return normalize(parts), normalize(valid_means)


def shard_paths(shard_dir: Path, shard_index: int):
    stem = "shard_%05d" % shard_index
    return shard_dir / (stem + ".pt"), shard_dir / (stem + ".summary.json")


def valid_completed_shard(summary_path: Path, expected_rows: int, args_hash):
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return summary.get("rows") == expected_rows and summary.get("args_hash") == args_hash


def encode_shard(
    rows,
    shard_index,
    shard_dir,
    image_root,
    boxes,
    image_size,
    fill,
    mean,
    std,
    expand,
    model,
    device,
    batch_size,
    num_workers,
    amp,
    args_hash,
    grid_rows,
    grid_cols,
):
    dataset = RoiDataset(
        rows,
        image_root,
        boxes,
        image_size,
        fill,
        mean,
        std,
        expand,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    cls_out = []
    valid_mean_out = []
    parts_out = []
    image_ids = []
    labels = []
    class_ids = []
    fallbacks = []
    crop_areas = []
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    started = time.time()
    with torch.inference_mode():
        for tensors, content_boxes, ids, batch_labels, batch_classes, batch_fallbacks, batch_areas in loader:
            tensors = tensors.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                tokens = model.forward_features(tensors)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise RuntimeError("Expected DINO ViT tokens shaped [B, tokens, dim]")
            if tokens.shape[1] <= prefix_tokens:
                raise RuntimeError("DINO output contains no patch tokens")
            patch_tokens = tokens[:, prefix_tokens:]
            pooled_parts, valid_mean = pool_parts(
                patch_tokens,
                content_boxes,
                image_size,
                grid_rows,
                grid_cols,
            )
            cls_out.append(normalize(tokens[:, 0]).half().cpu())
            valid_mean_out.append(valid_mean.half().cpu())
            parts_out.append(pooled_parts.half().cpu())
            image_ids.extend(list(ids))
            labels.extend(list(batch_labels))
            class_ids.extend(int(value) for value in batch_classes.tolist())
            fallbacks.extend(bool(value) for value in batch_fallbacks.tolist())
            crop_areas.extend(float(value) for value in batch_areas.tolist())
    payload = {
        "cls": torch.cat(cls_out, dim=0),
        "valid_mean": torch.cat(valid_mean_out, dim=0),
        "parts": torch.cat(parts_out, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "fallback": torch.tensor(fallbacks, dtype=torch.bool),
        "crop_area_fraction": torch.tensor(crop_areas, dtype=torch.float32),
        "args_hash": args_hash,
    }
    shard_path, summary_path = shard_paths(shard_dir, shard_index)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, shard_path)
    summary = {
        "shard_index": shard_index,
        "rows": len(image_ids),
        "cls_shape": list(payload["cls"].shape),
        "parts_shape": list(payload["parts"].shape),
        "fallback_rows": int(payload["fallback"].sum().item()),
        "elapsed_sec": time.time() - started,
        "shard": str(shard_path),
        "args_hash": args_hash,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def merge_shards(
    shard_dir,
    shard_count,
    out,
    classes,
    manifest,
    image_root,
    boxes_path,
    model_name,
    pretrained_file,
    image_size,
    expand,
    grid_rows,
    grid_cols,
):
    cls_values = []
    valid_means = []
    parts = []
    class_ids = []
    fallbacks = []
    crop_areas = []
    image_ids = []
    labels = []
    for shard_index in range(shard_count):
        shard_path, _summary_path = shard_paths(shard_dir, shard_index)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        cls_values.append(payload["cls"])
        valid_means.append(payload["valid_mean"])
        parts.append(payload["parts"])
        class_ids.append(payload["class_ids"])
        fallbacks.append(payload["fallback"])
        crop_areas.append(payload["crop_area_fraction"])
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
    output = {
        "cls": torch.cat(cls_values, dim=0),
        "valid_mean": torch.cat(valid_means, dim=0),
        "parts": torch.cat(parts, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.cat(class_ids, dim=0),
        "fallback": torch.cat(fallbacks, dim=0),
        "crop_area_fraction": torch.cat(crop_areas, dim=0),
        "classes": classes,
        "manifest": str(manifest),
        "image_root": str(image_root),
        "boxes": str(boxes_path),
        "model": model_name,
        "pretrained_file": str(pretrained_file),
        "image_size": image_size,
        "expand": expand,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "pool": f"valid_content_{grid_rows}x{grid_cols}_row_major",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    return {
        "out": str(out),
        "rows": len(image_ids),
        "cls_shape": list(output["cls"].shape),
        "parts_shape": list(output["parts"].shape),
        "fallback_rows": int(output["fallback"].sum().item()),
        "shards": shard_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument(
        "--class-map",
        type=Path,
        default=Path("work/full_manifests/seen_class_to_idx.json"),
    )
    parser.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--expand", type=float, default=0.08)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.grid_rows < 1 or args.grid_cols < 1:
        raise ValueError("--grid-rows and --grid-cols must be positive")

    import timm
    from timm.data import resolve_model_data_config

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows in %s" % args.manifest)
    classes = load_classes(args.class_map)
    boxes = load_boxes(args.boxes)
    missing = [row["image_id"] for row in rows if row["image_id"] not in boxes]
    if missing:
        raise RuntimeError("ROI boxes missing for %d rows; first=%s" % (len(missing), missing[0]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=0,
        img_size=args.image_size,
        pretrained_cfg_overlay={"file": str(args.pretrained_file)},
    ).to(device).eval()
    config = resolve_model_data_config(model)
    mean = config.get("mean", (0.485, 0.456, 0.406))
    std = config.get("std", (0.229, 0.224, 0.225))
    fill = tuple(round(float(value) * 255) for value in mean)
    amp = (not args.no_amp) and device.type == "cuda"
    args_hash = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "boxes": str(args.boxes),
        "model": args.model,
        "pretrained_file": str(args.pretrained_file),
        "image_size": args.image_size,
        "expand": args.expand,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "max_samples": args.max_samples,
    }
    shard_count = math.ceil(len(rows) / args.shard_size)
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        _shard_path, summary_path = shard_paths(args.shard_dir, shard_index)
        if args.resume and valid_completed_shard(summary_path, len(shard_rows), args_hash):
            print(
                json.dumps(
                    {"shard_index": shard_index, "status": "resume_skip"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        encode_shard(
            shard_rows,
            shard_index,
            args.shard_dir,
            args.image_root,
            boxes,
            args.image_size,
            fill,
            mean,
            std,
            args.expand,
            model,
            device,
            args.batch_size,
            args.num_workers,
            amp,
            args_hash,
            args.grid_rows,
            args.grid_cols,
        )
    summary = merge_shards(
        args.shard_dir,
        shard_count,
        args.out,
        classes,
        args.manifest,
        args.image_root,
        args.boxes,
        args.model,
        args.pretrained_file,
        args.image_size,
        args.expand,
        args.grid_rows,
        args.grid_cols,
    )
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
