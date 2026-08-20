#!/usr/bin/env python
"""Create text-conditioned CLIP occlusion heatmaps for selected images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from fishnet.image_preprocess import letterbox_to_square


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_image(image_id: str, manifest_by_id: dict[str, dict[str, str]], images_zip: Path) -> Image.Image:
    row = manifest_by_id[image_id]
    with zipfile.ZipFile(images_zip) as zf:
        with zf.open(row["zip_member"]) as fp:
            return Image.open(BytesIO(fp.read())).convert("RGB")


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def text_prompts(label: str) -> list[str]:
    return [
        f"a photo of a fish species {label}",
        f"a close-up photo of {label}",
        f"an underwater photo of {label}",
    ]


def encode_text(model, tokenizer, labels: list[str], device: torch.device) -> torch.Tensor:
    toks = tokenizer([prompt for label in labels for prompt in text_prompts(label)]).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        feats = model.encode_text(toks)
    feats = normalize(feats).view(len(labels), 3, -1).mean(dim=1)
    return normalize(feats)


def make_transform(image_size: int):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def occluded_variants(image: Image.Image, grid: int, fill: tuple[int, int, int]) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    base = letterbox_to_square(image, fill=fill)
    w, h = base.size
    variants = [base]
    boxes = []
    for gy in range(grid):
        for gx in range(grid):
            left = round(gx * w / grid)
            top = round(gy * h / grid)
            right = round((gx + 1) * w / grid)
            bottom = round((gy + 1) * h / grid)
            v = base.copy()
            ImageDraw.Draw(v).rectangle([left, top, right, bottom], fill=fill)
            variants.append(v)
            boxes.append((left, top, right, bottom))
    return variants, boxes


def score_image(model, transform, variants: list[Image.Image], text_feats: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    scores = []
    with torch.inference_mode():
        for start in range(0, len(variants), batch_size):
            batch = torch.stack([transform(img) for img in variants[start : start + batch_size]]).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                image_feats = normalize(model.encode_image(batch))
            scores.append(image_feats @ text_feats.T)
    return torch.cat(scores, dim=0).cpu()


def heatmap_overlay(base: Image.Image, drops: list[float], boxes: list[tuple[int, int, int, int]], grid: int, label: str) -> Image.Image:
    canvas = base.resize((336, 336), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    max_drop = max(max(drops), 1e-6)
    scale_x = canvas.width / base.width
    scale_y = canvas.height / base.height
    for drop, box in zip(drops, boxes):
        alpha = int(max(0.0, drop) / max_drop * 180)
        left, top, right, bottom = box
        rect = [round(left * scale_x), round(top * scale_y), round(right * scale_x), round(bottom * scale_y)]
        draw.rectangle(rect, fill=(255, 48, 0, alpha))
    out = Image.alpha_composite(canvas, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(out)
    draw2.rectangle([0, 0, out.width, 24], fill=(255, 255, 255))
    draw2.text((4, 4), label[:48], fill=(20, 20, 20))
    return out


def render_row(image: Image.Image, labels: list[str], score_matrix: torch.Tensor, boxes, grid: int) -> Image.Image:
    base = letterbox_to_square(image)
    original = base.resize((336, 336), Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(original)
    draw.rectangle([0, 0, original.width, 24], fill=(255, 255, 255))
    draw.text((4, 4), "original", fill=(20, 20, 20))
    tiles = [original]
    full_scores = score_matrix[0]
    occluded_scores = score_matrix[1:]
    for j, label in enumerate(labels):
        drops = (full_scores[j] - occluded_scores[:, j]).tolist()
        tiles.append(heatmap_overlay(base, drops, boxes, grid, f"{label} s={float(full_scores[j]):.3f}"))
    sheet = Image.new("RGB", (len(tiles) * 336, 336), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * 336, 0))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("work/seen_image_distribution_split_seed2027_frac20/val.csv"))
    parser.add_argument("--images-zip", type=Path, default=Path("dataset/images.zip"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="local-dir:work/hf_models/bioclip-2.5-vith14")
    args = parser.parse_args()

    import open_clip

    rows = read_rows(args.rows_csv)[: args.count]
    manifest_by_id = {row["image_id"]: row for row in read_rows(args.manifest)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preprocess_train, _preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=None,
        precision="fp16",
        device=device,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    transform = make_transform(args.image_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for row in rows:
        labels = []
        for key in ("true_label", "top1", "external_label"):
            label = str(row.get(key, "")).strip()
            if label and label not in labels:
                labels.append(label)
        if not labels:
            continue
        image = load_image(row["image_id"], manifest_by_id, args.images_zip)
        text_feats = encode_text(model, tokenizer, labels, device)
        variants, boxes = occluded_variants(image, args.grid, fill=(123, 117, 104))
        scores = score_image(model, transform, variants, text_feats, device, args.batch_size)
        sheet = render_row(image, labels, scores, boxes, args.grid)
        out_path = args.out_dir / f"{Path(row['image_id']).stem}_occlusion.jpg"
        sheet.save(out_path, quality=92)
        summary.append(
            {
                "image_id": row["image_id"],
                "labels": labels,
                "full_scores": {label: float(scores[0, i]) for i, label in enumerate(labels)},
                "out": str(out_path),
            }
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
