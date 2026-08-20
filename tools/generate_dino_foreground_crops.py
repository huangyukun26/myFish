from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def letterbox(image: Image.Image, side: int, fill: tuple[int, int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
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


def robust_zscore(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = values[valid]
    median = selected.median()
    mad = (selected - median).abs().median().clamp_min(1e-4)
    return (values - median) / (1.4826 * mad)


def components(mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    output: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(visited[row, col]):
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and bool(mask[next_row, next_col])
                        and not bool(visited[next_row, next_col])
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            output.append(component)
    return output


def content_patch_mask(grid: int, side: int, content_box: tuple[int, int, int, int]) -> torch.Tensor:
    left, top, right, bottom = content_box
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) * side / grid
    valid_x = centers.ge(left) & centers.lt(right)
    valid_y = centers.ge(top) & centers.lt(bottom)
    return valid_y[:, None] & valid_x[None, :]


def foreground_box(
    tokens: torch.Tensor,
    *,
    prefix_tokens: int,
    content_box: tuple[int, int, int, int],
    side: int,
    quantile: float,
    expansion_patches: int,
) -> tuple[tuple[int, int, int, int], dict[str, float | int | bool]]:
    patch_tokens = tokens[prefix_tokens:]
    grid = int(math.isqrt(len(patch_tokens)))
    if grid * grid != len(patch_tokens):
        raise RuntimeError(f"Patch-token count {len(patch_tokens)} is not square")
    patch_tokens = F.normalize(patch_tokens.float(), dim=1).view(grid, grid, -1)
    cls_token = F.normalize(tokens[0].float(), dim=0)
    valid = content_patch_mask(grid, side, content_box)

    cls_similarity = torch.einsum("hwd,d->hw", patch_tokens, cls_token)
    eroded = ~F.max_pool2d((~valid).float()[None, None], 3, 1, 1).bool()[0, 0]
    border = valid & ~eroded
    if int(border.sum()) < 2:
        border = valid
    background = F.normalize(patch_tokens[border].mean(dim=0), dim=0)
    background_difference = 1.0 - torch.einsum("hwd,d->hw", patch_tokens, background)
    score = robust_zscore(cls_similarity, valid) + robust_zscore(background_difference, valid)
    score = score.masked_fill(~valid, float("-inf"))

    threshold = torch.quantile(score[valid], quantile)
    selected = score.ge(threshold) & valid
    selected = F.max_pool2d(selected.float()[None, None], 3, 1, 1)[0, 0].bool() & valid
    candidates = components(selected)
    if not candidates:
        candidates = [[tuple(int(value) for value in torch.nonzero(score.eq(score[valid].max()))[0].tolist())]]

    def component_value(component: list[tuple[int, int]]) -> float:
        rows = torch.tensor([item[0] for item in component], dtype=torch.long)
        cols = torch.tensor([item[1] for item in component], dtype=torch.long)
        return float(score[rows, cols].clamp_min(0).sum()) + 0.05 * len(component)

    best = max(candidates, key=component_value)
    rows = [item[0] for item in best]
    cols = [item[1] for item in best]
    row_min = max(0, min(rows) - expansion_patches)
    row_max = min(grid, max(rows) + 1 + expansion_patches)
    col_min = max(0, min(cols) - expansion_patches)
    col_max = min(grid, max(cols) + 1 + expansion_patches)
    patch_size = side / grid
    square_box = (
        round(col_min * patch_size),
        round(row_min * patch_size),
        round(col_max * patch_size),
        round(row_max * patch_size),
    )
    left, top, right, bottom = content_box
    square_left = max(left, square_box[0])
    square_top = max(top, square_box[1])
    square_right = min(right, square_box[2])
    square_bottom = min(bottom, square_box[3])
    width = right - left
    height = bottom - top
    crop_box = (
        max(0, round((square_left - left) * side / width)),
        max(0, round((square_top - top) * side / height)),
        min(side, round((square_right - left) * side / width)),
        min(side, round((square_bottom - top) * side / height)),
    )
    area_fraction = max(0, crop_box[2] - crop_box[0]) * max(0, crop_box[3] - crop_box[1]) / (side * side)
    fallback = area_fraction < 0.03 or area_fraction > 0.95 or len(best) < 2
    if fallback:
        crop_box = (0, 0, side, side)
        area_fraction = 1.0
    return crop_box, {
        "grid": grid,
        "component_patches": len(best),
        "selected_patches": int(selected.sum()),
        "valid_patches": int(valid.sum()),
        "crop_area_fraction": area_fraction,
        "fallback": fallback,
    }


def resize_for_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def save_contact_sheets(panels: list[tuple[Image.Image, Image.Image, str]], out_dir: Path, page_size: int = 8) -> None:
    panel_width, panel_height = 600, 250
    for page_index, start in enumerate(range(0, len(panels), page_size)):
        page = Image.new("RGB", (panel_width * 2, panel_height * 2), "white")
        draw = ImageDraw.Draw(page)
        for offset, (overlay, crop, label) in enumerate(panels[start : start + page_size]):
            row, col = divmod(offset, 4)
            x = col * (panel_width // 2)
            y = row * panel_height
            page.paste(resize_for_panel(overlay, (145, 210)), (x, y + 24))
            page.paste(resize_for_panel(crop, (145, 210)), (x + 150, y + 24))
            draw.text((x + 4, y + 4), label[:43], fill="black")
        page.save(out_dir / f"contact_{page_index:03d}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--quantile", type=float, default=0.70)
    parser.add_argument("--expansion-patches", type=int, default=2)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--contact-sheet-limit", type=int, default=0)
    args = parser.parse_args()

    import timm
    from timm.data import resolve_model_data_config

    rows = read_manifest(args.manifest)
    if args.max_samples and args.max_samples < len(rows):
        rows = random.Random(args.seed).sample(rows, args.max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=args.image_size).to(device).eval()
    config = resolve_model_data_config(model)
    mean = config.get("mean", (0.485, 0.456, 0.406))
    std = config.get("std", (0.229, 0.224, 0.225))
    fill = tuple(round(float(value) * 255) for value in mean)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    crop_dir = args.out_dir / "crops"
    overlay_dir = args.out_dir / "overlays"
    crop_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    records = []
    panels: list[tuple[Image.Image, Image.Image, str]] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="dino_foreground"):
        batch_rows = rows[start : start + args.batch_size]
        originals = []
        model_inputs = []
        content_boxes = []
        for row in batch_rows:
            with Image.open(args.image_root / row["image_id"]) as handle:
                image = handle.convert("RGB")
            square, content_box = letterbox(image, args.image_size, fill)
            originals.append(image)
            model_inputs.append(normalize(square))
            content_boxes.append(content_box)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            tokens = model.forward_features(torch.stack(model_inputs).to(device)).float().cpu()
        for row, image, content_box, image_tokens in zip(batch_rows, originals, content_boxes, tokens):
            normalized_box, diagnostics = foreground_box(
                image_tokens,
                prefix_tokens=int(getattr(model, "num_prefix_tokens", 1)),
                content_box=content_box,
                side=args.image_size,
                quantile=args.quantile,
                expansion_patches=args.expansion_patches,
            )
            width, height = image.size
            crop_box = (
                round(normalized_box[0] * width / args.image_size),
                round(normalized_box[1] * height / args.image_size),
                round(normalized_box[2] * width / args.image_size),
                round(normalized_box[3] * height / args.image_size),
            )
            crop = image.crop(crop_box)
            crop.save(crop_dir / row["image_id"], quality=95)
            overlay = image.copy()
            ImageDraw.Draw(overlay).rectangle(crop_box, outline=(255, 40, 20), width=max(2, round(min(width, height) / 150)))
            if args.save_overlays:
                overlay.save(overlay_dir / row["image_id"], quality=92)
            if len(panels) < args.contact_sheet_limit:
                panels.append((overlay, crop, f"{row['image_id']} {row.get('label', '')}"))
            records.append(
                {
                    "image_id": row["image_id"],
                    "label": row.get("label", ""),
                    "original_width": width,
                    "original_height": height,
                    "aspect_ratio": width / max(1, height),
                    "crop_box": crop_box,
                    **diagnostics,
                }
            )

    with (args.out_dir / "boxes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if panels:
        save_contact_sheets(panels, args.out_dir)
    area = torch.tensor([float(record["crop_area_fraction"]) for record in records])
    summary = {
        "rows": len(records),
        "fallbacks": sum(bool(record["fallback"]) for record in records),
        "crop_area_median": float(area.median()),
        "crop_area_mean": float(area.mean()),
        "quantile": args.quantile,
        "expansion_patches": args.expansion_patches,
        "model": args.model,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
