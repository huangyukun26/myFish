from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import Sam2Model, Sam2Processor


DINO_MEAN_RGB = (124, 116, 104)
VIEW_NAMES = ("mask_blur", "mask_gray", "mask_crop", "mask_public")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_boxes(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["image_id"]] = row
    return result


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row["image_id"]] = row
    return completed


def stable_index(image_id: str, seed: int, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def image_files(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in allowed
    )


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    fraction: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * fraction
    pad_y = (y1 - y0) * fraction
    return (
        max(0, int(math.floor(x0 - pad_x))),
        max(0, int(math.floor(y0 - pad_y))),
        min(width, int(math.ceil(x1 + pad_x))),
        min(height, int(math.ceil(y1 + pad_y))),
    )


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    image.convert("RGB").save(
        temporary,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
    )
    temporary.replace(path)


def build_views(
    image: Image.Image,
    mask: np.ndarray,
    background_path: Path | None,
    crop_expand: float,
) -> tuple[dict[str, Image.Image], tuple[int, int, int, int]]:
    width, height = image.size
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    feather_radius = max(1.0, min(8.0, min(width, height) * 0.004))
    alpha = mask_image.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    blur_radius = max(5.0, min(48.0, min(width, height) * 0.045))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    gray = Image.new("RGB", image.size, DINO_MEAN_RGB)
    views = {
        "mask_blur": Image.composite(image, blurred, alpha),
        "mask_gray": Image.composite(image, gray, alpha),
    }

    bbox = mask_bbox(mask)
    if bbox is None:
        raise RuntimeError("Cannot build views from an empty mask")
    crop_box = expanded_box(bbox, width, height, crop_expand)
    crop_source = Image.composite(image, gray, alpha).crop(crop_box)
    views["mask_crop"] = crop_source

    if background_path is not None:
        with Image.open(background_path) as source:
            background = ImageOps.fit(
                source.convert("RGB"),
                image.size,
                method=Image.Resampling.LANCZOS,
            )
        background = ImageEnhance.Color(background).enhance(0.65)
        public_blur = max(3.0, min(32.0, min(width, height) * 0.02))
        background = background.filter(ImageFilter.GaussianBlur(radius=public_blur))
        views["mask_public"] = Image.composite(image, background, alpha)
    return views, crop_box


def write_view_manifests(
    *,
    records_path: Path,
    source_rows: list[dict[str, str]],
    out_root: Path,
) -> dict[str, int]:
    records = load_completed(records_path)
    source_by_id = {row["image_id"]: row for row in source_rows}
    counts: dict[str, int] = {}
    for view_name in VIEW_NAMES:
        view_dir = out_root / view_name
        selected: list[dict[str, str]] = []
        for image_id, row in source_by_id.items():
            record = records.get(image_id)
            if not record or record.get("status") not in {"ok", "fallback_original"}:
                continue
            if not record.get("views", {}).get(view_name):
                continue
            if not (view_dir / image_id).exists():
                continue
            selected.append(row)
        manifest_path = out_root / f"{view_name}_manifest.csv"
        fieldnames = list(source_rows[0])
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
        counts[view_name] = len(selected)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate class-agnostic SAM2 foreground views from competition images. "
            "The optional public backgrounds must be generic non-target images."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--background-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--crop-expand", type=float, default=0.08)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-area", type=float, default=0.90)
    parser.add_argument("--min-mask-inside-prompt", type=float, default=0.70)
    parser.add_argument("--min-mask-prompt-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-mask-prompt-area-ratio", type=float, default=1.50)
    parser.add_argument("--reject-dino-fallback", action="store_true")
    parser.add_argument(
        "--fallback-original-on-reject",
        action="store_true",
        help="Preserve full manifest coverage by writing the original image for rejected masks.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")
    boxes = load_boxes(args.boxes)
    missing_boxes = [row["image_id"] for row in rows if row["image_id"] not in boxes]
    if missing_boxes:
        raise RuntimeError(
            f"Missing DINO boxes for {len(missing_boxes)} rows; first={missing_boxes[:5]}"
        )

    backgrounds = image_files(args.background_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for view_name in VIEW_NAMES:
        (args.out_root / view_name).mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (args.out_root / "masks").mkdir(parents=True, exist_ok=True)

    records_path = args.out_root / "records.jsonl"
    completed = load_completed(records_path) if args.resume else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This pipeline requires CUDA for practical SAM2 inference")
    processor = Sam2Processor.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    model = Sam2Model.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
    ).eval().to(device)

    processed = 0
    ok = 0
    rejected = 0
    failed = 0
    started = time.perf_counter()
    with records_path.open("a", encoding="utf-8") as record_handle:
        for index, row in enumerate(rows):
            image_id = row["image_id"]
            prior = completed.get(image_id)
            if prior and prior.get("status") in {"ok", "fallback_original"}:
                expected = [
                    args.out_root / view_name / image_id
                    for view_name in prior.get("views", {})
                ]
                if expected and all(path.exists() for path in expected):
                    ok += 1
                    continue

            row_started = time.perf_counter()
            record: dict[str, Any] = {
                "index": index,
                "image_id": image_id,
                "status": "failed",
                "views": {},
            }
            try:
                box_row = boxes[image_id]
                with Image.open(args.image_root / image_id) as source:
                    image = source.convert("RGB")
                width, height = image.size
                x0, y0, x1, y1 = [float(value) for value in box_row["crop_box"]]
                x0 = min(max(x0, 0.0), float(width - 1))
                y0 = min(max(y0, 0.0), float(height - 1))
                x1 = min(max(x1, x0 + 1.0), float(width))
                y1 = min(max(y1, y0 + 1.0), float(height))

                inputs = processor(
                    images=image,
                    input_boxes=[[[x0, y0, x1, y1]]],
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16
                ):
                    outputs = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(
                    outputs.pred_masks.float().cpu(),
                    inputs["original_sizes"].cpu(),
                    binarize=True,
                )
                mask = masks[0][0, 0].numpy().astype(bool)
                score = float(outputs.iou_scores[0, 0, 0].float().cpu())
                area_pixels = int(mask.sum())
                area_fraction = area_pixels / float(width * height)
                ix0, iy0, ix1, iy1 = (
                    int(math.floor(x0)),
                    int(math.floor(y0)),
                    int(math.ceil(x1)),
                    int(math.ceil(y1)),
                )
                inside = int(mask[iy0:iy1, ix0:ix1].sum())
                inside_fraction = inside / max(area_pixels, 1)
                prompt_area = max((x1 - x0) * (y1 - y0), 1.0)
                mask_prompt_area_ratio = area_pixels / prompt_area
                bbox = mask_bbox(mask)
                dino_fallback = bool(box_row.get("fallback", False))

                valid = (
                    bbox is not None
                    and score >= args.min_score
                    and args.min_area <= area_fraction <= args.max_area
                    and inside_fraction >= args.min_mask_inside_prompt
                    and args.min_mask_prompt_area_ratio
                    <= mask_prompt_area_ratio
                    <= args.max_mask_prompt_area_ratio
                    and not (args.reject_dino_fallback and dino_fallback)
                )
                record.update(
                    {
                        "width": width,
                        "height": height,
                        "prompt_box": [x0, y0, x1, y1],
                        "dino_fallback": dino_fallback,
                        "sam_score": score,
                        "mask_pixels": area_pixels,
                        "mask_area_fraction": area_fraction,
                        "mask_inside_prompt_fraction": inside_fraction,
                        "mask_prompt_area_ratio": mask_prompt_area_ratio,
                        "mask_box": list(bbox) if bbox else None,
                    }
                )
                if not valid:
                    record["reason"] = "mask_quality_gate"
                    if args.fallback_original_on_reject:
                        fallback_views = ["mask_blur", "mask_gray", "mask_crop"]
                        if backgrounds:
                            fallback_views.append("mask_public")
                        for view_name in fallback_views:
                            out_path = args.out_root / view_name / image_id
                            save_jpeg(image, out_path, args.jpeg_quality)
                            record["views"][view_name] = str(
                                out_path.relative_to(args.out_root)
                            ).replace("\\", "/")
                        record["status"] = "fallback_original"
                    else:
                        record["status"] = "rejected"
                        rejected += 1
                else:
                    background_path = None
                    if backgrounds:
                        background_path = backgrounds[
                            stable_index(image_id, args.seed, len(backgrounds))
                        ]
                    views, crop_box = build_views(
                        image,
                        mask,
                        background_path,
                        args.crop_expand,
                    )
                    for view_name, view in views.items():
                        out_path = args.out_root / view_name / image_id
                        save_jpeg(view, out_path, args.jpeg_quality)
                        record["views"][view_name] = str(
                            out_path.relative_to(args.out_root)
                        ).replace("\\", "/")
                    if args.save_masks:
                        mask_path = args.out_root / "masks" / f"{Path(image_id).stem}.png"
                        Image.fromarray(mask.astype(np.uint8) * 255).save(
                            mask_path,
                            optimize=True,
                        )
                        record["mask_path"] = str(
                            mask_path.relative_to(args.out_root)
                        ).replace("\\", "/")
                    record["crop_box"] = list(crop_box)
                    record["background_image_id"] = (
                        background_path.stem if background_path else None
                    )
                    record["status"] = "ok"
                    ok += 1
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            record["elapsed_sec"] = time.perf_counter() - row_started
            record_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_handle.flush()
            processed += 1
            if processed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "processed_this_run": processed,
                            "ok_total": ok,
                            "rejected_total": rejected,
                            "failed_total": failed,
                            "elapsed_sec": round(time.perf_counter() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    manifest_counts = write_view_manifests(
        records_path=records_path,
        source_rows=rows,
        out_root=args.out_root,
    )
    latest = load_completed(records_path)
    scores = [
        float(latest[row["image_id"]]["sam_score"])
        for row in rows
        if latest.get(row["image_id"], {}).get("status") == "ok"
    ]
    areas = [
        float(latest[row["image_id"]]["mask_area_fraction"])
        for row in rows
        if latest.get(row["image_id"], {}).get("status") == "ok"
    ]
    summary = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "boxes": str(args.boxes),
        "model_dir": str(args.model_dir),
        "background_root": str(args.background_root) if args.background_root else None,
        "background_count": len(backgrounds),
        "requested_rows": len(rows),
        "ok_rows": len(scores),
        "fallback_original_rows": sum(
            latest.get(row["image_id"], {}).get("status") == "fallback_original"
            for row in rows
        ),
        "rejected_rows": sum(
            latest.get(row["image_id"], {}).get("status") == "rejected"
            for row in rows
        ),
        "failed_rows": sum(
            latest.get(row["image_id"], {}).get("status") == "failed"
            for row in rows
        ),
        "usable_fraction": len(scores) / len(rows),
        "mean_sam_score": float(np.mean(scores)) if scores else None,
        "median_sam_score": float(np.median(scores)) if scores else None,
        "mean_mask_area_fraction": float(np.mean(areas)) if areas else None,
        "median_mask_area_fraction": float(np.median(areas)) if areas else None,
        "view_manifest_counts": manifest_counts,
        "elapsed_sec": time.perf_counter() - started,
        "device": str(device),
        "quality_gate": {
            "min_score": args.min_score,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_mask_inside_prompt": args.min_mask_inside_prompt,
            "min_mask_prompt_area_ratio": args.min_mask_prompt_area_ratio,
            "max_mask_prompt_area_ratio": args.max_mask_prompt_area_ratio,
            "reject_dino_fallback": args.reject_dino_fallback,
            "fallback_original_on_reject": args.fallback_original_on_reject,
        },
    }
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
