from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def padded_box(box: list[float], width: int, height: int, pad_frac: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    pad = pad_frac * max(bw, bh)
    x0 = max(0, int(round(x0 - pad)))
    y0 = max(0, int(round(y0 - pad)))
    x1 = min(width, int(round(x1 + pad)))
    y1 = min(height, int(round(y1 + pad)))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1


def choose_box(boxes: torch.Tensor, scores: torch.Tensor, width: int, height: int) -> tuple[list[float] | None, float]:
    if boxes.numel() == 0:
        return None, 0.0
    areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(1) * (boxes[:, 3] - boxes[:, 1]).clamp_min(1)
    image_area = max(1.0, float(width * height))
    area_frac = areas / image_area
    # Prefer confident large boxes; this avoids tiny fish-like detections in multi-object scenes.
    rank_score = scores.float() * area_frac.sqrt().clamp_min(0.05)
    idx = int(rank_score.argmax().item())
    return [float(x) for x in boxes[idx].tolist()], float(scores[idx].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--prompt", default="fish.")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--pad-frac", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    stats = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "out_root": str(args.out_root),
        "model": args.model,
        "prompt": args.prompt,
        "rows": len(rows),
        "detected": 0,
        "fallback_full": 0,
        "scores": [],
        "crop_area_frac": [],
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_summary.with_suffix(".jsonl")
    with detail_path.open("w", encoding="utf-8") as detail_fp:
        for row in tqdm(rows, desc="fish_crop"):
            image_id = row["image_id"]
            src = args.image_root / image_id
            dst = args.out_root / image_id
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as image:
                image = image.convert("RGB")
                width, height = image.size
                inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    target_sizes=[(height, width)],
                )[0]
                box, score = choose_box(result["boxes"].detach().cpu(), result["scores"].detach().cpu(), width, height)
                if box is None:
                    crop_box = (0, 0, width, height)
                    stats["fallback_full"] += 1
                else:
                    crop_box = padded_box(box, width, height, args.pad_frac)
                    stats["detected"] += 1
                    stats["scores"].append(score)
                area_frac = ((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) / max(1, width * height)
                stats["crop_area_frac"].append(area_frac)
                image.crop(crop_box).save(dst, quality=95)
            detail_fp.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "box": list(crop_box),
                        "score": score,
                        "area_frac": area_frac,
                        "fallback": box is None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    summary = {
        **{key: value for key, value in stats.items() if key not in {"scores", "crop_area_frac"}},
        "detected_frac": stats["detected"] / max(1, stats["rows"]),
        "avg_score": mean(stats["scores"]),
        "avg_crop_area_frac": mean(stats["crop_area_frac"]),
        "detail_jsonl": str(detail_path),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
