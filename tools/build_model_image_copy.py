#!/usr/bin/env python3
"""Build a compact, aspect-ratio-preserving image copy for model inference.

Small JPEGs are copied byte-for-byte. Oversized images are resized and/or
re-encoded so a few unusually large originals do not dominate transfer cost.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class Result:
    name: str
    action: str
    input_bytes: int
    output_bytes: int
    width: int = 0
    height: int = 0
    error: str = ""


def _process_one(args: tuple[str, str, int, int, int]) -> Result:
    source_raw, destination_raw, max_side, quality, copy_below_bytes = args
    source = Path(source_raw)
    destination = Path(destination_raw)
    input_bytes = source.stat().st_size

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return Result(
                name=source.name,
                action="existing",
                input_bytes=input_bytes,
                output_bytes=destination.stat().st_size,
            )

        # Dataset sampling shows that almost every sub-threshold JPEG is already
        # below the model-useful resolution. Avoid decoding those 95% of files.
        if input_bytes <= copy_below_bytes:
            shutil.copyfile(source, destination)
            return Result(
                name=source.name,
                action="copied",
                input_bytes=input_bytes,
                output_bytes=destination.stat().st_size,
            )

        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if image.mode != "RGB":
                image = image.convert("RGB")

            action = "reencoded"
            if max(width, height) > max_side:
                scale = max_side / max(width, height)
                target = (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                )
                image = image.resize(target, Image.Resampling.LANCZOS)
                action = "resized"

            temporary = destination.with_suffix(destination.suffix + ".tmp")
            image.save(
                temporary,
                format="JPEG",
                quality=quality,
                subsampling="keep" if getattr(image, "format", None) == "JPEG" else 2,
                optimize=False,
            )
            temporary.replace(destination)
            return Result(
                name=source.name,
                action=action,
                input_bytes=input_bytes,
                output_bytes=destination.stat().st_size,
                width=width,
                height=height,
            )
    except Exception as exc:  # Keep a complete audit instead of aborting the pool.
        return Result(
            name=source.name,
            action="error",
            input_bytes=input_bytes,
            output_bytes=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _tasks(
    files: Iterable[Path],
    output_dir: Path,
    max_side: int,
    quality: int,
    copy_below_bytes: int,
) -> Iterable[tuple[str, str, int, int, int]]:
    for source in files:
        yield (
            str(source),
            str(output_dir / source.name),
            max_side,
            quality,
            copy_below_bytes,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--copy-below-mb", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No JPEG files found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_below_bytes = round(args.copy_below_mb * 1024 * 1024)

    results: list[Result] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        mapped = executor.map(
            _process_one,
            _tasks(files, args.output_dir, args.max_side, args.quality, copy_below_bytes),
            chunksize=16,
        )
        results.extend(tqdm(mapped, total=len(files), desc="compact images"))

    counts: dict[str, int] = {}
    for result in results:
        counts[result.action] = counts.get(result.action, 0) + 1
    report = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "file_count": len(files),
        "max_side": args.max_side,
        "quality": args.quality,
        "copy_below_bytes": copy_below_bytes,
        "actions": counts,
        "input_bytes": sum(item.input_bytes for item in results),
        "output_bytes": sum(item.output_bytes for item in results),
        "errors": [asdict(item) for item in results if item.error],
    }
    report_path = args.report or (args.output_dir.parent / "compact_image_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise RuntimeError(f"Failed to process {len(report['errors'])} images")


if __name__ == "__main__":
    main()
