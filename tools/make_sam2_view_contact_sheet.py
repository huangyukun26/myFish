from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


VIEWS = ("original", "mask_overlay", "mask_blur", "mask_gray", "mask_crop", "mask_public")


def fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (238, 238, 238))
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.count]
    tile_size = (220, 165)
    header_height = 28
    label_height = 34
    sheet = Image.new(
        "RGB",
        (len(VIEWS) * tile_size[0], header_height + len(rows) * (tile_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, name in enumerate(VIEWS):
        draw.text((column * tile_size[0] + 6, 8), name, fill=(15, 15, 15), font=font)

    for row_index, row in enumerate(rows):
        image_id = row["image_id"]
        with Image.open(args.image_root / image_id) as source:
            original = source.convert("RGB")
        mask_path = args.view_root / "masks" / f"{Path(image_id).stem}.png"
        with Image.open(mask_path) as source:
            mask = source.convert("L")
        red = Image.new("RGB", original.size, (255, 20, 20))
        overlay = Image.blend(original, Image.composite(red, original, mask), 0.35)
        images = {"original": original, "mask_overlay": overlay}
        for view_name in VIEWS[2:]:
            with Image.open(args.view_root / view_name / image_id) as source:
                images[view_name] = source.convert("RGB")

        y = header_height + row_index * (tile_size[1] + label_height)
        for column, name in enumerate(VIEWS):
            sheet.paste(fit_tile(images[name], tile_size), (column * tile_size[0], y))
        label = f"{image_id}  {row.get('label', '')}"
        draw.text((6, y + tile_size[1] + 5), label[:90], fill=(20, 20, 20), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=94)
    print(args.out)


if __name__ == "__main__":
    main()
