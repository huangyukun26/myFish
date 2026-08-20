from __future__ import annotations

import argparse
import csv
import random
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=Path("dataset/images.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cols", type=int, default=8)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.count, len(rows)))
    tiles = []
    with zipfile.ZipFile(args.zip) as zf:
        for row in sample:
            with zf.open(row["zip_member"]) as fp:
                image = Image.open(BytesIO(fp.read())).convert("RGB")
            image.thumbnail((180, 135), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (200, 175), (245, 245, 245))
            tile.paste(image, ((200 - image.width) // 2, 8))
            draw = ImageDraw.Draw(tile)
            label = row.get("label") or row.get("split", "")
            draw.text((8, 146), Path(row["image_id"]).name, fill=(20, 20, 20))
            draw.text((8, 160), label[:32], fill=(20, 20, 20))
            tiles.append(tile)

    rows_n = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * 200, rows_n * 175), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % args.cols) * 200, (i // args.cols) * 175))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(args.out)


if __name__ == "__main__":
    main()

