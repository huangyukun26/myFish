from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("dataset/images.zip"))
    parser.add_argument("--output", type=Path, default=Path("dataset/images"))
    parser.add_argument("--limit", type=int, default=0, help="0 extracts all images.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--summary-out", type=Path, default=Path("work/extract_images_summary.json"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    extracted = 0
    skipped = 0
    bytes_written = 0

    with zipfile.ZipFile(args.zip) as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if args.limit:
            members = members[: args.limit]
        total = len(members)
        for idx, info in enumerate(members, start=1):
            target = args.output / Path(info.filename).name
            if args.resume and target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
            else:
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                        bytes_written += len(chunk)
                extracted += 1
            if idx == 1 or idx % args.progress_every == 0 or idx == total:
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"processed={idx}/{total} extracted={extracted} skipped={skipped} "
                    f"written_gib={bytes_written/(1024**3):.2f} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    summary = {
        "zip": str(args.zip),
        "output": str(args.output),
        "total_members": total,
        "extracted": extracted,
        "skipped": skipped,
        "bytes_written": bytes_written,
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

