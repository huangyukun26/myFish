from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


OFFICIAL_S3_BASE_URL = "https://open-images-dataset.s3.amazonaws.com"
IMAGE_ID_RE = re.compile(r"^[0-9a-fA-F]{16}$")
MANIFEST_FIELDS = [
    "index",
    "image_id",
    "split",
    "relative_path",
    "url",
    "status",
    "http_status",
    "attempts",
    "bytes",
    "sha256",
    "width",
    "height",
    "mode",
    "format",
    "content_type",
    "error",
    "checked_at_utc",
    "safe_positive_labels",
    "license",
    "author",
    "author_profile_url",
    "original_landing_url",
    "original_url",
    "original_md5",
    "original_size",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_id_line(raw: str, line_number: int) -> tuple[str, str] | None:
    value = raw.strip()
    if not value or value.startswith("#"):
        return None
    if "/" in value:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError(f"Line {line_number}: invalid split/id entry: {value!r}")
        split, image_id = parts
    else:
        split, image_id = "validation", value
    if split != "validation":
        raise ValueError(
            f"Line {line_number}: only the official validation split is supported, "
            f"got {split!r}"
        )
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise ValueError(f"Line {line_number}: invalid Open Images ID: {image_id!r}")
    return split, image_id.lower()


def read_ids(path: Path, limit: int | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig") as fp:
        for line_number, raw in enumerate(fp, start=1):
            parsed = parse_id_line(raw, line_number)
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            rows.append(parsed)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No valid image IDs found in {path}")
    return rows


def read_source_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            image_id = row.get("image_id", "").strip().lower()
            if IMAGE_ID_RE.fullmatch(image_id):
                result[image_id] = row
    return result


def hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    # verify() checks the encoded stream; load() on a fresh handle forces a full
    # pixel decode and catches truncated files that can otherwise appear valid.
    with Image.open(path) as image:
        image_format = image.format
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        decoded_format = image.format
    # Open Images serves a small number of JPEG-compatible multi-picture
    # objects with a .jpg key. Pillow identifies those streams as MPO; the
    # first frame is still fully decodable and usable as an RGB image.
    accepted_formats = {"JPEG", "MPO"}
    if image_format not in accepted_formats or decoded_format not in accepted_formats:
        raise ValueError(
            "Expected a JPEG-compatible Open Images object, got "
            f"{image_format!r}/{decoded_format!r}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid decoded dimensions: {width}x{height}")
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "format": decoded_format,
    }


def inspect_existing(path: Path) -> dict[str, Any]:
    image_info = inspect_image(path)
    size, sha256 = hash_file(path)
    if size <= 0:
        raise ValueError("File is empty")
    return {**image_info, "bytes": size, "sha256": sha256}


def clean_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    return " ".join(text.replace("\x00", " ").split())[:1000]


def source_columns(metadata: dict[str, str]) -> dict[str, str]:
    return {
        "safe_positive_labels": metadata.get("safe_positive_labels", ""),
        "license": metadata.get("license", ""),
        "author": metadata.get("author", ""),
        "author_profile_url": metadata.get("author_profile_url", ""),
        "original_landing_url": metadata.get("original_landing_url", ""),
        "original_url": metadata.get("original_url", ""),
        "original_md5": metadata.get("original_md5", ""),
        "original_size": metadata.get("original_size", ""),
    }


def download_one(
    *,
    index: int,
    split: str,
    image_id: str,
    output_dir: Path,
    base_url: str,
    timeout_seconds: float,
    retries: int,
    metadata: dict[str, str],
) -> dict[str, Any]:
    relative_path = f"{image_id}.jpg"
    destination = output_dir / relative_path
    url = f"{base_url.rstrip('/')}/{split}/{image_id}.jpg"
    common: dict[str, Any] = {
        "index": index,
        "image_id": image_id,
        "split": split,
        "relative_path": relative_path,
        "url": url,
        "http_status": "",
        "attempts": 0,
        "bytes": "",
        "sha256": "",
        "width": "",
        "height": "",
        "mode": "",
        "format": "",
        "content_type": "",
        "error": "",
        "checked_at_utc": utc_now(),
        **source_columns(metadata),
    }

    existing_error = ""
    if destination.exists():
        try:
            info = inspect_existing(destination)
            return {
                **common,
                **info,
                "status": "existing_valid",
                "checked_at_utc": utc_now(),
            }
        except Exception as error:  # noqa: BLE001 - recorded and repaired below
            existing_error = f"Existing file invalid ({clean_error(error)}). "

    last_error = ""
    last_http_status: int | str = ""
    last_content_type = ""
    for attempt in range(1, retries + 1):
        temp_path: Path | None = None
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "fishnet-openimages-subset-downloader/1.0",
                    "Accept": "image/jpeg,image/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                http_status = int(response.getcode())
                last_http_status = http_status
                if http_status != 200:
                    raise RuntimeError(f"Unexpected HTTP status {http_status}")
                content_type = response.headers.get("Content-Type", "")
                last_content_type = content_type
                expected_length_raw = response.headers.get("Content-Length")
                expected_length = (
                    int(expected_length_raw) if expected_length_raw else None
                )
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{image_id}.",
                    suffix=".part",
                    dir=output_dir,
                    delete=False,
                ) as temp_fp:
                    temp_path = Path(temp_fp.name)
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temp_fp.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    temp_fp.flush()
                    os.fsync(temp_fp.fileno())

            if size <= 0:
                raise ValueError("Downloaded file is empty")
            if expected_length is not None and size != expected_length:
                raise ValueError(
                    f"Content-Length mismatch: expected {expected_length}, got {size}"
                )
            image_info = inspect_image(temp_path)
            os.replace(temp_path, destination)
            temp_path = None
            return {
                **common,
                **image_info,
                "status": (
                    "downloaded_replaced_invalid"
                    if existing_error
                    else "downloaded"
                ),
                "http_status": http_status,
                "attempts": attempt,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "content_type": content_type,
                "error": existing_error.strip(),
                "checked_at_utc": utc_now(),
            }
        except Exception as error:  # noqa: BLE001 - bounded retries and audit log
            last_error = clean_error(error)
            if isinstance(error, urllib.error.HTTPError):
                last_http_status = error.code
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))

    return {
        **common,
        "status": "failed",
        "http_status": last_http_status,
        "attempts": retries,
        "content_type": last_content_type,
        "error": (existing_error + last_error).strip(),
        "checked_at_utc": utc_now(),
    }


def write_manifest_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["index"])))
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp_path, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an auditable, resumable Open Images V7 validation subset "
            "from the official public S3 bucket."
        )
    )
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional CSV carrying license/author/original-source metadata.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--base-url", default=OFFICIAL_S3_BASE_URL)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even when one or more files fail.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.retries <= 0:
        parser.error("--retries must be positive")
    return args


def main() -> None:
    args = parse_args()
    ids = read_ids(args.ids, args.limit)
    metadata_by_id = read_source_metadata(args.source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    results: list[dict[str, Any]] = []
    with args.events.open("a", encoding="utf-8", buffering=1) as event_fp:
        event_fp.write(
            json.dumps(
                {
                    "event": "run_started",
                    "at_utc": started_at,
                    "ids": str(args.ids),
                    "requested": len(ids),
                    "output_dir": str(args.output_dir),
                    "base_url": args.base_url,
                    "workers": args.workers,
                    "retries": args.retries,
                    "timeout_seconds": args.timeout_seconds,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one,
                    index=index,
                    split=split,
                    image_id=image_id,
                    output_dir=args.output_dir,
                    base_url=args.base_url,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                    metadata=metadata_by_id.get(image_id, {}),
                ): (index, image_id)
                for index, (split, image_id) in enumerate(ids)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                event_fp.write(
                    json.dumps(
                        {"event": "item_finished", **result},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if completed == 1 or completed % 25 == 0 or completed == len(ids):
                    valid = sum(
                        row["status"] != "failed" for row in results
                    )
                    failed = completed - valid
                    print(
                        f"[{completed}/{len(ids)}] valid={valid} failed={failed}",
                        flush=True,
                    )

        write_manifest_atomic(args.manifest, results)
        status_counts: dict[str, int] = {}
        for row in results:
            status = str(row["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        valid_rows = [row for row in results if row["status"] != "failed"]
        failed_rows = [row for row in results if row["status"] == "failed"]
        summary = {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "ids_path": str(args.ids),
            "source_manifest": (
                str(args.source_manifest) if args.source_manifest else None
            ),
            "output_dir": str(args.output_dir),
            "manifest": str(args.manifest),
            "events": str(args.events),
            "base_url": args.base_url,
            "requested": len(ids),
            "valid": len(valid_rows),
            "failed": len(failed_rows),
            "status_counts": status_counts,
            "valid_bytes": sum(int(row["bytes"]) for row in valid_rows),
            "failed_items": [
                {
                    "image_id": row["image_id"],
                    "url": row["url"],
                    "http_status": row["http_status"],
                    "attempts": row["attempts"],
                    "error": row["error"],
                }
                for row in failed_rows
            ],
        }
        write_json_atomic(args.summary, summary)
        event_fp.write(
            json.dumps(
                {"event": "run_finished", **summary},
                ensure_ascii=False,
            )
            + "\n"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed_rows and not args.allow_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
