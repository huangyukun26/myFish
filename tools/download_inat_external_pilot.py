#!/usr/bin/env python
"""Download audited iNaturalist pilot images.

Input must be the metadata CSV produced by audit_inat_external_coverage.py.
The output manifest keeps source URL, license, attribution, and local path for
each downloaded image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_manifest(path: Path, manifest: list[dict[str, str]]) -> None:
    fieldnames = []
    for row in manifest:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
    tmp.replace(path)


def write_external_manifest(path: Path, manifest: list[dict[str, str]], split: str) -> None:
    fieldnames = ["image_id", "zip_member", "label", "class_id", "split"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            local_name = Path(row["local_path"]).name
            writer.writerow(
                {
                    "image_id": local_name,
                    "zip_member": local_name,
                    "label": row["fishnet_species"],
                    "class_id": row["class_id"],
                    "split": split,
                }
            )
    tmp.replace(path)


def write_summary(args: argparse.Namespace, counts: dict[str, int], manifest: list[dict[str, str]], events_path: Path, partial: bool) -> None:
    summary = {
        "metadata_csv": str(args.metadata_csv),
        "image_dir": str(args.image_dir),
        "downloaded": len(manifest),
        "classes": len(counts),
        "max_per_class": args.max_per_class,
        "max_total": args.max_total,
        "events_jsonl": str(events_path),
        "manifest_csv": str(args.out / "download_manifest.csv"),
        "external_manifest_csv": str(args.out / "external_manifest.csv"),
        "partial": partial,
    }
    with (args.out / "download_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def fetch_bytes(url: str, *, timeout: float, retries: int, pause: float) -> bytes:
    headers = {
        "User-Agent": "fishnet-external-data-pilot/1.0 (public image download)",
        "Accept": "image/*,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"download failed after {retries + 1} attempts: {url} :: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument("--max-total", type=int, default=500)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Load existing download_manifest.csv and continue.")
    parser.add_argument("--flush-every", type=int, default=10, help="Write partial manifests after this many successful downloads.")
    parser.add_argument("--external-split", default="external_inat")
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.metadata_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: (int(r["class_id"]), r["fishnet_species"], r["observation_id"], r["photo_id"]))

    manifest: list[dict[str, str]] = read_existing_csv(args.out / "download_manifest.csv") if args.resume else []
    counts: dict[str, int] = {}
    completed_keys: set[tuple[str, str, str]] = set()
    for item in manifest:
        counts[item["class_id"]] = counts.get(item["class_id"], 0) + 1
        completed_keys.add((item["class_id"], item.get("observation_id", ""), item.get("photo_id", "")))
    events_path = args.out / "download_events.jsonl"

    event_mode = "a" if args.resume else "w"
    successes_since_flush = 0
    with events_path.open(event_mode, encoding="utf-8") as events:
        for row in rows:
            class_id = row["class_id"]
            if counts.get(class_id, 0) >= args.max_per_class:
                continue
            if len(manifest) >= args.max_total:
                break
            row_key = (class_id, row.get("observation_id", ""), row.get("photo_id", ""))
            if row_key in completed_keys:
                continue

            local_name = row.get("local_filename") or f"{class_id}_{row['observation_id']}_{row['photo_id']}.jpg"
            local_path = args.image_dir / local_name
            event = {
                "class_id": class_id,
                "species": row["fishnet_species"],
                "photo_id": row["photo_id"],
                "url": row["photo_url"],
                "local_path": str(local_path),
                "status": "started",
            }
            try:
                if args.skip_existing and local_path.exists() and local_path.stat().st_size > 0:
                    data = local_path.read_bytes()
                else:
                    data = fetch_bytes(row["photo_url"], timeout=args.timeout, retries=args.retries, pause=args.pause)
                    local_path.write_bytes(data)
                sha256 = hashlib.sha256(data).hexdigest()
                record = dict(row)
                record.update(
                    {
                        "local_path": str(local_path),
                        "bytes": str(len(data)),
                        "sha256": sha256,
                    }
                )
                manifest.append(record)
                counts[class_id] = counts.get(class_id, 0) + 1
                completed_keys.add(row_key)
                event.update({"status": "ok", "bytes": len(data), "sha256": sha256})
                successes_since_flush += 1
            except Exception as exc:
                event.update({"status": "error", "error": str(exc)})
            events.write(json.dumps(event, ensure_ascii=False) + "\n")
            events.flush()
            if args.flush_every > 0 and successes_since_flush >= args.flush_every:
                write_manifest(args.out / "download_manifest.csv", manifest)
                write_external_manifest(args.out / "external_manifest.csv", manifest, args.external_split)
                write_summary(args, counts, manifest, events_path, partial=True)
                successes_since_flush = 0
            time.sleep(args.pause)

    write_manifest(args.out / "download_manifest.csv", manifest)
    write_external_manifest(args.out / "external_manifest.csv", manifest, args.external_split)
    write_summary(args, counts, manifest, events_path, partial=False)
    summary = json.loads((args.out / "download_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
