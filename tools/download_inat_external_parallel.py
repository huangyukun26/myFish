#!/usr/bin/env python
"""Download audited iNaturalist images with bounded concurrency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_external_manifest(path: Path, rows: list[dict[str, str]], split: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "zip_member", "label", "class_id", "split"])
        writer.writeheader()
        for row in rows:
            name = Path(row["local_path"]).name
            writer.writerow(
                {
                    "image_id": name,
                    "zip_member": name,
                    "label": row["fishnet_species"],
                    "class_id": row["class_id"],
                    "split": split,
                }
            )
    tmp.replace(path)


def fetch(url: str, timeout: float, retries: int, pause: float) -> bytes:
    headers = {
        "User-Agent": "fishnet-external-data-parallel/1.0 (public image download)",
        "Accept": "image/*,*/*;q=0.8",
    }
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt >= retries:
                break
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"{last}")


def worker(row: dict[str, str], image_dir: Path, timeout: float, retries: int, pause: float, skip_existing: bool) -> dict[str, str]:
    local_name = row.get("local_filename") or f"{row['class_id']}_{row['observation_id']}_{row['photo_id']}.jpg"
    local_path = image_dir / local_name
    if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
        data = local_path.read_bytes()
    else:
        data = fetch(row["photo_url"], timeout, retries, pause)
        local_path.write_bytes(data)
    out = dict(row)
    out.update(
        {
            "local_path": str(local_path),
            "bytes": str(len(data)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--max-total", type=int, default=930)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--external-split", default="external_inat_parallel")
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = read_csv(args.metadata_csv)
    metadata.sort(key=lambda r: (int(r["class_id"]), r["fishnet_species"], r["observation_id"], r["photo_id"]))
    manifest_path = args.out / "download_manifest.csv"
    manifest = read_csv(manifest_path) if args.resume else []
    completed = {(r["class_id"], r.get("observation_id", ""), r.get("photo_id", "")) for r in manifest}
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row["class_id"]] = counts.get(row["class_id"], 0) + 1

    tasks: list[dict[str, str]] = []
    planned_counts = dict(counts)
    for row in metadata:
        if len(manifest) + len(tasks) >= args.max_total:
            break
        key = (row["class_id"], row.get("observation_id", ""), row.get("photo_id", ""))
        if key in completed:
            continue
        class_id = row["class_id"]
        if planned_counts.get(class_id, 0) >= args.max_per_class:
            continue
        tasks.append(row)
        planned_counts[class_id] = planned_counts.get(class_id, 0) + 1

    lock = threading.Lock()
    events_path = args.out / "download_events.jsonl"
    event_mode = "a" if args.resume else "w"
    ok_since_flush = 0
    with events_path.open(event_mode, encoding="utf-8") as events:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(worker, row, args.image_dir, args.timeout, args.retries, args.pause, args.skip_existing): row
                for row in tasks
            }
            for future in as_completed(futures):
                row = futures[future]
                event = {
                    "class_id": row["class_id"],
                    "species": row["fishnet_species"],
                    "photo_id": row["photo_id"],
                    "url": row["photo_url"],
                }
                try:
                    record = future.result()
                    event.update({"status": "ok", "bytes": int(record["bytes"]), "sha256": record["sha256"]})
                    with lock:
                        manifest.append(record)
                        ok_since_flush += 1
                except Exception as exc:
                    event.update({"status": "error", "error": str(exc)})
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                if ok_since_flush >= args.flush_every:
                    write_csv(manifest_path, manifest)
                    write_external_manifest(args.out / "external_manifest.csv", manifest, args.external_split)
                    ok_since_flush = 0

    write_csv(manifest_path, manifest)
    write_external_manifest(args.out / "external_manifest.csv", manifest, args.external_split)
    summary = {
        "metadata_csv": str(args.metadata_csv),
        "image_dir": str(args.image_dir),
        "downloaded": len(manifest),
        "classes": len({row["class_id"] for row in manifest}),
        "max_per_class": args.max_per_class,
        "max_total": args.max_total,
        "workers": args.workers,
        "events_jsonl": str(events_path),
        "manifest_csv": str(manifest_path),
        "external_manifest_csv": str(args.out / "external_manifest.csv"),
    }
    (args.out / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
