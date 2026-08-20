#!/usr/bin/env python
"""Quarantine exact or near-duplicate external images against official images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(path: Path, size: int) -> int:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        vals = list(im.getdata())
    mean = sum(vals) / len(vals)
    bits = 0
    for idx, val in enumerate(vals):
        if val >= mean:
            bits |= 1 << idx
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def official_path(root: Path, row: dict[str, str]) -> Path:
    member = row.get("zip_member") or row.get("image_id")
    if member.startswith("images/"):
        member = member[len("images/") :]
    return root / member


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--official-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--official-image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument("--official-hash-cache", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    official_rows: list[dict[str, str]] = []
    official_hashes: dict[str, str] = {}
    official_ahashes: list[tuple[str, int]] = []
    skipped_official = 0
    if args.official_hash_cache and args.official_hash_cache.exists():
        cache = json.loads(args.official_hash_cache.read_text(encoding="utf-8"))
        if int(cache.get("hash_size", -1)) != args.hash_size:
            raise RuntimeError(f"Hash cache was built with hash_size={cache.get('hash_size')}, expected {args.hash_size}")
        for row in cache["rows"]:
            official_hashes[row["sha256"]] = row["image_id"]
            official_ahashes.append((row["image_id"], int(row["ahash"])))
        skipped_official = int(cache.get("skipped_official", 0))
        official_rows = [{} for _ in range(int(cache.get("official_rows", len(official_ahashes))))]
    else:
        for manifest in args.official_manifests:
            official_rows.extend(read_csv(manifest))
        cache_rows = []
        for row in official_rows:
            image_id = row.get("image_id") or Path(row.get("zip_member", "")).name
            path = official_path(args.official_image_root, row)
            if not path.exists():
                skipped_official += 1
                continue
            try:
                sha = sha256_file(path)
                img_hash = ahash(path, args.hash_size)
                official_hashes[sha] = image_id
                official_ahashes.append((image_id, img_hash))
                cache_rows.append({"image_id": image_id, "sha256": sha, "ahash": str(img_hash)})
            except Exception:
                skipped_official += 1
        if args.official_hash_cache:
            args.official_hash_cache.parent.mkdir(parents=True, exist_ok=True)
            args.official_hash_cache.write_text(
                json.dumps(
                    {
                        "hash_size": args.hash_size,
                        "official_rows": len(official_rows),
                        "official_hashed": len(official_ahashes),
                        "skipped_official": skipped_official,
                        "rows": cache_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    clean_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    for row in read_csv(args.external_manifest):
        local_path = Path(row.get("local_path") or row.get("image_id") or "")
        if not local_path.exists():
            record = dict(row)
            record.update({"quarantine_reason": "missing_external_file"})
            quarantine_rows.append(record)
            continue
        try:
            sha = sha256_file(local_path)
            if sha in official_hashes:
                record = dict(row)
                record.update({"quarantine_reason": "exact_sha256", "nearest_official_image_id": official_hashes[sha], "nearest_hamming": 0})
                quarantine_rows.append(record)
                continue
            ext_hash = ahash(local_path, args.hash_size)
            nearest_id = ""
            nearest_dist = args.hash_size * args.hash_size + 1
            for image_id, off_hash in official_ahashes:
                dist = hamming(ext_hash, off_hash)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = image_id
            if nearest_dist <= args.max_hamming:
                record = dict(row)
                record.update({"quarantine_reason": "near_ahash", "nearest_official_image_id": nearest_id, "nearest_hamming": nearest_dist})
                quarantine_rows.append(record)
                continue
            clean_rows.append(row)
        except Exception as exc:
            record = dict(row)
            record.update({"quarantine_reason": f"error:{exc}"})
            quarantine_rows.append(record)

    manifest_fields = []
    for row in clean_rows + quarantine_rows:
        for key in row:
            if key not in manifest_fields:
                manifest_fields.append(key)
    write_csv(args.out / "download_manifest_clean.csv", clean_rows, manifest_fields)
    write_csv(args.out / "download_manifest_quarantine.csv", quarantine_rows, manifest_fields)

    ext_fields = ["image_id", "zip_member", "label", "class_id", "split"]
    external_rows = []
    for row in clean_rows:
        local_name = Path(row["local_path"]).name
        external_rows.append(
            {
                "image_id": local_name,
                "zip_member": local_name,
                "label": row["fishnet_species"],
                "class_id": row["class_id"],
                "split": "external_inat_clean",
            }
        )
    write_csv(args.out / "external_manifest_clean.csv", external_rows, ext_fields)

    summary = {
        "external_manifest": str(args.external_manifest),
        "official_rows": len(official_rows),
        "official_hashed": len(official_ahashes),
        "skipped_official": skipped_official,
        "external_rows": len(clean_rows) + len(quarantine_rows),
        "clean_rows": len(clean_rows),
        "quarantine_rows": len(quarantine_rows),
        "hash_size": args.hash_size,
        "max_hamming": args.max_hamming,
        "outputs": {
            "download_manifest_clean": str(args.out / "download_manifest_clean.csv"),
            "download_manifest_quarantine": str(args.out / "download_manifest_quarantine.csv"),
            "external_manifest_clean": str(args.out / "external_manifest_clean.csv"),
        },
    }
    (args.out / "quarantine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
