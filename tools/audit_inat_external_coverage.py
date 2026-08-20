#!/usr/bin/env python
"""Audit iNaturalist public image coverage for low-shot FishNet classes.

This script only queries metadata. It does not download images and does not
touch test predictions. It is intended as the first gate for external public
data experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


INAT_API = "https://api.inaturalist.org/v1"
DEFAULT_LICENSES = ["cc0", "cc-by", "cc-by-nc"]


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_")


def read_class_stats(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        r["class_id"] = int(r["class_id"])
        r["rows"] = int(r["rows"])
        r["train"] = int(r["train"])
        r["val"] = int(r["val"])
    return rows


def read_class_labels(manifest_paths: list[Path]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for path in manifest_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                labels[int(row["class_id"])] = row["label"]
    return labels


SUMMARY_FIELDS = [
    "class_id",
    "species",
    "fishnet_rows",
    "fishnet_train_rows",
    "fishnet_val_rows",
    "inat_taxon_id",
    "inat_taxon_name",
    "inat_total_results",
    "usable_photos",
    "error",
]

META_FIELDS = [
    "class_id",
    "fishnet_species",
    "fishnet_rows",
    "fishnet_train_rows",
    "fishnet_val_rows",
    "source",
    "taxon_id",
    "taxon_name",
    "preferred_common_name",
    "observation_id",
    "observed_on",
    "created_at",
    "quality_grade",
    "uri",
    "photo_id",
    "photo_url",
    "photo_license",
    "photo_attribution",
    "photo_width",
    "photo_height",
    "local_filename",
]


def read_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_report(
    *,
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    events_path: Path,
    partial: bool,
) -> None:
    write_csv(args.out / "inat_coverage_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.out / "inat_photo_metadata.csv", metadata_rows, META_FIELDS)
    report = {
        "targets": len(targets),
        "processed_classes": len({int(r["class_id"]) for r in summary_rows if str(r.get("class_id", "")).strip()}),
        "classes_with_any_photo": sum(1 for r in summary_rows if int(r.get("usable_photos") or 0) > 0),
        "total_usable_photos": len(metadata_rows),
        "per_species": args.per_species,
        "max_train_rows": args.max_train_rows,
        "max_classes": args.max_classes,
        "selection": args.selection,
        "seed": args.seed,
        "created_after": args.created_after,
        "licenses": args.licenses,
        "quality_grade": args.quality_grade,
        "partial": partial,
        "outputs": {
            "summary_csv": str(args.out / "inat_coverage_summary.csv"),
            "metadata_csv": str(args.out / "inat_photo_metadata.csv"),
            "events_jsonl": str(events_path),
        },
    }
    with (args.out / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def http_get_json(url: str, *, timeout: float, retries: int, pause: float) -> dict[str, Any]:
    headers = {
        "User-Agent": "fishnet-external-data-audit/1.0 (metadata coverage audit)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries + 1} attempts: {url} :: {last_error}")


def query_taxon(species: str, args: argparse.Namespace) -> dict[str, Any] | None:
    params = {
        "q": species,
        "rank": "species",
        "is_active": "true",
        "per_page": "5",
    }
    url = f"{INAT_API}/taxa?{urllib.parse.urlencode(params)}"
    data = http_get_json(url, timeout=args.timeout, retries=args.retries, pause=args.pause)
    results = data.get("results", [])
    if not results:
        return None

    species_lower = species.lower()
    exact = []
    for item in results:
        names = [str(item.get("name", ""))]
        names.extend(str(n) for n in item.get("matched_term", "").split("|") if n)
        if any(n.lower() == species_lower for n in names):
            exact.append(item)
    return exact[0] if exact else results[0]


def best_photo(obs: dict[str, Any], licenses: set[str]) -> dict[str, Any] | None:
    for photo in obs.get("photos", []) or []:
        code = str(photo.get("license_code") or "").lower()
        if code not in licenses:
            continue
        url = photo.get("url") or ""
        if not url:
            continue
        return photo
    return None


def normalize_photo_url(url: str, size: str) -> str:
    for token in ("square", "small", "medium", "large", "original"):
        url = url.replace(f"/{token}.", f"/{size}.")
    return url


def query_observations(species: str, taxon_id: int | None, args: argparse.Namespace) -> tuple[int | None, list[dict[str, Any]]]:
    licenses = [x.strip().lower() for x in args.licenses.split(",") if x.strip()]
    params: dict[str, str] = {
        "photos": "true",
        "quality_grade": args.quality_grade,
        "per_page": str(args.per_species),
        "page": "1",
        "order_by": "created_at",
        "order": "desc",
        "photo_license": ",".join(licenses),
    }
    if taxon_id is not None:
        params["taxon_id"] = str(taxon_id)
    else:
        params["taxon_name"] = species
    if args.created_after:
        params["created_d1"] = args.created_after

    url = f"{INAT_API}/observations?{urllib.parse.urlencode(params)}"
    data = http_get_json(url, timeout=args.timeout, retries=args.retries, pause=args.pause)
    total = data.get("total_results")
    rows: list[dict[str, Any]] = []
    allowed = set(licenses)

    for obs in data.get("results", []) or []:
        photo = best_photo(obs, allowed)
        if photo is None:
            continue
        taxon = obs.get("taxon") or {}
        rows.append(
            {
                "source": "inaturalist",
                "species_query": species,
                "taxon_id": taxon.get("id") or taxon_id,
                "taxon_name": taxon.get("name"),
                "preferred_common_name": taxon.get("preferred_common_name"),
                "observation_id": obs.get("id"),
                "observed_on": obs.get("observed_on"),
                "created_at": obs.get("created_at"),
                "quality_grade": obs.get("quality_grade"),
                "uri": obs.get("uri"),
                "photo_id": photo.get("id"),
                "photo_url": normalize_photo_url(photo.get("url"), args.photo_size),
                "photo_license": str(photo.get("license_code") or "").lower(),
                "photo_attribution": photo.get("attribution"),
                "photo_width": (photo.get("original_dimensions") or {}).get("width"),
                "photo_height": (photo.get("original_dimensions") or {}).get("height"),
            }
        )
    return int(total) if isinstance(total, int) else None, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-stats", type=Path, default=Path("work/seen_image_distribution_split_seed2027_frac20/class_stats.json"))
    parser.add_argument("--train-manifest", type=Path, default=Path("work/seen_image_distribution_split_seed2027_frac20/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("work/seen_image_distribution_split_seed2027_frac20/val.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=2)
    parser.add_argument("--max-classes", type=int, default=100)
    parser.add_argument("--selection", choices=["ordered", "random"], default="ordered")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--per-species", type=int, default=10)
    parser.add_argument("--created-after", default="2023-01-01")
    parser.add_argument("--licenses", default=",".join(DEFAULT_LICENSES))
    parser.add_argument("--quality-grade", default="research")
    parser.add_argument("--photo-size", default="large", choices=["small", "medium", "large", "original"])
    parser.add_argument("--skip-taxon-lookup", action="store_true")
    parser.add_argument("--pause", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Append to existing events and skip classes already present in summary CSV.")
    parser.add_argument("--flush-every", type=int, default=5, help="Write partial CSV/summary after this many processed classes.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stats = read_class_stats(args.class_stats)
    labels = read_class_labels([args.train_manifest, args.val_manifest])
    targets = [
        r for r in stats
        if r["class_id"] in labels and r["rows"] <= args.max_train_rows
    ]
    targets.sort(key=lambda r: (r["rows"], r["val"], r["class_id"]))
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(targets)
    targets = targets[: args.max_classes]

    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    events_path = args.out / "events.jsonl"
    if args.resume:
        summary_rows = read_existing_csv(args.out / "inat_coverage_summary.csv")
        metadata_rows = read_existing_csv(args.out / "inat_photo_metadata.csv")
    done_class_ids = {int(r["class_id"]) for r in summary_rows if str(r.get("class_id", "")).strip()}

    event_mode = "a" if args.resume else "w"
    processed_since_flush = 0
    with events_path.open(event_mode, encoding="utf-8") as events:
        for idx, stat in enumerate(targets, start=1):
            class_id = stat["class_id"]
            if class_id in done_class_ids:
                continue
            species = labels[class_id]
            event: dict[str, Any] = {
                "index": idx,
                "class_id": class_id,
                "species": species,
                "train_rows": stat["train"],
                "val_rows": stat["val"],
                "status": "started",
            }
            try:
                taxon = None if args.skip_taxon_lookup else query_taxon(species, args)
                taxon_id = int(taxon["id"]) if taxon and taxon.get("id") is not None else None
                total, obs_rows = query_observations(species, taxon_id, args)
                for row in obs_rows:
                    row.update(
                        {
                            "class_id": class_id,
                            "fishnet_species": species,
                            "fishnet_rows": stat["rows"],
                            "fishnet_train_rows": stat["train"],
                            "fishnet_val_rows": stat["val"],
                            "local_filename": f"{class_id:05d}_{slugify(species)}_{row['observation_id']}_{row['photo_id']}.jpg",
                        }
                    )
                metadata_rows.extend(obs_rows)
                summary = {
                    "class_id": class_id,
                    "species": species,
                    "fishnet_rows": stat["rows"],
                    "fishnet_train_rows": stat["train"],
                    "fishnet_val_rows": stat["val"],
                    "inat_taxon_id": taxon_id,
                    "inat_taxon_name": (taxon or {}).get("name") if taxon else None,
                    "inat_total_results": total,
                    "usable_photos": len(obs_rows),
                }
                summary_rows.append(summary)
                event.update({"status": "ok", **summary})
            except Exception as exc:
                summary_rows.append(
                    {
                        "class_id": class_id,
                        "species": species,
                        "fishnet_rows": stat["rows"],
                        "fishnet_train_rows": stat["train"],
                        "fishnet_val_rows": stat["val"],
                        "inat_taxon_id": None,
                        "inat_taxon_name": None,
                        "inat_total_results": None,
                        "usable_photos": 0,
                        "error": str(exc),
                    }
                )
                event.update({"status": "error", "error": str(exc)})
            events.write(json.dumps(event, ensure_ascii=False) + "\n")
            events.flush()
            processed_since_flush += 1
            if args.flush_every > 0 and processed_since_flush >= args.flush_every:
                write_report(
                    args=args,
                    targets=targets,
                    summary_rows=summary_rows,
                    metadata_rows=metadata_rows,
                    events_path=events_path,
                    partial=True,
                )
                processed_since_flush = 0
            time.sleep(args.pause)

    write_report(
        args=args,
        targets=targets,
        summary_rows=summary_rows,
        metadata_rows=metadata_rows,
        events_path=events_path,
        partial=False,
    )
    report = json.loads((args.out / "audit_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
