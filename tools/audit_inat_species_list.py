#!/usr/bin/env python
"""Audit iNaturalist coverage for an arbitrary species list."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_inat_external_coverage import (  # noqa: E402
    DEFAULT_LICENSES,
    query_observations,
    query_taxon,
    read_existing_csv,
    slugify,
    write_report,
)


def load_species(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return [str(x) for x in data]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-json", type=Path, required=True)
    parser.add_argument("--all-classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-classes", type=int, default=500)
    parser.add_argument("--selection", choices=["ordered", "random"], default="ordered")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--per-species", type=int, default=8)
    parser.add_argument("--created-after", default="2020-01-01")
    parser.add_argument("--licenses", default=",".join(DEFAULT_LICENSES))
    parser.add_argument("--quality-grade", default="research")
    parser.add_argument("--photo-size", default="large", choices=["small", "medium", "large", "original"])
    parser.add_argument("--skip-taxon-lookup", action="store_true")
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_classes = load_species(args.all_classes_json)
    class_id_by_species = {name: idx for idx, name in enumerate(all_classes)}
    targets = [s for s in load_species(args.species_json) if s in class_id_by_species]
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(targets)
    targets = targets[: args.max_classes]
    target_rows: list[dict[str, Any]] = [
        {
            "class_id": class_id_by_species[species],
            "rows": 0,
            "train": 0,
            "val": 0,
            "species": species,
        }
        for species in targets
    ]

    metadata_rows = read_existing_csv(args.out / "inat_photo_metadata.csv") if args.resume else []
    summary_rows = read_existing_csv(args.out / "inat_coverage_summary.csv") if args.resume else []
    done_class_ids = {int(r["class_id"]) for r in summary_rows if str(r.get("class_id", "")).strip()}
    events_path = args.out / "events.jsonl"
    event_mode = "a" if args.resume else "w"
    processed_since_flush = 0
    args.max_train_rows = -1

    with events_path.open(event_mode, encoding="utf-8") as events:
        for idx, stat in enumerate(target_rows, start=1):
            class_id = int(stat["class_id"])
            if class_id in done_class_ids:
                continue
            species = str(stat["species"])
            event: dict[str, Any] = {
                "index": idx,
                "class_id": class_id,
                "species": species,
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
                            "fishnet_rows": 0,
                            "fishnet_train_rows": 0,
                            "fishnet_val_rows": 0,
                            "local_filename": f"{class_id:05d}_{slugify(species)}_{row['observation_id']}_{row['photo_id']}.jpg",
                        }
                    )
                metadata_rows.extend(obs_rows)
                summary = {
                    "class_id": class_id,
                    "species": species,
                    "fishnet_rows": 0,
                    "fishnet_train_rows": 0,
                    "fishnet_val_rows": 0,
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
                        "fishnet_rows": 0,
                        "fishnet_train_rows": 0,
                        "fishnet_val_rows": 0,
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
                    targets=target_rows,
                    summary_rows=summary_rows,
                    metadata_rows=metadata_rows,
                    events_path=events_path,
                    partial=True,
                )
                processed_since_flush = 0
            time.sleep(args.pause)

    write_report(
        args=args,
        targets=target_rows,
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
