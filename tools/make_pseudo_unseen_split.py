from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def genus_of(class_name: str) -> str:
    return class_name.split()[0] if class_name.split() else class_name


def load_seen_class_order(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def pick_species_holdout(classes: List[str], count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    picked = list(classes)
    rng.shuffle(picked)
    return sorted(picked[:count])


def pick_genus_holdout(classes: List[str], count: int, seed: int) -> List[str]:
    by_genus: Dict[str, List[str]] = defaultdict(list)
    for class_name in classes:
        by_genus[genus_of(class_name)].append(class_name)
    genera = list(by_genus)
    random.Random(seed).shuffle(genera)
    picked: List[str] = []
    for genus in genera:
        picked.extend(by_genus[genus])
        if len(picked) >= count:
            break
    return sorted(picked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", type=Path, default=Path("work/supervised_splits/val.csv"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["species", "genus"], default="species")
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_classes = load_seen_class_order(args.class_map)
    if args.classes <= 0 or args.classes > len(all_classes):
        raise ValueError(f"--classes must be in 1..{len(all_classes)}")

    if args.mode == "species":
        selected_classes = pick_species_holdout(all_classes, args.classes, args.seed)
    else:
        selected_classes = pick_genus_holdout(all_classes, args.classes, args.seed)

    selected = set(selected_classes)
    rows = [row for row in read_rows(args.val_manifest) if row.get("label") in selected]
    if not rows:
        raise RuntimeError("No validation rows matched the selected classes")

    class_to_idx = {name: idx for idx, name in enumerate(selected_classes)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "manifest.csv", rows)
    (args.out_dir / "classes.json").write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")

    genera = sorted({genus_of(name) for name in selected_classes})
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "requested_classes": args.classes,
        "selected_classes": len(selected_classes),
        "manifest_rows": len(rows),
        "genera": len(genera),
        "val_manifest": str(args.val_manifest),
        "class_map": str(args.class_map),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
