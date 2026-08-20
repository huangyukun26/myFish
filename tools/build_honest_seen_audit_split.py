from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


BUCKETS = ("1", "2", "3-5", "6-10", "11-50", "51+")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def stable_int(text: str) -> int:
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path, cosine: np.ndarray) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
    coeff = cosine @ pixels @ cosine.T
    low = coeff[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    bits = 0
    for value in low:
        bits = (bits << 1) | int(value > threshold)
    return bits


def make_cosine(n: int = 32) -> np.ndarray:
    grid = np.arange(n, dtype=np.float32)
    out = np.cos(np.pi / n * (grid[:, None] + 0.5) * grid[None, :]).astype(np.float32)
    out[0] *= np.float32(1.0 / np.sqrt(n))
    out[1:] *= np.float32(np.sqrt(2.0 / n))
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_rows(bank_path: Path, feature_path: Path) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    image_ids = [str(x) for x in bank["image_ids"]]
    if image_ids != [str(x) for x in feature["image_ids"]]:
        raise RuntimeError("bank and feature cache image_ids are not aligned")
    labels = [str(x) for x in feature.get("labels", [])]
    if not labels:
        classes = [str(x) for x in feature["classes"]]
        labels = [classes[int(x)] for x in feature["class_ids"].tolist()]
    if len(labels) != len(image_ids):
        raise RuntimeError("label/image row count mismatch")
    old_dev = torch.as_tensor(bank["dev"], dtype=torch.bool)
    old_sealed = torch.as_tensor(bank["sealed"], dtype=torch.bool)
    return image_ids, labels, old_dev, old_sealed


def bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 50:
        return "11-50"
    return "51+"


def build_groups(image_ids: list[str], image_root: Path, phash_distance: int) -> tuple[list[int], dict[str, Any]]:
    n = len(image_ids)
    uf = UnionFind(n)
    exact: dict[str, list[int]] = defaultdict(list)
    hashes: list[str] = []
    phashes: list[int] = []
    cosine = make_cosine()
    missing = 0
    for i, image_id in enumerate(image_ids):
        path = image_root / Path(image_id).name
        if not path.exists():
            missing += 1
            hashes.append("")
            phashes.append(0)
            continue
        digest = file_hash(path)
        hashes.append(digest)
        exact[digest].append(i)
        phashes.append(phash(path, cosine))
    for members in exact.values():
        for other in members[1:]:
            uf.union(members[0], other)

    # Bucket by the high 16 pHash bits, then check nearby prefix variants.
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, value in enumerate(phashes):
        if hashes[i]:
            buckets[value >> 48].append(i)
    prefix_masks: list[int] = []
    for radius in range(phash_distance + 1):
        for combo in itertools.combinations(range(16), radius):
            mask = 0
            for bit in combo:
                mask |= 1 << (15 - bit)
            prefix_masks.append(mask)
    for i, value in enumerate(phashes):
        if not hashes[i]:
            continue
        prefix = value >> 48
        for mask in prefix_masks:
            for j in buckets.get(prefix ^ mask, ()):
                if j > i and hamming(value, phashes[j]) <= phash_distance:
                    uf.union(i, j)

    roots: dict[int, int] = {}
    group_ids: list[int] = []
    for i in range(n):
        root = uf.find(i)
        if root not in roots:
            roots[root] = len(roots)
        group_ids.append(roots[root])
    sizes = Counter(group_ids)
    summary = {
        "rows": n,
        "missing_images": missing,
        "exact_hash_groups": sum(1 for x in exact.values() if len(x) > 1),
        "exact_duplicate_rows": sum(len(x) - 1 for x in exact.values() if len(x) > 1),
        "phash_distance": phash_distance,
        "components": len(sizes),
        "singleton_components": sum(1 for x in sizes.values() if x == 1),
        "largest_component": max(sizes.values(), default=0),
    }
    return group_ids, {"summary": summary, "hashes": hashes, "phashes": phashes, "sizes": sizes}


def assign_seed(group_ids: list[int], labels: list[str], seed: int) -> list[str]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        groups[group_id].append(i)
    assignment: dict[int, str] = {
        group_id: ("dev" if stable_int(f"{seed}:{group_id}") % 2 == 0 else "sealed")
        for group_id in groups
    }
    class_groups: dict[str, set[int]] = defaultdict(set)
    for i, label in enumerate(labels):
        class_groups[label].add(group_ids[i])
    # Repair classes with >=2 disconnected components so both panels are used.
    # A component stays intact even if it contains conflicting labels.
    for _ in range(4):
        changed = False
        for label in sorted(class_groups):
            members = sorted(class_groups[label], key=lambda g: (len(groups[g]), stable_int(f"{seed}:{label}:{g}")))
            sides = {assignment[g] for g in members}
            if len(members) >= 2 and len(sides) == 1:
                assignment[members[0]] = "sealed" if assignment[members[0]] == "dev" else "dev"
                changed = True
        if not changed:
            break
    return [assignment[g] for g in group_ids]


def class_table(labels: list[str], image_ids: list[str], group_ids: list[int], old_dev: torch.Tensor, old_sealed: torch.Tensor, assignments: dict[int, list[str]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)
    rows: list[dict[str, Any]] = []
    for label in sorted(by_class):
        idxs = by_class[label]
        groups = sorted({group_ids[i] for i in idxs})
        row: dict[str, Any] = {
            "label": label,
            "genus": label.split()[0],
            "val_rows": len(idxs),
            "class_bucket": bucket(len(idxs)),
            "group_count": len(groups),
            "group_sizes": ";".join(str(sum(group_ids[i] == g for i in idxs)) for g in groups),
            "old_dev_rows": int(old_dev[idxs].sum().item()),
            "old_sealed_rows": int(old_sealed[idxs].sum().item()),
        }
        for name, values in assignments.items():
            row[f"{name}_dev_rows"] = sum(values[i] == "dev" for i in idxs)
            row[f"{name}_sealed_rows"] = sum(values[i] == "sealed" for i in idxs)
            row[f"{name}_both"] = int(row[f"{name}_dev_rows"] > 0 and row[f"{name}_sealed_rows"] > 0)
        rows.append(row)
    return rows


def composition_rows(class_rows: list[dict[str, Any]], labels: list[str], assignments: dict[int, list[str]], old_dev: torch.Tensor, old_sealed: torch.Tensor) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    parts: dict[str, list[str]] = {"old_dev": ["old_dev_rows"], "old_sealed": ["old_sealed_rows"]}
    for seed in assignments:
        parts[f"{seed}_dev"] = [f"{seed}_dev_rows"]
        parts[f"{seed}_sealed"] = [f"{seed}_sealed_rows"]
    for part, keys in parts.items():
        counts = Counter()
        classes = 0
        genera: set[str] = set()
        total = 0
        for row in class_rows:
            n = int(row[keys[0]])
            if n:
                classes += 1
                genera.add(str(row["genus"]))
                total += n
                counts[str(row["class_bucket"])] += 1
        record: dict[str, Any] = {"partition": part, "rows": total, "classes": classes, "genera": len(genera), "mean_rows_per_class": total / max(1, classes)}
        record.update({f"classes_bucket_{b}": counts[b] for b in BUCKETS})
        out.append(record)
    return out


def draw_composition_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1600, 900
    left, right, top, bottom = 140, 50, 100, 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((left, 28), "Old vs honest seen panel class-count composition", fill="black", font=title_font)
    plot_w, plot_h = width - left - right, height - top - bottom
    partitions = ["old_dev", "old_sealed", "42_dev", "42_sealed", "43_dev", "43_sealed"]
    colors = [(70, 110, 180), (200, 80, 70), (70, 150, 95), (220, 150, 45), (120, 90, 170), (80, 160, 175)]
    max_value = max((int(r.get(f"classes_bucket_{b}", 0)) for r in rows for b in BUCKETS), default=1)
    group_w = plot_w / len(BUCKETS)
    bar_w = group_w / (len(partitions) + 1)
    for bi, bucket_name in enumerate(BUCKETS):
        for pi, part in enumerate(partitions):
            row = next((r for r in rows if r["partition"] == part), None)
            value = int(row.get(f"classes_bucket_{bucket_name}", 0)) if row else 0
            x0 = left + bi * group_w + (pi + 0.35) * bar_w
            y1 = height - bottom
            y0 = y1 - int(plot_h * value / max_value)
            draw.rectangle((x0, y0, x0 + max(2, bar_w - 2), y1), fill=colors[pi])
        draw.text((left + bi * group_w + group_w * 0.28, height - bottom + 12), bucket_name, fill="black", font=font)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    legend_x = left
    legend_y = height - 62
    for pi, part in enumerate(partitions):
        x = legend_x + pi * 205
        draw.rectangle((x, legend_y, x + 18, legend_y + 18), fill=colors[pi])
        draw.text((x + 24, legend_y + 2), part, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-bank", type=Path, default=Path("runs/local_20260807_seen_candidate_bank_fusion/candidate_bank_scores_gateboost76.pt"))
    parser.add_argument("--val-feature-cache", type=Path, default=Path("work/seen_20260715_joint/val_bio_hflip_letterbox.pt"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/honest_seen"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_ids, labels, old_dev, old_sealed = read_rows(args.seen_bank, args.val_feature_cache)
    group_ids, group_meta = build_groups(image_ids, args.image_root, args.phash_distance)
    assignments = {str(seed): assign_seed(group_ids, labels, seed) for seed in args.seeds}
    class_rows = class_table(labels, image_ids, group_ids, old_dev, old_sealed, assignments)
    comp = composition_rows(class_rows, labels, assignments, old_dev, old_sealed)

    assignment_rows: list[dict[str, Any]] = []
    for i, image_id in enumerate(image_ids):
        row: dict[str, Any] = {"image_id": image_id, "label": labels[i], "genus": labels[i].split()[0], "group_id": group_ids[i], "group_size": group_meta["sizes"][group_ids[i]], "old_split": "dev" if old_dev[i] else "sealed"}
        for seed, values in assignments.items():
            row[f"seed_{seed}"] = values[i]
        assignment_rows.append(row)
    with (args.out_dir / "assignment.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = list(assignment_rows[0])
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignment_rows)
    with (args.out_dir / "class_rows.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = list(class_rows[0])
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(class_rows)
    with (args.out_dir / "composition.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = list(comp[0])
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comp)
    draw_composition_plot(args.out_dir / "old_new_composition.png", comp)
    summary = {"seeds": args.seeds, "rows": len(image_ids), "classes": len(set(labels)), "grouping": group_meta["summary"], "composition": comp, "note": "Group assignment uses exact BLAKE2 content groups plus pHash Hamming-distance components; components never cross panels."}
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
