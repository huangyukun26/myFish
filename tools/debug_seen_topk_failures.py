#!/usr/bin/env python
"""Diagnose seen validation topK failures and external-gallery conflicts."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def load_cache(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    obj["features"] = F.normalize(obj["features"].float(), dim=1)
    obj["image_ids"] = list(obj["image_ids"])
    if "class_ids" in obj:
        obj["class_ids"] = torch.as_tensor(obj["class_ids"]).long()
    return obj


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_topk(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["image_id"]] = row
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            preds = json.loads(row["top_classes"]) if row.get("top_classes", "").startswith("[") else []
            scores = json.loads(row["top_scores"]) if row.get("top_scores", "").startswith("[") else []
            rows[row["image_id"]] = {
                "image_id": row["image_id"],
                "label": row.get("true_label", ""),
                "predictions": preds,
                "scores": scores,
            }
    return rows


def genus(label: str) -> str:
    return (label or "").split()[0] if label else ""


def find_rank(preds: list[str], label: str) -> int:
    try:
        return preds.index(label) + 1
    except ValueError:
        return 999


def score_external(
    query_cache: dict[str, Any],
    external_cache: dict[str, Any],
    class_names: list[str],
    *,
    max_rows: int = 0,
) -> dict[str, dict[str, Any]]:
    if not external_cache or "features" not in external_cache:
        return {}
    ext_class_ids = torch.as_tensor(external_cache["class_ids"]).long()
    valid_cids = sorted(int(x) for x in torch.unique(ext_class_ids).tolist() if int(x) >= 0)
    if not valid_cids:
        return {}
    qfeat = query_cache["features"]
    qids = query_cache["image_ids"]
    if max_rows:
        qfeat = qfeat[:max_rows]
        qids = qids[:max_rows]
    sims = qfeat @ external_cache["features"].T
    out: dict[str, dict[str, Any]] = {}
    for row_idx, image_id in enumerate(qids):
        best_label = ""
        best_cid = -1
        best_score = float("-inf")
        second_score = float("-inf")
        for cid in valid_cids:
            cols = torch.nonzero(ext_class_ids == cid, as_tuple=False).flatten()
            if cols.numel() == 0:
                continue
            score = float(sims[row_idx, cols].max().item())
            if score > best_score:
                second_score = best_score
                best_score = score
                best_cid = cid
                best_label = class_names[cid] if 0 <= cid < len(class_names) else ""
            elif score > second_score:
                second_score = score
        out[image_id] = {
            "external_label": best_label,
            "external_class_id": best_cid,
            "external_score": best_score,
            "external_second_score": second_score,
            "external_gap": best_score - second_score if second_score != float("-inf") else 0.0,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_sheet(path: Path, rows: list[dict[str, Any]], manifest_by_id: dict[str, dict[str, str]], images_zip: Path, *, count: int, cols: int) -> None:
    sample = rows[:count]
    tiles = []
    with zipfile.ZipFile(images_zip) as zf:
        for row in sample:
            manifest = manifest_by_id.get(row["image_id"])
            if not manifest:
                continue
            with zf.open(manifest["zip_member"]) as fp:
                image = Image.open(BytesIO(fp.read())).convert("RGB")
            image.thumbnail((220, 165), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (260, 245), (246, 246, 246))
            tile.paste(image, ((260 - image.width) // 2, 8))
            draw = ImageDraw.Draw(tile)
            lines = [
                row["image_id"],
                f"T: {row['true_label'][:30]}",
                f"P: {row['top1'][:30]}",
                f"E: {row.get('external_label', '')[:30]}",
            ]
            for i, text in enumerate(lines):
                draw.text((8, 176 + i * 16), text, fill=(20, 20, 20))
            tiles.append(tile)
    if not tiles:
        return
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 260, rows_n * 245), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 260, (i // cols) * 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-manifest", type=Path, default=Path("work/seen_image_distribution_split_seed2027_frac20/val.csv"))
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, default=Path("work/cloud_20260713/artifacts/bioclip25_letterbox_full/val_letterbox.pt"))
    parser.add_argument("--external-cache", type=Path, default=None)
    parser.add_argument("--classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--images-zip", type=Path, default=Path("dataset/images.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sheet-count", type=int, default=48)
    parser.add_argument("--sheet-cols", type=int, default=6)
    args = parser.parse_args()

    val_rows = read_manifest(args.val_manifest)
    manifest_by_id = {row["image_id"]: row for row in val_rows}
    true_by_id = {row["image_id"]: row["label"] for row in val_rows}
    class_names = json.loads(args.classes_json.read_text(encoding="utf-8"))
    topk = load_topk(args.topk)
    query_cache = load_cache(args.query_cache)
    external_by_id: dict[str, dict[str, Any]] = {}
    if args.external_cache:
        external_by_id = score_external(query_cache, load_cache(args.external_cache), class_names)

    rows = []
    for image_id, true_label in true_by_id.items():
        row = topk.get(image_id)
        if not row:
            continue
        preds = list(row.get("predictions") or [])
        scores = [float(x) for x in row.get("scores") or []]
        top1 = preds[0] if preds else ""
        rank = find_rank(preds, true_label)
        margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
        item = {
            "image_id": image_id,
            "true_label": true_label,
            "top1": top1,
            "correct_top1": top1 == true_label,
            "true_rank": rank,
            "true_in_top5": rank <= 5,
            "true_in_top20": rank <= 20,
            "top1_same_genus": genus(top1) == genus(true_label),
            "margin_top1_top2": margin,
        }
        item.update(external_by_id.get(image_id, {}))
        if "external_label" in item:
            item["external_true_same_genus"] = genus(str(item["external_label"])) == genus(true_label)
            item["external_top1_same_genus"] = genus(str(item["external_label"])) == genus(top1)
        rows.append(item)

    wrong = [r for r in rows if not r["correct_top1"]]
    wrong.sort(key=lambda r: (r["true_rank"], -float(r["margin_top1_top2"])))
    external_conflicts = [
        r for r in rows
        if r.get("external_label") and r.get("external_label") != r["top1"]
    ]
    external_conflicts.sort(key=lambda r: (str(r.get("external_true_same_genus")) != "True", -float(r.get("external_score", 0.0))))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "all_rows.csv", rows)
    write_csv(args.out / "wrong_top1_rows.csv", wrong)
    write_csv(args.out / "external_conflict_rows.csv", external_conflicts)
    render_sheet(args.out / "wrong_top1_sheet.jpg", wrong, manifest_by_id, args.images_zip, count=args.sheet_count, cols=args.sheet_cols)
    render_sheet(args.out / "external_conflict_sheet.jpg", external_conflicts, manifest_by_id, args.images_zip, count=args.sheet_count, cols=args.sheet_cols)

    summary = {
        "rows": len(rows),
        "top1_correct": sum(1 for r in rows if r["correct_top1"]),
        "top1_accuracy": sum(1 for r in rows if r["correct_top1"]) / max(1, len(rows)),
        "wrong_top1": len(wrong),
        "wrong_true_in_top5": sum(1 for r in wrong if r["true_in_top5"]),
        "wrong_true_in_top20": sum(1 for r in wrong if r["true_in_top20"]),
        "wrong_same_genus_top1": sum(1 for r in wrong if r["top1_same_genus"]),
        "external_conflicts": len(external_conflicts),
        "external_conflict_same_genus_true": sum(1 for r in external_conflicts if r.get("external_true_same_genus")),
        "outputs": {
            "all_rows": str(args.out / "all_rows.csv"),
            "wrong_top1_rows": str(args.out / "wrong_top1_rows.csv"),
            "external_conflict_rows": str(args.out / "external_conflict_rows.csv"),
            "wrong_top1_sheet": str(args.out / "wrong_top1_sheet.jpg"),
            "external_conflict_sheet": str(args.out / "external_conflict_sheet.jpg"),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
