from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_predictions(path: Path) -> dict[str, str]:
    return {row["image_id"]: row["prediction"] for row in read_rows(path)}


def read_margins(path: Path) -> dict[str, float]:
    out = {}
    for row in read_rows(path):
        try:
            out[row["image_id"]] = float(row.get("margin_top1_top2", 0.0))
        except ValueError:
            out[row["image_id"]] = 0.0
    return out


def parse_candidates(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=path, got {value}")
        name, path = value.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def bucket_freq(count: int) -> str:
    if count <= 2:
        return "freq_000_002"
    if count <= 5:
        return "freq_003_005"
    if count <= 10:
        return "freq_006_010"
    if count <= 25:
        return "freq_011_025"
    if count <= 50:
        return "freq_026_050"
    return "freq_051_plus"


def bucket_aspect(width: int, height: int) -> list[tuple[str, str]]:
    wh = width / max(1, height)
    long_ratio = max(wh, 1.0 / max(wh, 1e-9))
    buckets = []
    if wh >= 2.25:
        buckets.append(("aspect", "wide_ge_2p25"))
    elif wh >= 1.75:
        buckets.append(("aspect", "wide_1p75_2p25"))
    elif wh >= 1.35:
        buckets.append(("aspect", "wide_1p35_1p75"))
    elif wh <= 0.55:
        buckets.append(("aspect", "tall_le_0p55"))
    elif wh <= 0.75:
        buckets.append(("aspect", "tall_0p55_0p75"))
    elif 0.85 <= wh <= 1.20:
        buckets.append(("aspect", "squareish_0p85_1p20"))
    else:
        buckets.append(("aspect", "mid_aspect"))
    if long_ratio >= 2.25:
        buckets.append(("long_ratio", "long_ge_2p25"))
    elif long_ratio >= 1.75:
        buckets.append(("long_ratio", "long_1p75_2p25"))
    elif long_ratio >= 1.35:
        buckets.append(("long_ratio", "long_1p35_1p75"))
    else:
        buckets.append(("long_ratio", "long_lt_1p35"))
    return buckets


def bucket_margin(margin: float) -> str:
    if margin <= 0.05:
        return "margin_le_0p05"
    if margin <= 0.20:
        return "margin_0p05_0p20"
    if margin <= 0.50:
        return "margin_0p20_0p50"
    if margin <= 1.00:
        return "margin_0p50_1p00"
    if margin <= 2.00:
        return "margin_1p00_2p00"
    return "margin_gt_2p00"


def load_feature_clusters(cache_path: Path, image_ids: list[str], k: int, seed: int) -> dict[str, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_ids = list(payload["image_ids"])
    id_to_idx = {image_id: idx for idx, image_id in enumerate(cache_ids)}
    missing = [image_id for image_id in image_ids if image_id not in id_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} image ids missing from feature cache; first={missing[:5]}")
    x = payload["features"][torch.tensor([id_to_idx[image_id] for image_id in image_ids])].float().numpy()
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    n_components = min(50, x.shape[0], x.shape[1])
    x_reduced = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto").fit_predict(x_reduced)
    return {image_id: int(label) for image_id, label in zip(image_ids, labels)}


def safe_top1(preds: dict[str, str], labels: dict[str, str], ids: Iterable[str]) -> float:
    ids_list = list(ids)
    if not ids_list:
        return 0.0
    return sum(preds.get(image_id) == labels[image_id] for image_id in ids_list) / len(ids_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-stats", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--base-topk-csv", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=predictions.csv")
    parser.add_argument("--cluster-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    image_ids = [row["image_id"] for row in rows]
    labels = {row["image_id"]: row["label"] for row in rows}
    class_ids = {row["image_id"]: row.get("class_id", "") for row in rows}
    class_stats = {str(row["class_id"]): row for row in json.loads(args.class_stats.read_text(encoding="utf-8"))}
    base = read_predictions(args.base_csv)
    margins = read_margins(args.base_topk_csv)
    candidates = {name: read_predictions(path) for name, path in parse_candidates(args.candidate).items()}
    clusters = load_feature_clusters(args.feature_cache, image_ids, args.cluster_k, args.seed)

    bucket_to_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    metadata_rows = []
    for row in rows:
        image_id = row["image_id"]
        image_path = args.image_root / image_id
        with Image.open(image_path) as image:
            width, height = image.size
        cid = class_ids[image_id]
        stats = class_stats.get(str(cid), {})
        count = int(stats.get("rows", stats.get("train", 0)) or 0)
        row_buckets = [
            ("frequency", bucket_freq(count)),
            ("margin", bucket_margin(margins.get(image_id, 0.0))),
            ("cluster", f"cluster_{clusters[image_id]:02d}"),
            *bucket_aspect(width, height),
        ]
        for key in row_buckets:
            bucket_to_ids[key].append(image_id)
        metadata_rows.append(
            {
                "image_id": image_id,
                "label": labels[image_id],
                "class_id": cid,
                "class_count": count,
                "width": width,
                "height": height,
                "aspect_w_h": width / max(1, height),
                "base_margin": margins.get(image_id, 0.0),
                "cluster": clusters[image_id],
                "base_pred": base.get(image_id, ""),
                "base_ok": int(base.get(image_id) == labels[image_id]),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for (bucket_type, bucket), ids in sorted(bucket_to_ids.items()):
        base_top1 = safe_top1(base, labels, ids)
        out = {
            "bucket_type": bucket_type,
            "bucket": bucket,
            "rows": len(ids),
            "base_top1": base_top1,
        }
        for name, preds in candidates.items():
            top1 = safe_top1(preds, labels, ids)
            changed = sum(preds.get(image_id) != base.get(image_id) for image_id in ids)
            wins = sum(base.get(image_id) != labels[image_id] and preds.get(image_id) == labels[image_id] for image_id in ids)
            losses = sum(base.get(image_id) == labels[image_id] and preds.get(image_id) != labels[image_id] for image_id in ids)
            out[f"{name}_top1"] = top1
            out[f"{name}_delta"] = top1 - base_top1
            out[f"{name}_changed"] = changed
            out[f"{name}_wins"] = wins
            out[f"{name}_losses"] = losses
            out[f"{name}_net"] = wins - losses
        summary_rows.append(out)

    with (args.out_dir / "bucket_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    overall = {
        "rows": len(image_ids),
        "base_top1": safe_top1(base, labels, image_ids),
        "candidate_overall": {},
        "bucket_summary_csv": str(args.out_dir / "bucket_summary.csv"),
        "metadata_csv": str(args.out_dir / "metadata.csv"),
    }
    for name, preds in candidates.items():
        overall["candidate_overall"][name] = {
            "top1": safe_top1(preds, labels, image_ids),
            "changed": sum(preds.get(image_id) != base.get(image_id) for image_id in image_ids),
            "wins": sum(base.get(image_id) != labels[image_id] and preds.get(image_id) == labels[image_id] for image_id in image_ids),
            "losses": sum(base.get(image_id) == labels[image_id] and preds.get(image_id) != labels[image_id] for image_id in image_ids),
        }
        overall["candidate_overall"][name]["net"] = (
            overall["candidate_overall"][name]["wins"] - overall["candidate_overall"][name]["losses"]
        )
    (args.out_dir / "summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
