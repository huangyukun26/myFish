#!/usr/bin/env python
"""Build and audit an iNaturalist external-gallery sidecar for FishNet seen.

The sidecar is intentionally inference-side first: it scores query images
against external same-species galleries, compares the best external class to
the current prediction's official-train support score, and emits reversible
candidate packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_cache(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "features" not in obj or "class_ids" not in obj:
        raise ValueError(f"Unsupported cache format: {path}")
    obj["features"] = F.normalize(obj["features"].float(), dim=1)
    obj["class_ids"] = torch.as_tensor(obj["class_ids"]).long()
    obj["image_ids"] = list(obj["image_ids"])
    obj["classes"] = list(obj.get("classes") or [])
    return obj


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def group_indices(class_ids: torch.Tensor, target_class_ids: list[int]) -> dict[int, torch.Tensor]:
    return {
        cid: torch.nonzero(class_ids == cid, as_tuple=False).flatten()
        for cid in target_class_ids
    }


def score_against_groups(
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    groups: dict[int, torch.Tensor],
    target_class_ids: list[int],
    *,
    mode: str,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.full((query_features.shape[0], len(target_class_ids)), float("-inf"), dtype=torch.float32)
    for start in range(0, query_features.shape[0], chunk_size):
        end = min(start + chunk_size, query_features.shape[0])
        sims = query_features[start:end] @ support_features.T
        for j, cid in enumerate(target_class_ids):
            idx = groups[cid]
            if idx.numel() == 0:
                continue
            vals = sims[:, idx]
            if mode == "max":
                out[start:end, j] = vals.max(dim=1).values
            elif mode == "mean":
                out[start:end, j] = vals.mean(dim=1)
            else:
                raise ValueError(mode)
    return out


def external_top_scores(external: dict[str, Any], query_features: torch.Tensor, mode: str, chunk_size: int) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    ext_class_ids = sorted(int(x) for x in torch.unique(external["class_ids"]).tolist() if int(x) >= 0)
    groups = group_indices(external["class_ids"], ext_class_ids)
    scores = score_against_groups(
        query_features,
        external["features"],
        groups,
        ext_class_ids,
        mode=mode,
        chunk_size=chunk_size,
    )
    top = scores.max(dim=1)
    pred_cids = torch.tensor([ext_class_ids[int(i)] for i in top.indices.tolist()], dtype=torch.long)
    return ext_class_ids, scores, pred_cids, top.values


def internal_current_scores(
    query_features: torch.Tensor,
    train: dict[str, Any],
    current_cids: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    unique_cids = sorted(int(x) for x in torch.unique(current_cids).tolist() if int(x) >= 0)
    groups = group_indices(train["class_ids"], unique_cids)
    out = torch.full((query_features.shape[0],), float("-inf"), dtype=torch.float32)
    for start in range(0, query_features.shape[0], chunk_size):
        end = min(start + chunk_size, query_features.shape[0])
        q = query_features[start:end]
        for cid in unique_cids:
            row_mask = current_cids[start:end] == cid
            if not bool(row_mask.any()):
                continue
            idx = groups[cid]
            if idx.numel() == 0:
                continue
            vals = q[row_mask] @ train["features"][idx].T
            out[start:end][row_mask] = vals.max(dim=1).values
    return out


def internal_loo_scores_for_train_queries(
    train: dict[str, Any],
    query_indices: list[int],
    target_class_ids: list[int],
    *,
    chunk_size: int,
) -> torch.Tensor:
    qfeat = train["features"][query_indices]
    qids = [train["image_ids"][i] for i in query_indices]
    qtrue = train["class_ids"][query_indices]
    groups = group_indices(train["class_ids"], target_class_ids)
    out = torch.full((len(query_indices), len(target_class_ids)), float("-inf"), dtype=torch.float32)
    for start in range(0, len(query_indices), chunk_size):
        end = min(start + chunk_size, len(query_indices))
        q = qfeat[start:end]
        sims = q @ train["features"].T
        for j, cid in enumerate(target_class_ids):
            idx = groups[cid]
            if idx.numel() == 0:
                continue
            vals = sims[:, idx].clone()
            for local_row, global_qi in enumerate(range(start, end)):
                if int(qtrue[global_qi]) != cid:
                    continue
                support_image_ids = [train["image_ids"][int(k)] for k in idx.tolist()]
                for support_col, image_id in enumerate(support_image_ids):
                    if image_id == qids[global_qi]:
                        vals[local_row, support_col] = float("-inf")
            out[start:end, j] = vals.max(dim=1).values
    return out


def metrics_for_selection(selected: torch.Tensor, base_pred: torch.Tensor, cand_pred: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    changed = selected & cand_pred.ne(base_pred)
    base_correct = base_pred.eq(y)
    cand_correct = cand_pred.eq(y)
    wins = changed & (~base_correct) & cand_correct
    losses = changed & base_correct & (~cand_correct)
    return {
        "selected": int(changed.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net": int(wins.sum().item() - losses.sum().item()),
        "efficiency": float((wins.sum().item() - losses.sum().item()) / max(1, int(changed.sum().item()))),
        "base_correct": int(base_correct.sum().item()),
        "after_correct": int(base_correct.sum().item() + wins.sum().item() - losses.sum().item()),
        "rows": int(y.numel()),
    }


def command_proxy(args: argparse.Namespace) -> None:
    train = load_cache(args.train_cache)
    external = load_cache(args.external_cache)
    ext_class_ids = sorted(int(x) for x in torch.unique(external["class_ids"]).tolist() if int(x) >= 0)
    covered = torch.zeros_like(train["class_ids"], dtype=torch.bool)
    for cid in ext_class_ids:
        covered |= train["class_ids"] == cid
    query_indices = torch.nonzero(covered, as_tuple=False).flatten().tolist()
    if args.max_queries:
        query_indices = query_indices[: args.max_queries]
    y = train["class_ids"][query_indices]
    _ext_class_ids, _ext_scores, ext_pred_q, ext_score_q = external_top_scores(
        external,
        train["features"][query_indices],
        args.external_mode,
        args.chunk_size,
    )

    internal_scores = internal_loo_scores_for_train_queries(
        train,
        query_indices,
        ext_class_ids,
        chunk_size=args.chunk_size,
    )
    internal_top = internal_scores.max(dim=1)
    internal_pred = torch.tensor([ext_class_ids[int(i)] for i in internal_top.indices.tolist()], dtype=torch.long)
    margin = ext_score_q - internal_top.values

    rows = []
    for threshold in parse_floats(args.thresholds):
        selected = margin >= threshold
        rows.append({"kind": "threshold", "value": threshold, **metrics_for_selection(selected, internal_pred, ext_pred_q, y)})
    order = torch.argsort(margin, descending=True)
    for count in parse_ints(args.top_counts):
        selected = torch.zeros_like(margin, dtype=torch.bool)
        selected[order[: min(count, order.numel())]] = True
        rows.append({"kind": "top_count", "value": count, **metrics_for_selection(selected, internal_pred, ext_pred_q, y)})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "proxy_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pred_rows = []
    class_names = {i: name for i, name in enumerate(train["classes"])}
    for local_i, train_i in enumerate(query_indices):
        pred_rows.append(
            {
                "image_id": train["image_ids"][train_i],
                "true_class_id": int(y[local_i]),
                "true_label": class_names.get(int(y[local_i]), ""),
                "internal_pred_class_id": int(internal_pred[local_i]),
                "internal_pred_label": class_names.get(int(internal_pred[local_i]), ""),
                "external_pred_class_id": int(ext_pred_q[local_i]),
                "external_pred_label": class_names.get(int(ext_pred_q[local_i]), ""),
                "internal_score": float(internal_top.values[local_i]),
                "external_score": float(ext_score_q[local_i]),
                "margin": float(margin[local_i]),
            }
        )
    with (args.out / "proxy_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(pred_rows[0].keys()) if pred_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    summary = {
        "train_cache": str(args.train_cache),
        "external_cache": str(args.external_cache),
        "external_classes": len(ext_class_ids),
        "query_rows": len(query_indices),
        "external_mode": args.external_mode,
        "best_by_net": sorted(rows, key=lambda r: (r["net"], r["efficiency"], r["selected"]), reverse=True)[:20],
    }
    (args.out / "proxy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_seen_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["image_id"] for row in csv.DictReader(f)}


def write_zip(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")


def write_package(out_dir: Path, name: str, current: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    variant = out_dir / name
    variant.mkdir(parents=True, exist_ok=True)
    pred = dict(current)
    changed = []
    for row in rows:
        image_id = row["image_id"]
        new_label = row["external_label"]
        if pred.get(image_id) == new_label:
            continue
        pred[image_id] = new_label
        changed.append(row)
    pred_path = variant / "prediction.json"
    zip_path = variant / "submission.zip"
    pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_zip(pred_path, zip_path)
    with (variant / "changed_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(changed[0].keys()) if changed else ["image_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed)
    audit = {
        "name": name,
        "changed": len(changed),
        "prediction_json": str(pred_path),
        "submission_zip": str(zip_path),
    }
    (variant / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def cap_by_external_class(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if cap <= 0:
        return rows
    counts: dict[str, int] = {}
    kept = []
    for row in rows:
        label = str(row.get("external_label", ""))
        if counts.get(label, 0) >= cap:
            continue
        counts[label] = counts.get(label, 0) + 1
        kept.append(row)
    return kept


def command_apply(args: argparse.Namespace) -> None:
    train = load_cache(args.train_cache)
    external = load_cache(args.external_cache)
    query = load_cache(args.query_cache)
    current = json.loads(args.current_prediction.read_text(encoding="utf-8"))
    lineage = json.loads(args.lineage_prediction.read_text(encoding="utf-8")) if args.lineage_prediction else {}
    seen_ids = read_seen_ids(args.test_seen_manifest)
    label_to_cid = {label: i for i, label in enumerate(train["classes"])}

    keep_indices = [i for i, image_id in enumerate(query["image_ids"]) if image_id in seen_ids and image_id in current]
    qfeat = query["features"][keep_indices]
    qids = [query["image_ids"][i] for i in keep_indices]
    current_cids = torch.tensor([label_to_cid.get(current[image_id], -1) for image_id in qids], dtype=torch.long)

    ext_class_ids, _ext_scores, ext_pred, ext_score = external_top_scores(external, qfeat, args.external_mode, args.chunk_size)
    current_score = internal_current_scores(qfeat, train, current_cids, chunk_size=args.chunk_size)
    margin = ext_score - current_score
    class_names = {i: name for i, name in enumerate(train["classes"])}

    candidate_rows = []
    protected_count = 0
    for i, image_id in enumerate(qids):
        ext_cid = int(ext_pred[i])
        cur_cid = int(current_cids[i])
        protected = bool(lineage) and current.get(image_id) != lineage.get(image_id)
        protected_count += int(protected)
        row = {
            "image_id": image_id,
            "current_label": current[image_id],
            "current_class_id": cur_cid,
            "external_label": class_names.get(ext_cid, ""),
            "external_class_id": ext_cid,
            "current_score": float(current_score[i]),
            "external_score": float(ext_score[i]),
            "margin": float(margin[i]),
            "protected": protected,
        }
        if (
            ext_cid >= 0
            and cur_cid >= 0
            and ext_cid != cur_cid
            and not protected
            and float(ext_score[i]) >= args.min_external_score
            and float(margin[i]) >= args.min_margin
        ):
            candidate_rows.append(row)

    candidate_rows.sort(key=lambda r: (r["margin"], r["external_score"]), reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidate_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(candidate_rows[0].keys()) if candidate_rows else [
            "image_id",
            "current_label",
            "external_label",
            "margin",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)

    package_audits = {}
    for count in parse_ints(args.package_top_counts):
        rows = cap_by_external_class(candidate_rows[: min(count, len(candidate_rows))], args.package_per_external_cap)
        package_audits[f"top{count:04d}"] = write_package(args.out / "packages", f"top{count:04d}", current, rows)
    for threshold in parse_floats(args.package_thresholds):
        rows = [r for r in candidate_rows if float(r["margin"]) >= threshold]
        rows = cap_by_external_class(rows, args.package_per_external_cap)
        safe = str(threshold).replace("-", "m").replace(".", "p")
        package_audits[f"margin_ge_{safe}"] = write_package(args.out / "packages", f"margin_ge_{safe}", current, rows)

    summary = {
        "train_cache": str(args.train_cache),
        "external_cache": str(args.external_cache),
        "query_cache": str(args.query_cache),
        "query_rows": len(qids),
        "external_classes": len(ext_class_ids),
        "protected_rows": protected_count,
        "candidate_rows": len(candidate_rows),
        "min_margin": args.min_margin,
        "min_external_score": args.min_external_score,
        "package_per_external_cap": args.package_per_external_cap,
        "external_mode": args.external_mode,
        "packages": package_audits,
    }
    (args.out / "apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    proxy = sub.add_parser("proxy")
    proxy.add_argument("--train-cache", type=Path, required=True)
    proxy.add_argument("--external-cache", type=Path, required=True)
    proxy.add_argument("--out", type=Path, required=True)
    proxy.add_argument("--external-mode", choices=["max", "mean"], default="max")
    proxy.add_argument("--chunk-size", type=int, default=256)
    proxy.add_argument("--thresholds", default="-0.10,-0.05,0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20")
    proxy.add_argument("--top-counts", default="50,100,200,400,800,1200,2000")
    proxy.add_argument("--max-queries", type=int, default=0)
    proxy.set_defaults(func=command_proxy)

    apply_cmd = sub.add_parser("apply")
    apply_cmd.add_argument("--train-cache", type=Path, required=True)
    apply_cmd.add_argument("--external-cache", type=Path, required=True)
    apply_cmd.add_argument("--query-cache", type=Path, required=True)
    apply_cmd.add_argument("--current-prediction", type=Path, required=True)
    apply_cmd.add_argument("--lineage-prediction", type=Path, default=None)
    apply_cmd.add_argument("--test-seen-manifest", type=Path, required=True)
    apply_cmd.add_argument("--out", type=Path, required=True)
    apply_cmd.add_argument("--external-mode", choices=["max", "mean"], default="max")
    apply_cmd.add_argument("--chunk-size", type=int, default=256)
    apply_cmd.add_argument("--min-margin", type=float, default=0.0)
    apply_cmd.add_argument("--min-external-score", type=float, default=-1.0)
    apply_cmd.add_argument("--package-top-counts", default="50,100,200,400,800")
    apply_cmd.add_argument("--package-thresholds", default="0.02,0.04,0.06,0.08,0.10,0.12")
    apply_cmd.add_argument("--package-per-external-cap", type=int, default=0)
    apply_cmd.set_defaults(func=command_apply)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
