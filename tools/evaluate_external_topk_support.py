#!/usr/bin/env python
"""Evaluate public external image support as a top-k-only reranking signal.

This deliberately does not use nearest-neighbor labels directly. External images
can only support labels already present in the model top-k list.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def genus(label: str) -> str:
    parts = label.split()
    return parts[0] if parts else label


def load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "features" not in payload or "image_ids" not in payload:
        raise RuntimeError(f"{path} is not a feature cache")
    payload["features"] = F.normalize(payload["features"].float(), dim=1)
    payload["image_ids"] = list(payload["image_ids"])
    if "labels" in payload:
        payload["labels"] = list(payload["labels"])
    return payload


def read_topk(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "image_id": row["image_id"],
                        "label": row.get("label"),
                        "predictions": list(row["predictions"]),
                        "scores": [float(x) for x in row["scores"]],
                    }
                )
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            preds: list[str] = []
            scores: list[float] = []
            if "predictions" in row and row["predictions"].strip().startswith("["):
                preds = list(json.loads(row["predictions"]))
                scores = [float(x) for x in json.loads(row["scores"])]
            else:
                for idx in range(1, 101):
                    pred_key = f"prediction_{idx}"
                    score_key = f"score_{idx}"
                    if pred_key not in row:
                        break
                    if row[pred_key] == "":
                        break
                    preds.append(row[pred_key])
                    scores.append(float(row[score_key]))
            rows.append(
                {
                    "image_id": row["image_id"],
                    "label": row.get("label"),
                    "predictions": preds,
                    "scores": scores,
                }
            )
    return rows


def build_external_by_label(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    labels = cache.get("labels")
    if labels is None:
        raise RuntimeError("external cache needs labels")
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[str(label)].append(idx)
    features = cache["features"]
    return {label: features[indices] for label, indices in grouped.items()}


def support_for_candidates(
    query_feature: torch.Tensor,
    candidates: list[str],
    external_by_label: dict[str, torch.Tensor],
) -> tuple[list[float], list[int]]:
    support: list[float] = []
    counts: list[int] = []
    for label in candidates:
        feats = external_by_label.get(label)
        if feats is None:
            support.append(float("-inf"))
            counts.append(0)
        else:
            sims = feats @ query_feature
            support.append(float(sims.max().item()))
            counts.append(int(feats.shape[0]))
    return support, counts


def zscore(values: list[float], mask_missing: bool = True) -> list[float]:
    finite = torch.tensor([v for v in values if v != float("-inf")], dtype=torch.float32)
    if finite.numel() == 0:
        return [0.0 for _ in values]
    mean = finite.mean()
    std = finite.std(unbiased=False).clamp_min(1e-6)
    out: list[float] = []
    for value in values:
        if value == float("-inf") and mask_missing:
            out.append(-10.0)
        elif value == float("-inf"):
            out.append(0.0)
        else:
            out.append(float((torch.tensor(value) - mean) / std))
    return out


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "rows": len(rows),
        "covered_any": 0,
        "base_wrong": 0,
        "base_wrong_true_in_topk": 0,
        "base_wrong_true_has_external": 0,
        "base_wrong_true_has_external_same_genus_top1": 0,
        "base_wrong_any_nonbase_external": 0,
        "covered_true_labels": 0,
        "covered_pred_labels": 0,
    }
    covered_true_labels: set[str] = set()
    covered_pred_labels: set[str] = set()
    for row in rows:
        label = row.get("label")
        if any(c > 0 for c in row["external_counts"]):
            out["covered_any"] += 1
        for pred, count in zip(row["predictions"], row["external_counts"]):
            if count > 0:
                covered_pred_labels.add(pred)
        if label and label in row["predictions"]:
            pos = row["predictions"].index(label)
            if row["external_counts"][pos] > 0:
                covered_true_labels.add(label)
        if not label:
            continue
        base = row["predictions"][0]
        base_wrong = base != label
        if base_wrong:
            out["base_wrong"] += 1
            if label in row["predictions"]:
                out["base_wrong_true_in_topk"] += 1
                pos = row["predictions"].index(label)
                if row["external_counts"][pos] > 0:
                    out["base_wrong_true_has_external"] += 1
                    if genus(label) == genus(base):
                        out["base_wrong_true_has_external_same_genus_top1"] += 1
            if any(c > 0 for c in row["external_counts"][1:]):
                out["base_wrong_any_nonbase_external"] += 1
    out["covered_true_labels"] = len(covered_true_labels)
    out["covered_pred_labels"] = len(covered_pred_labels)
    return out


def split_dev_outer(image_ids: list[str]) -> tuple[set[str], set[str]]:
    dev: set[str] = set()
    outer: set[str] = set()
    for image_id in image_ids:
        if stable_hash("external-support:" + image_id) % 2 == 0:
            dev.add(image_id)
        else:
            outer.add(image_id)
    return dev, outer


def evaluate_rows(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    image_ids = [row["image_id"] for row in rows]
    dev, outer = split_dev_outer(image_ids)
    result: dict[str, Any] = {}
    masks = {
        "all": set(image_ids),
        "dev": dev,
        "outer": outer,
    }
    for rule in rules:
        name = rule["name"]
        per_split: dict[str, Any] = {}
        for split_name, split_ids in masks.items():
            base_correct = 0
            cand_correct = 0
            changed = 0
            wins = 0
            losses = 0
            covered = 0
            for row in rows:
                if row["image_id"] not in split_ids:
                    continue
                label = row.get("label")
                if not label:
                    continue
                base = row["predictions"][0]
                cand = row.get(f"pred_{name}", base)
                base_ok = base == label
                cand_ok = cand == label
                base_correct += int(base_ok)
                cand_correct += int(cand_ok)
                changed += int(cand != base)
                wins += int((cand != base) and (not base_ok) and cand_ok)
                losses += int((cand != base) and base_ok and (not cand_ok))
                covered += int(row.get("covered_candidates", 0) > 0)
            total = max(1, len(split_ids))
            per_split[split_name] = {
                "rows": len(split_ids),
                "base_correct": base_correct,
                "candidate_correct": cand_correct,
                "net": cand_correct - base_correct,
                "gain": (cand_correct - base_correct) / total,
                "changed": changed,
                "wins": wins,
                "losses": losses,
                "efficiency": (wins - losses) / max(1, changed),
                "covered_rows": covered,
            }
        result[name] = per_split
    return result


def annotate_candidates(
    topk_rows: list[dict[str, Any]],
    query_cache: dict[str, Any],
    external_by_label: dict[str, torch.Tensor],
    k: int,
) -> list[dict[str, Any]]:
    qpos = {image_id: idx for idx, image_id in enumerate(query_cache["image_ids"])}
    annotated: list[dict[str, Any]] = []
    for row in topk_rows:
        image_id = row["image_id"]
        if image_id not in qpos:
            continue
        preds = row["predictions"][:k]
        scores = row["scores"][:k]
        if not preds:
            continue
        support, counts = support_for_candidates(query_cache["features"][qpos[image_id]], preds, external_by_label)
        row2 = dict(row)
        row2["predictions"] = preds
        row2["scores"] = scores
        row2["external_support"] = support
        row2["external_counts"] = counts
        row2["covered_candidates"] = sum(1 for c in counts if c > 0)
        annotated.append(row2)
    return annotated


def apply_rules(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    for row in rows:
        preds = row["predictions"]
        scores = row["scores"]
        support = row["external_support"]
        counts = row["external_counts"]
        if len(preds) < 2:
            for rule in rules:
                row[f"pred_{rule['name']}"] = preds[0]
            continue

        score_z = zscore(scores, mask_missing=False)
        support_z = zscore(support, mask_missing=True)
        base = preds[0]
        base_margin = scores[0] - scores[1]
        base_support = support[0]

        for rule in rules:
            best_idx = 0
            best_value = score_z[0]
            for idx in range(1, len(preds)):
                if counts[idx] < rule["min_external"]:
                    continue
                if rule["same_genus"] and genus(preds[idx]) != genus(base):
                    continue
                if base_margin > rule["max_base_margin"]:
                    continue
                # Missing external evidence is unknown, not a low score. A
                # missing base label must never create an artificial support
                # advantage for the candidate label.
                if base_support == float("-inf") or support[idx] == float("-inf"):
                    continue
                support_gap = support[idx] - base_support
                if support_gap < rule["min_support_gap"]:
                    continue
                value = score_z[idx] + rule["support_weight"] * support_z[idx]
                if value > best_value + rule["min_combined_gap"]:
                    best_idx = idx
                    best_value = value
            row[f"pred_{rule['name']}"] = preds[best_idx]
            row[f"slot_{rule['name']}"] = best_idx


def write_val_rows(path: Path, rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "label",
        "base_pred",
        "base_score",
        "base_margin",
        "covered_candidates",
        "predictions",
        "scores",
        "external_support",
        "external_counts",
    ]
    for rule in rules:
        fields.extend([f"pred_{rule['name']}", f"slot_{rule['name']}"])
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "image_id": row["image_id"],
                "label": row.get("label", ""),
                "base_pred": row["predictions"][0],
                "base_score": row["scores"][0],
                "base_margin": row["scores"][0] - row["scores"][1] if len(row["scores"]) > 1 else "",
                "covered_candidates": row["covered_candidates"],
                "predictions": json.dumps(row["predictions"], ensure_ascii=False),
                "scores": json.dumps(row["scores"]),
                "external_support": json.dumps(row["external_support"]),
                "external_counts": json.dumps(row["external_counts"]),
            }
            for rule in rules:
                out[f"pred_{rule['name']}"] = row.get(f"pred_{rule['name']}", row["predictions"][0])
                out[f"slot_{rule['name']}"] = row.get(f"slot_{rule['name']}", 0)
            writer.writerow(out)


def load_prediction(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            return json.loads(zf.read("prediction.json").decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [row["image_id"] for row in csv.DictReader(fp)]


def write_zip(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")


def build_packages(
    rows: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    out_dir: Path,
    current_json: Path,
    previous_json: Path,
    test_seen_manifest: Path,
    max_changes: int,
) -> dict[str, Any]:
    current = load_prediction(current_json)
    previous = load_prediction(previous_json)
    seen_ids = set(read_ids(test_seen_manifest))
    protected = {image_id for image_id in seen_ids if current.get(image_id) != previous.get(image_id)}
    audits: dict[str, Any] = {}
    for rule in rules:
        pred = dict(current)
        candidates: list[dict[str, Any]] = []
        for row in rows:
            image_id = row["image_id"]
            if image_id not in seen_ids:
                continue
            if image_id in protected:
                continue
            base = row["predictions"][0]
            new = row.get(f"pred_{rule['name']}", base)
            if new == base:
                continue
            if current.get(image_id) != base:
                continue
            slot = int(row.get(f"slot_{rule['name']}", 0))
            support = row["external_support"][slot]
            base_margin = row["scores"][0] - row["scores"][1] if len(row["scores"]) > 1 else 999.0
            candidates.append(
                {
                    "image_id": image_id,
                    "old": current[image_id],
                    "new": new,
                    "slot": slot,
                    "support": support,
                    "base_margin": base_margin,
                    "covered_candidates": row["covered_candidates"],
                }
            )
        candidates.sort(key=lambda x: (x["support"], -x["base_margin"]), reverse=True)
        candidates = candidates[:max_changes]
        for item in candidates:
            pred[item["image_id"]] = item["new"]
        variant_dir = out_dir / "packages" / rule["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        pred_path = variant_dir / "prediction.json"
        zip_path = variant_dir / "submission.zip"
        changed_path = variant_dir / "changed_rows.csv"
        pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_zip(pred_path, zip_path)
        with changed_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["image_id", "old", "new", "slot", "support", "base_margin", "covered_candidates"],
            )
            writer.writeheader()
            writer.writerows(candidates)
        audit = {
            "rule": rule,
            "changed": len(candidates),
            "protected_online_gain_rows": len(protected),
            "prediction_json": str(pred_path),
            "zip": str(zip_path),
            "changed_rows": str(changed_path),
        }
        (variant_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audits[rule["name"]] = audit
    return audits


def default_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for same_genus in [True, False]:
        for max_base_margin in [0.25, 0.5, 1.0, 2.0]:
            for support_weight in [0.25, 0.5, 1.0, 1.5]:
                for min_support_gap in [0.0, 0.02, 0.05]:
                    name = (
                        f"{'samegenus' if same_genus else 'anygenus'}"
                        f"_bm{str(max_base_margin).replace('.', 'p')}"
                        f"_sw{str(support_weight).replace('.', 'p')}"
                        f"_sg{str(min_support_gap).replace('.', 'p')}"
                    )
                    rules.append(
                        {
                            "name": name,
                            "same_genus": same_genus,
                            "max_base_margin": max_base_margin,
                            "support_weight": support_weight,
                            "min_support_gap": min_support_gap,
                            "min_combined_gap": 0.0,
                            "min_external": 1,
                        }
                    )
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-topk", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--test-topk", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--current-json", type=Path, default=Path("runs/current_best_online_20260808_overall051/submission/prediction.json"))
    parser.add_argument("--previous-json", type=Path, default=Path("runs/current_best_archive_20260730_seen078046/submission/prediction.json"))
    parser.add_argument("--test-seen-manifest", type=Path, default=Path("work/full_manifests/test_seen.csv"))
    parser.add_argument("--max-package-changes", type=int, default=25)
    parser.add_argument("--min-outer-net-for-package", type=int, default=1)
    parser.add_argument("--max-packages", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    external_cache = load_cache(args.external_cache)
    external_by_label = build_external_by_label(external_cache)
    val_rows = annotate_candidates(read_topk(args.val_topk), load_cache(args.val_cache), external_by_label, args.k)
    rules = default_rules()
    apply_rules(val_rows, rules)
    metrics = evaluate_rows(val_rows, rules)
    ranked = sorted(
        (
            {
                "rule": name,
                **splits,
            }
            for name, splits in metrics.items()
        ),
        key=lambda x: (x["outer"]["net"], x["outer"]["efficiency"], x["all"]["net"], -x["all"]["changed"]),
        reverse=True,
    )
    summary = {
        "val_topk": str(args.val_topk),
        "val_cache": str(args.val_cache),
        "external_cache": str(args.external_cache),
        "external_labels": len(external_by_label),
        "external_images": len(external_cache["image_ids"]),
        "val_rows": len(val_rows),
        "val_rows_with_external_candidate": sum(1 for row in val_rows if row["covered_candidates"] > 0),
        "oracle": oracle_summary(val_rows),
        "best_by_outer": ranked[:30],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_val_rows(args.out_dir / "annotated_val_rows.csv", val_rows, rules[:0])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    eligible_names = [
        item["rule"]
        for item in ranked
        if item["outer"]["net"] >= args.min_outer_net_for_package and item["all"]["net"] >= 0 and item["all"]["changed"] > 0
    ][: args.max_packages]
    eligible_rules = [rule for rule in rules if rule["name"] in set(eligible_names)]
    if args.test_topk is not None and args.test_cache is not None and eligible_rules:
        test_rows = annotate_candidates(read_topk(args.test_topk), load_cache(args.test_cache), external_by_label, args.k)
        apply_rules(test_rows, eligible_rules)
        audits = build_packages(
            test_rows,
            eligible_rules,
            args.out_dir,
            args.current_json,
            args.previous_json,
            args.test_seen_manifest,
            args.max_package_changes,
        )
        (args.out_dir / "packages_summary.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"packages": audits}, ensure_ascii=False, indent=2), flush=True)
    elif args.test_topk is not None and args.test_cache is not None:
        print(json.dumps({"packages": "skipped_no_val_positive_rule"}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
