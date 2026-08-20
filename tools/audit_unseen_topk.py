from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_classes(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty branch name in {value}")
        out[name] = Path(path.strip())
    return out


def parse_float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def quantiles(values: list[float], qs: Iterable[float]) -> dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    out = {}
    for q in qs:
        pos = min(n - 1, max(0, int(round(q * (n - 1)))))
        out[f"q{int(q * 100):02d}"] = float(values_sorted[pos])
    return out


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log(p + 1e-12)
    return float(value)


def read_topk_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_text_features(path: Path, candidates: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidates missing from {path}; first={missing[:5]}")
    indices = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize_features(payload["features"][indices])


def topk_from_feature_cache(
    *,
    image_features_path: Path,
    text_features: Path,
    candidate_classes: Path | None,
    extra_text_features: Path | None,
    extra_weight: float,
    rerank_text_features: Path | None,
    rerank_weight: float,
    score_normalization: str,
    topk: int,
    batch_size: int,
    branch_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    image_payload = torch.load(image_features_path, map_location="cpu", weights_only=False)
    image_features = normalize_features(image_payload["features"])
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))

    text_payload = torch.load(text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(candidate_classes, list(text_payload["classes"]))
    candidate_to_idx = {name: idx for idx, name in enumerate(candidates)}
    base_features = load_text_features(text_features, candidates)
    extra_features = load_text_features(extra_text_features, candidates) if extra_text_features else None
    rerank_features = load_text_features(rerank_text_features, candidates) if rerank_text_features else None
    branch_features = {name: load_text_features(path, candidates) for name, path in branch_paths.items()}

    rows: list[dict[str, Any]] = []
    branch_rows: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in branch_paths}
    k = min(topk, len(candidates))
    for start in range(0, image_features.shape[0], batch_size):
        end = min(start + batch_size, image_features.shape[0])
        image_batch = image_features[start:end]
        base_logits = image_batch @ base_features.T
        if score_normalization == "zscore":
            base_logits = row_zscore(base_logits)
        combined = base_logits
        if extra_features is not None:
            extra_logits = image_batch @ extra_features.T
            if score_normalization == "zscore":
                extra_logits = row_zscore(extra_logits)
            combined = base_logits * (1.0 - extra_weight) + extra_logits * extra_weight
        top_scores, top_indices = combined.topk(k, dim=1)
        if rerank_features is not None and rerank_weight != 0:
            rerank_logits = image_batch @ rerank_features.T
            rerank_top_scores = torch.gather(rerank_logits, 1, top_indices)
            rerank_top_scores = row_zscore(rerank_top_scores)
            final_scores = top_scores + rerank_weight * rerank_top_scores
            order = final_scores.argsort(dim=1, descending=True)
            top_scores = torch.gather(final_scores, 1, order)
            top_indices = torch.gather(top_indices, 1, order)

        branch_batch_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, features in branch_features.items():
            logits = image_batch @ features.T
            scores, indices = logits.topk(min(2, logits.shape[1]), dim=1)
            branch_batch_outputs[name] = (scores, indices)

        for local_idx, global_idx in enumerate(range(start, end)):
            label = labels[global_idx] if global_idx < len(labels) else ""
            preds = [candidates[int(idx)] for idx in top_indices[local_idx].tolist()]
            scores = [float(score) for score in top_scores[local_idx].tolist()]
            true_rank: int | str = ""
            if label:
                true_idx = candidate_to_idx.get(label)
                if true_idx is not None:
                    match = (top_indices[local_idx] == true_idx).nonzero(as_tuple=False)
                    true_rank = int(match[0, 0].item()) + 1 if match.numel() else k + 1
            row = {
                "image_id": image_ids[global_idx],
                "label": label,
                "predictions": preds,
                "scores": scores,
                "true_rank": true_rank,
            }
            rows.append(row)
            for name, (scores_t, indices_t) in branch_batch_outputs.items():
                branch_pred = candidates[int(indices_t[local_idx, 0].item())]
                second_score = float(scores_t[local_idx, 1].item()) if scores_t.shape[1] > 1 else float("nan")
                branch_rows[name][image_ids[global_idx]] = {
                    "prediction": branch_pred,
                    "score": float(scores_t[local_idx, 0].item()),
                    "margin": float(scores_t[local_idx, 0].item() - second_score),
                    "correct": bool(label and branch_pred == label),
                }

    meta = {
        "image_features": str(image_features_path),
        "text_features": str(text_features),
        "extra_text_features": str(extra_text_features) if extra_text_features else None,
        "extra_weight": extra_weight,
        "rerank_text_features": str(rerank_text_features) if rerank_text_features else None,
        "rerank_weight": rerank_weight,
        "score_normalization": score_normalization,
        "candidate_classes": str(candidate_classes) if candidate_classes else None,
        "candidate_count": len(candidates),
        "rows": len(rows),
        "branches": {name: str(path) for name, path in branch_paths.items()},
    }
    return rows, branch_rows, meta


def write_topk_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_branch_topks(paths: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in paths.items():
        branch_rows = {}
        for row in read_topk_jsonl(path):
            scores = row.get("scores", [])
            preds = row.get("predictions", [])
            margin = ""
            if len(scores) >= 2:
                margin = float(scores[0]) - float(scores[1])
            branch_rows[row["image_id"]] = {
                "prediction": preds[0] if preds else "",
                "score": float(scores[0]) if scores else "",
                "margin": margin,
                "correct": bool(row.get("label") and preds and row.get("label") == preds[0]),
            }
        out[name] = branch_rows
    return out


def build_audit_rows(
    rows: list[dict[str, Any]],
    branch_rows: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows = []
    margins: list[float] = []
    dominant_genus_fracs: list[float] = []
    top1_genus_fracs: list[float] = []
    unique_genera_counts: list[int] = []
    labeled_count = 0
    top1_correct = 0
    top5_correct = 0
    top20_correct = 0
    correct_top20_not_top1 = 0
    conflict_counts = Counter()
    branch_correct = {name: 0 for name in branch_rows}
    branch_seen = {name: 0 for name in branch_rows}

    for row in rows:
        preds = list(row.get("predictions", []))
        scores = [float(value) for value in row.get("scores", [])]
        label = str(row.get("label") or "")
        image_id = row["image_id"]
        margin12 = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        span = scores[0] - scores[-1] if len(scores) >= 2 else 0.0
        margins.append(float(margin12))
        genera = [genus(pred) for pred in preds]
        genus_counts = Counter(genera)
        top1_genus = genera[0] if genera else ""
        top1_genus_count = genus_counts.get(top1_genus, 0)
        dominant_genus, dominant_genus_count = ("", 0)
        if genus_counts:
            dominant_genus, dominant_genus_count = genus_counts.most_common(1)[0]
        unique_genera = len(genus_counts)
        top1_frac = top1_genus_count / max(1, len(preds))
        dominant_frac = dominant_genus_count / max(1, len(preds))
        top1_genus_fracs.append(float(top1_frac))
        dominant_genus_fracs.append(float(dominant_frac))
        unique_genera_counts.append(unique_genera)

        true_rank_raw = row.get("true_rank", "")
        try:
            true_rank = int(true_rank_raw)
        except (TypeError, ValueError):
            true_rank = None
        is_labeled = bool(label)
        if is_labeled:
            labeled_count += 1
            top1_correct += int(true_rank == 1)
            top5_correct += int(true_rank is not None and true_rank <= 5)
            top20_correct += int(true_rank is not None and true_rank <= 20)
            correct_top20_not_top1 += int(true_rank is not None and 1 < true_rank <= 20)

        branch_pred_values = []
        branch_genus_values = []
        branch_fields: dict[str, Any] = {}
        for name, by_image in branch_rows.items():
            item = by_image.get(image_id, {})
            pred = item.get("prediction", "")
            branch_pred_values.append(pred)
            branch_genus_values.append(genus(pred))
            branch_fields[f"{name}_top1"] = pred
            branch_fields[f"{name}_margin12"] = item.get("margin", "")
            branch_fields[f"{name}_same_as_primary"] = bool(pred and preds and pred == preds[0])
            if is_labeled and pred:
                branch_seen[name] += 1
                branch_correct[name] += int(pred == label)

        branch_pred_conflicts = len(set(value for value in branch_pred_values if value))
        branch_genus_conflicts = len(set(value for value in branch_genus_values if value))
        conflict_counts[(branch_pred_conflicts, branch_genus_conflicts)] += 1

        audit_rows.append(
            {
                "image_id": image_id,
                "label": label,
                "primary_top1": preds[0] if preds else "",
                "primary_score1": scores[0] if scores else "",
                "primary_score2": scores[1] if len(scores) >= 2 else "",
                "top1_top2_margin": margin12,
                "score_span_top20": span,
                "true_rank": true_rank if true_rank is not None else "",
                "top1_correct": bool(is_labeled and true_rank == 1),
                "correct_in_top20_not_top1": bool(is_labeled and true_rank is not None and 1 < true_rank <= 20),
                "top1_genus": top1_genus,
                "top1_genus_count_top20": top1_genus_count,
                "top1_genus_frac_top20": top1_frac,
                "dominant_genus": dominant_genus,
                "dominant_genus_count_top20": dominant_genus_count,
                "dominant_genus_frac_top20": dominant_frac,
                "unique_genera_top20": unique_genera,
                "genus_entropy_top20": entropy_from_counts(genus_counts),
                "label_genus": genus(label),
                "top1_same_genus_as_label": bool(label and top1_genus == genus(label)),
                "branch_pred_conflicts": branch_pred_conflicts,
                "branch_genus_conflicts": branch_genus_conflicts,
                **branch_fields,
            }
        )

    margin_q = quantiles(margins, [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95])
    low_margin_q20 = margin_q.get("q20", 0.0)
    for row in audit_rows:
        row["low_margin_q20"] = float(row["top1_top2_margin"]) <= low_margin_q20
        row["same_genus_clustered"] = float(row["top1_genus_frac_top20"]) >= 0.30
        row["high_uncertainty"] = bool(
            row["low_margin_q20"]
            or int(row["branch_genus_conflicts"]) >= 2
            or (float(row["top1_genus_frac_top20"]) >= 0.30 and float(row["top1_top2_margin"]) <= margin_q.get("q50", 0.0))
        )

    summary: dict[str, Any] = {
        "rows": len(rows),
        "labeled": labeled_count,
        "top1": top1_correct / labeled_count if labeled_count else None,
        "top5": top5_correct / labeled_count if labeled_count else None,
        "top20": top20_correct / labeled_count if labeled_count else None,
        "correct_in_top20_not_top1": correct_top20_not_top1,
        "correct_in_top20_not_top1_frac": correct_top20_not_top1 / labeled_count if labeled_count else None,
        "margin_quantiles": margin_q,
        "avg_top1_top2_margin": sum(margins) / len(margins) if margins else None,
        "avg_top1_genus_frac_top20": sum(top1_genus_fracs) / len(top1_genus_fracs) if top1_genus_fracs else None,
        "avg_dominant_genus_frac_top20": sum(dominant_genus_fracs) / len(dominant_genus_fracs) if dominant_genus_fracs else None,
        "avg_unique_genera_top20": sum(unique_genera_counts) / len(unique_genera_counts) if unique_genera_counts else None,
        "high_uncertainty_rows": sum(int(row["high_uncertainty"]) for row in audit_rows),
        "low_margin_q20_rows": sum(int(row["low_margin_q20"]) for row in audit_rows),
        "same_genus_clustered_rows": sum(int(row["same_genus_clustered"]) for row in audit_rows),
        "branch_conflict_buckets": {
            f"pred{pred_count}_genus{genus_count}": count
            for (pred_count, genus_count), count in sorted(conflict_counts.items())
        },
        "branch_top1_accuracy": {
            name: branch_correct[name] / branch_seen[name] if branch_seen[name] else None
            for name in branch_rows
        },
    }
    return audit_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, default=None)
    parser.add_argument("--branch-topk", action="append", default=[], help="name=topk.jsonl")
    parser.add_argument("--image-features", type=Path, default=None)
    parser.add_argument("--text-features", type=Path, default=None)
    parser.add_argument("--extra-text-features", type=Path, default=None)
    parser.add_argument("--extra-weight", type=float, default=0.0)
    parser.add_argument("--rerank-text-features", type=Path, default=None)
    parser.add_argument("--rerank-weight", type=float, default=0.0)
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--branch-feature", action="append", default=[], help="name=text_features.pt")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_meta: dict[str, Any] = {}
    if args.image_features is not None:
        if args.text_features is None:
            raise ValueError("--text-features is required with --image-features")
        rows, branch_rows, feature_meta = topk_from_feature_cache(
            image_features_path=args.image_features,
            text_features=args.text_features,
            candidate_classes=args.candidate_classes,
            extra_text_features=args.extra_text_features,
            extra_weight=args.extra_weight,
            rerank_text_features=args.rerank_text_features,
            rerank_weight=args.rerank_weight,
            score_normalization=args.score_normalization,
            topk=args.topk,
            batch_size=args.batch_size,
            branch_paths=parse_named_paths(args.branch_feature),
        )
        write_topk_jsonl(args.out_dir / "primary_topk.jsonl", rows)
    else:
        if args.topk_jsonl is None:
            raise ValueError("Provide either --topk-jsonl or --image-features")
        rows = read_topk_jsonl(args.topk_jsonl)
        branch_rows = load_branch_topks(parse_named_paths(args.branch_topk))

    audit_rows, summary = build_audit_rows(rows, branch_rows)
    summary["source_topk_jsonl"] = str(args.topk_jsonl) if args.topk_jsonl else None
    summary["feature_meta"] = feature_meta
    summary["branch_topk_files"] = {name: str(path) for name, path in parse_named_paths(args.branch_topk).items()}
    write_csv(args.out_dir / "per_image_audit.csv", audit_rows)
    write_csv(
        args.out_dir / "correct_in_top20_not_top1.csv",
        [row for row in audit_rows if row.get("correct_in_top20_not_top1")],
    )
    write_csv(
        args.out_dir / "high_uncertainty.csv",
        [row for row in audit_rows if row.get("high_uncertainty")],
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
