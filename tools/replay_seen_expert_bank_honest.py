"""Replay the frozen seen expert bank on the old and honest grouped splits.

This script deliberately does not train or tune anything.  It only changes the
evaluation mask, so a non-positive honest sealed result is a stop signal for
seen overlays rather than an invitation to search more gates.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stat(base: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    b = base.eq(labels)
    c = pred.eq(labels)
    changed = mask & base.ne(pred)
    wins = changed & (~b) & c
    losses = changed & b & (~c)
    n = int(mask.sum().item())
    return {
        "rows": n,
        "base_correct": int((mask & b).sum().item()),
        "candidate_correct": int((mask & c).sum().item()),
        "base_accuracy": float((mask & b).sum().item() / max(1, n)),
        "candidate_accuracy": float((mask & c).sum().item() / max(1, n)),
        "changed": int(changed.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net": int(wins.sum().item() - losses.sum().item()),
        "efficiency": float((wins.sum().item() - losses.sum().item()) / max(1, int(changed.sum().item()))),
    }


def bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 50:
        return "11-50"
    return "51+"


def simple_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 760
    left, right, top, bottom = 120, 40, 90, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
    draw.text((left, 25), title, fill="black", font=title_font)
    plot_w, plot_h = width - left - right, height - top - bottom
    vmax = max([abs(float(v)) for v in values] + [1.0])
    if min(values) < 0 < max(values):
        y0 = top + plot_h // 2
        scale = plot_h / (2.0 * vmax)
    else:
        y0 = height - bottom
        scale = plot_h / vmax
    draw.line((left, y0, width - right, y0), fill="gray", width=2)
    n = max(1, len(values))
    bar_w = max(3, int(plot_w / n * 0.75))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + int((i + 0.5) * plot_w / n) - bar_w // 2
        y = y0 - int(float(value) * scale)
        draw.rectangle((x, min(y, y0), x + bar_w, max(y, y0)), fill=(55, 110, 190) if value >= 0 else (205, 85, 75))
        if n <= 30 or i % max(1, n // 30) == 0:
            draw.text((x, height - bottom + 8), str(label)[:18], fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-bank", type=Path, default=Path("runs/local_20260807_seen_candidate_bank_fusion/candidate_bank_scores_gateboost76.pt"))
    parser.add_argument("--assignment", type=Path, default=Path("runs/research_next_20260820/honest_seen_phash1/assignment.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/honest_seen_phash1/replay"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bank = load(args.seen_bank)
    labels = torch.as_tensor(bank["labels"], dtype=torch.long)
    base = torch.as_tensor(bank["top_indices"][:, 0], dtype=torch.long)
    top_indices = torch.as_tensor(bank["top_indices"], dtype=torch.long)
    scores = {str(k): torch.as_tensor(v, dtype=torch.float32) for k, v in bank["scores"].items()}
    n = labels.numel()
    image_ids = [Path(str(x)).name for x in bank.get("image_ids", range(n))]
    assignment: dict[str, dict[str, str]] = {}
    with args.assignment.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            assignment[Path(row["image_id"]).name] = row
    missing = [x for x in image_ids if x not in assignment]
    if missing:
        raise RuntimeError("assignment is missing bank rows: " + repr(missing[:3]))

    # Masks include the legacy split and each of the five locked seeds.
    masks: dict[str, torch.Tensor] = {
        "old_dev": torch.as_tensor(bank.get("dev", torch.ones(n, dtype=torch.bool))).bool(),
    }
    masks["old_sealed"] = ~masks["old_dev"]
    seed_names: list[str] = []
    for seed in (42, 43, 44, 45, 46):
        for partition in ("dev", "sealed"):
            name = f"seed_{seed}_{partition}"
            seed_names.append(name)
            masks[name] = torch.tensor([assignment[x][f"seed_{seed}"] == partition for x in image_ids], dtype=torch.bool)

    # Frozen predictions: the bank's top five candidate IDs are the same IDs
    # used when the bank was built; only the expert score column changes.
    predictions: dict[str, torch.Tensor] = {}
    for name, value in scores.items():
        predictions[name] = top_indices.gather(1, value.argmax(dim=1, keepdim=True)).squeeze(1)

    rows: list[dict[str, Any]] = []
    for split, mask in masks.items():
        for expert, pred in predictions.items():
            row = {"split": split, "expert": expert}
            row.update(stat(base, pred, labels, mask))
            rows.append(row)

    # Oracle is a diagnostic only: it is the union of all frozen experts and
    # must not be turned into a submission rule.
    oracle_correct = torch.zeros(n, dtype=torch.bool)
    for pred in predictions.values():
        oracle_correct |= pred.eq(labels)
    oracle_rows: list[dict[str, Any]] = []
    for split, mask in masks.items():
        b = mask & base.eq(labels)
        o = mask & oracle_correct
        oracle_rows.append({
            "split": split,
            "rows": int(mask.sum()),
            "base_correct": int(b.sum()),
            "oracle_correct": int(o.sum()),
            "oracle_complement": int((o & ~b).sum()),
            "oracle_gain_over_base": int(o.sum() - b.sum()),
        })

    # Composition uses the text label in assignment.csv rather than the
    # candidate-bank integer ID.  The bank was built by an older cache and its
    # integer namespace is not a safe grouping key, while image_id/label text
    # is the immutable audit identity.
    class_text = [assignment[x].get("label", "") for x in image_ids]
    class_genus = [assignment[x].get("genus", "") for x in image_ids]
    total_class_counts = Counter(class_text)
    composition: list[dict[str, Any]] = []
    for split, mask in masks.items():
        idx = torch.where(mask)[0].tolist()
        counts = Counter(class_text[i] for i in idx)
        genera = {class_genus[i] for i in idx}
        # Buckets describe the class's total validation support, not the
        # number accidentally assigned to this panel.  This is the quantity
        # that exposes the legacy sealed-panel bias.
        buckets = Counter(bucket(total_class_counts[k]) for k in counts)
        composition.append({
            "split": split,
            "rows": len(idx),
            "classes": len(counts),
            "genera": len(genera - {""}),
            "mean_rows_per_class": float(len(idx) / max(1, len(counts))),
            "classes_bucket_1": buckets["1"],
            "classes_bucket_2": buckets["2"],
            "classes_bucket_3-5": buckets["3-5"],
            "classes_bucket_6-10": buckets["6-10"],
            "classes_bucket_11-50": buckets["11-50"],
            "classes_bucket_51+": buckets["51+"],
        })

    write_csv(args.out_dir / "expert_replay.csv", rows)
    write_csv(args.out_dir / "oracle_replay.csv", oracle_rows)
    write_csv(args.out_dir / "split_composition.csv", composition)

    sealed_rows = [r for r in rows if r["split"].endswith("sealed")]
    best_by_split: dict[str, dict[str, Any]] = {}
    for split in masks:
        candidates = [r for r in rows if r["split"] == split]
        best_by_split[split] = max(candidates, key=lambda r: (int(r["net"]), int(r["wins"]), -int(r["losses"])))
    expert_seed_worst: list[dict[str, Any]] = []
    for expert in predictions:
        cand = [r for r in sealed_rows if r["expert"] == expert and r["split"].startswith("seed_")]
        if cand:
            worst = min(cand, key=lambda r: (int(r["net"]), int(r["wins"]), -int(r["losses"])))
            expert_seed_worst.append({"expert": expert, "worst_seed_split": worst["split"], "worst_net": int(worst["net"]), "worst_wins": int(worst["wins"]), "worst_losses": int(worst["losses"])})
    expert_seed_worst.sort(key=lambda r: (r["worst_net"], -r["worst_wins"]))
    stable_experts = [x for x in expert_seed_worst if int(x["worst_net"]) >= 0]
    summary = {
        "rows": int(n),
        "experts": len(predictions),
        "old_split_bias": {
            "old_dev_classes": next(r["classes"] for r in composition if r["split"] == "old_dev"),
            "old_sealed_classes": next(r["classes"] for r in composition if r["split"] == "old_sealed"),
            "old_sealed_singleton_classes": next(r["classes_bucket_1"] for r in composition if r["split"] == "old_sealed"),
        },
        "best_by_split": best_by_split,
        "worst_seed_sealed_by_expert": expert_seed_worst,
        "all_expert_seed_sealed_nonnegative": all(int(x["worst_net"]) >= 0 for x in expert_seed_worst),
        "experts_with_all_seed_sealed_nonnegative": stable_experts,
        "stable_expert_count": len(stable_experts),
        "oracle": oracle_rows,
    }
    (args.out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot best expert per seed sealed and the oracle complement.
    seed_labels: list[str] = []
    seed_nets: list[float] = []
    for seed in (42, 43, 44, 45, 46):
        split = f"seed_{seed}_sealed"
        best = best_by_split[split]
        seed_labels.append(str(seed))
        seed_nets.append(float(best["net"]))
    simple_bar(args.out_dir / "honest_sealed_best_net_by_seed.png", seed_labels, seed_nets, "Best frozen expert net on honest sealed folds")
    oracle_seed = [next(r for r in oracle_rows if r["split"] == f"seed_{seed}_sealed")["oracle_complement"] for seed in (42, 43, 44, 45, 46)]
    simple_bar(args.out_dir / "honest_sealed_oracle_complement.png", seed_labels, [float(x) for x in oracle_seed], "Frozen-bank oracle complement on honest sealed folds")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
