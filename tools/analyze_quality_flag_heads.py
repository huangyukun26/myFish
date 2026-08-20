from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], class_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, class_ids.tolist())):
        groups[int(class_id)].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for position, (_digest, index) in enumerate(rows):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def zscore_rows(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def paired(base: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    base_ok = base.eq(labels)
    candidate_ok = candidate.eq(labels)
    changed = mask & candidate.ne(base)
    wins = mask & ~base_ok & candidate_ok
    losses = mask & base_ok & ~candidate_ok
    return {
        "rows": int(mask.sum()),
        "base_correct": int((mask & base_ok).sum()),
        "candidate_correct": int((mask & candidate_ok).sum()),
        "net": int(wins.sum() - losses.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "changed": int(changed.sum()),
    }


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate(reference: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    for key in ("image_ids", "classes"):
        if list(reference[key]) != list(candidate[key]):
            raise RuntimeError(f"{key} mismatch: {path}")
    if not torch.equal(reference["class_ids"].long(), candidate["class_ids"].long()):
        raise RuntimeError(f"class_ids mismatch: {path}")


def read_flags(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for category in row["categories"].split("|"):
                if category:
                    result[row["image_id"]].add(category)
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--flags-val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    labels = reference["class_ids"].long()
    image_ids = list(reference["image_ids"])
    reference_logits = reference["logits"].float()
    reference_pred = reference_logits.argmax(dim=1)
    dev, sealed = split_dev_sealed(image_ids, labels)
    masks = {"all": torch.ones(len(labels), dtype=torch.bool), "dev": dev, "sealed": sealed}
    flags = read_flags(args.flags_val)
    categories = sorted({category for values in flags.values() for category in values})
    flag_masks = {
        category: torch.tensor([category in flags.get(image_id, set()) for image_id in image_ids])
        for category in categories
    }
    flag_masks["ANY_FLAG"] = torch.tensor([image_id in flags for image_id in image_ids])

    variants: dict[str, dict[str, Any]] = {}
    for directory in sorted(args.head_root.iterdir()):
        path = directory / "val_logits.pt"
        if not path.exists():
            continue
        candidate = load(path)
        validate(reference, candidate, path)
        logits = candidate["logits"].float()
        prediction = logits.argmax(dim=1)
        direct = {name: paired(reference_pred, prediction, labels, mask) for name, mask in masks.items()}
        direct["oracle_complement"] = int((~reference_pred.eq(labels) & prediction.eq(labels)).sum())
        direct["union_correct"] = int((reference_pred.eq(labels) | prediction.eq(labels)).sum())
        per_category = {
            name: paired(reference_pred, prediction, labels, mask)
            for name, mask in flag_masks.items()
        }

        ref_z = zscore_rows(reference_logits)
        candidate_z = zscore_rows(logits)
        blend_rows = []
        for step in range(51):
            alpha = step / 100.0
            blend_pred = ((1.0 - alpha) * ref_z + alpha * candidate_z).argmax(dim=1)
            blend_rows.append(
                {
                    "alpha": alpha,
                    **{name: paired(reference_pred, blend_pred, labels, mask) for name, mask in masks.items()},
                }
            )
        selected = max(blend_rows, key=lambda row: (row["dev"]["net"], -row["alpha"]))
        variants[directory.name] = {
            "logits": str(path),
            "direct": direct,
            "per_category": per_category,
            "dev_selected_blend": selected,
            "blend_scan": blend_rows,
        }
        del logits, candidate_z

    reference_metrics = {
        name: {
            "rows": int(mask.sum()),
            "correct": int((mask & reference_pred.eq(labels)).sum()),
            "top1": float((mask & reference_pred.eq(labels)).sum() / mask.sum()),
        }
        for name, mask in masks.items()
    }
    result = {
        "protocol_warning": (
            "Exploratory only: train_embedding_mlp_classifier selected each checkpoint on full validation top-1, "
            "so the displayed sealed partition is not an untouched confirmatory holdout."
        ),
        "reference": str(args.reference),
        "reference_metrics": reference_metrics,
        "split": {"dev_rows": int(dev.sum()), "sealed_rows": int(sealed.sum())},
        "flagged_val_rows": int(flag_masks["ANY_FLAG"].sum()),
        "variants": variants,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Quality-flag head ablation",
        "",
        f"Reference correct: {reference_metrics['all']['correct']}/{reference_metrics['all']['rows']} "
        f"({reference_metrics['all']['top1']:.6f}).",
        "",
        "> Warning: these are exploratory results because each head checkpoint was selected on full validation top-1. "
        "The sealed column is diagnostic, not a clean confirmatory read.",
        "",
        "| variant | direct correct | direct net | oracle complement | dev-selected alpha | blend dev net | blend sealed net | blend all net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in variants.items():
        locked = row["dev_selected_blend"]
        lines.append(
            f"| {name} | {row['direct']['all']['candidate_correct']} | {row['direct']['all']['net']:+d} | "
            f"{row['direct']['oracle_complement']} | {locked['alpha']:.2f} | {locked['dev']['net']:+d} | "
            f"{locked['sealed']['net']:+d} | {locked['all']['net']:+d} |"
        )
    lines.extend(["", "## Flagged validation rows", ""])
    for name, row in variants.items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| category | rows | reference correct | candidate correct | net | wins | losses |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for category, stats in row["per_category"].items():
            lines.append(
                f"| {category} | {stats['rows']} | {stats['base_correct']} | {stats['candidate_correct']} | "
                f"{stats['net']:+d} | {stats['wins']} | {stats['losses']} |"
            )
        lines.append("")
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
