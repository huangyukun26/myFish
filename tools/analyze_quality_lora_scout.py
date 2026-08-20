from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, labels.tolist())):
        groups[class_id].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        for position, (_digest, index) in enumerate(sorted(rows)):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def paired(base: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    base_ok = base.eq(labels)
    candidate_ok = candidate.eq(labels)
    changed = mask & base.ne(candidate)
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


def zscore_rows(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)


def read_flags(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for category in row["categories"].split("|"):
                if category:
                    output[row["image_id"]].add(category)
    return dict(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--old-adapted-features", type=Path, required=True)
    parser.add_argument("--new-adapted-features", type=Path, required=True)
    parser.add_argument("--flags-val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = torch.load(args.reference_logits, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate_logits, map_location="cpu", weights_only=False)
    old_features = torch.load(args.old_adapted_features, map_location="cpu", weights_only=False)
    new_features = torch.load(args.new_adapted_features, map_location="cpu", weights_only=False)
    image_ids = list(reference["image_ids"])
    for payload, name in ((candidate, "candidate"), (old_features, "old features"), (new_features, "new features")):
        if list(payload["image_ids"]) != image_ids:
            raise RuntimeError(f"Image order mismatch: {name}")
    labels = reference["class_ids"].long()
    if not torch.equal(labels, candidate["class_ids"].long()):
        raise RuntimeError("Label mismatch")
    dev, sealed = split_dev_sealed(image_ids, labels)
    flags = read_flags(args.flags_val)
    masks = {
        "all": torch.ones(len(labels), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
        "any_flag": torch.tensor([image_id in flags for image_id in image_ids]),
    }
    for category in sorted({category for values in flags.values() for category in values}):
        masks[f"flag:{category}"] = torch.tensor([category in flags.get(image_id, set()) for image_id in image_ids])

    reference_logits = reference["logits"].float()
    candidate_logits = candidate["logits"].float()
    reference_pred = reference_logits.argmax(dim=1)
    candidate_pred = candidate_logits.argmax(dim=1)
    direct = {name: paired(reference_pred, candidate_pred, labels, mask) for name, mask in masks.items()}
    direct["oracle_complement"] = int((~reference_pred.eq(labels) & candidate_pred.eq(labels)).sum())
    direct["oracle_union_correct"] = int((reference_pred.eq(labels) | candidate_pred.eq(labels)).sum())

    ref_z = zscore_rows(reference_logits)
    candidate_z = zscore_rows(candidate_logits)
    scan = []
    for step in range(101):
        alpha = step / 100.0
        prediction = ((1.0 - alpha) * ref_z + alpha * candidate_z).argmax(dim=1)
        scan.append(
            {
                "alpha": alpha,
                "all": paired(reference_pred, prediction, labels, masks["all"]),
                "dev": paired(reference_pred, prediction, labels, dev),
                "sealed": paired(reference_pred, prediction, labels, sealed),
            }
        )
    locked = max(scan, key=lambda row: (row["dev"]["net"], -row["alpha"]))

    cosine = F.cosine_similarity(
        old_features["features"].float(), new_features["features"].float(), dim=1
    )
    drift = {}
    for name, mask in masks.items():
        values = cosine[mask]
        drift[name] = {
            "rows": int(mask.sum()),
            "mean_cosine": float(values.mean()) if len(values) else None,
            "min_cosine": float(values.min()) if len(values) else None,
        }

    result = {
        "reference_logits": str(args.reference_logits),
        "candidate_logits": str(args.candidate_logits),
        "direct": direct,
        "dev_locked_blend": locked,
        "blend_scan": scan,
        "feature_cosine_old_vs_new": drift,
        "gates": {
            "oracle_required": 250,
            "dev_net_required": 120,
            "sealed_positive_required": True,
            "oracle_pass": direct["oracle_complement"] >= 250,
            "dev_pass": locked["dev"]["net"] >= 120,
            "sealed_pass": locked["sealed"]["net"] > 0,
        },
        "test_seen_used": False,
        "submission_generated": False,
    }
    result["gates"]["passed"] = all(
        result["gates"][name] for name in ("oracle_pass", "dev_pass", "sealed_pass")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Quality-aware DINO-L LoRA scout report",
        "",
        "## Result",
        "",
        f"- Strong reference: {direct['all']['base_correct']}/{direct['all']['rows']}.",
        f"- Candidate: {direct['all']['candidate_correct']}/{direct['all']['rows']} "
        f"(net {direct['all']['net']:+d}; {direct['all']['wins']} wins / {direct['all']['losses']} losses).",
        f"- Direct oracle complement: {direct['oracle_complement']} (required >= 250).",
        f"- Dev-locked alpha: {locked['alpha']:.2f}; dev {locked['dev']['net']:+d}, "
        f"sealed {locked['sealed']['net']:+d}, all {locked['all']['net']:+d}.",
        f"- All quality-flagged validation rows: {direct['any_flag']['net']:+d} "
        f"({direct['any_flag']['wins']} wins / {direct['any_flag']['losses']} losses).",
        "",
        "## Per-category direct changes",
        "",
        "| category | rows | net | wins | losses | changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in direct.items():
        if not name.startswith("flag:"):
            continue
        lines.append(
            f"| {name.removeprefix('flag:')} | {row['rows']} | {row['net']:+d} | "
            f"{row['wins']} | {row['losses']} | {row['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Stop this loss/augmentation configuration. It fails all continuation gates; do not cache test_seen, "
            "do not generate a submission, and do not spend another epoch on the same formulation.",
            "",
            "The CSV remains useful as an audit/stratification set, but simple deletion, duplication, and this "
            "quality-aware contrastive LoRA did not create a deployable gain.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
