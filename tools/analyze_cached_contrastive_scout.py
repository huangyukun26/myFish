from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


CURRENT_BASE_CORRECT = {"all": 9823, "dev": 6662, "sealed": 3161}


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (image_id, cls) in enumerate(zip(image_ids, y.tolist())):
        groups[int(cls)].append((stable_hash(image_id), idx))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for position, (_digest, idx) in enumerate(rows):
            if position % 5 in {0, 1, 2}:
                dev[idx] = True
    return dev, ~dev


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def average_run_logits(run_names: list[str], arms: dict[str, Any]) -> torch.Tensor:
    average = None
    for name in run_names:
        logits = load(Path(arms[name]["path"]) / "val_logits.pt")["logits"].float()
        if average is None:
            average = logits
        else:
            average.add_(logits)
    if average is None:
        raise RuntimeError("Cannot average an empty run family")
    return average.div_(len(run_names))


def paired_prediction_stats(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    base_correct = base_prediction.eq(y)
    candidate_correct = candidate_prediction.eq(y)
    changed = mask & base_prediction.ne(candidate_prediction)
    wins = changed & ~base_correct & candidate_correct
    losses = changed & base_correct & ~candidate_correct
    return {
        "rows": int(mask.sum()),
        "base_correct": int((base_correct & mask).sum()),
        "candidate_correct": int((candidate_correct & mask).sum()),
        "raw_net": int((candidate_correct & mask).sum() - (base_correct & mask).sum()),
        "changed": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "efficiency": float((wins.sum() - losses.sum()) / max(1, int(changed.sum()))),
        "oracle_complement": int((~base_correct & candidate_correct & mask).sum()),
    }


def direct_metrics(
    prediction: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    top5 = logits.topk(5, dim=1).indices
    output = {}
    for name, mask in masks.items():
        correct = int(prediction[mask].eq(y[mask]).sum())
        top5_correct = int(top5[mask].eq(y[mask, None]).any(dim=1).sum())
        output[name] = {
            "rows": int(mask.sum()),
            "correct": correct,
            "top1": correct / max(1, int(mask.sum())),
            "top5": top5_correct / max(1, int(mask.sum())),
            "raw_net_vs_current_base_count": correct - CURRENT_BASE_CORRECT[name],
        }
    return output


def same_genus_error_metrics(
    prediction: torch.Tensor,
    y: torch.Tensor,
    classes: list[str],
) -> dict[str, Any]:
    genera = [name.split(maxsplit=1)[0] for name in classes]
    wrong = prediction.ne(y)
    same_genus = torch.tensor(
        [
            genera[int(pred)] == genera[int(target)]
            for pred, target in zip(prediction.tolist(), y.tolist())
        ],
        dtype=torch.bool,
    )
    same_genus_wrong = wrong & same_genus
    return {
        "errors": int(wrong.sum()),
        "same_genus_errors": int(same_genus_wrong.sum()),
        "same_genus_error_fraction": float(
            same_genus_wrong.sum() / max(1, int(wrong.sum()))
        ),
    }


def frequency_metrics(
    prediction: torch.Tensor,
    y: torch.Tensor,
    full_counts: torch.Tensor,
) -> dict[str, Any]:
    buckets = {
        "count_2": full_counts[y].eq(2),
        "count_3_5": full_counts[y].ge(3) & full_counts[y].le(5),
        "count_6_10": full_counts[y].ge(6) & full_counts[y].le(10),
        "count_11_20": full_counts[y].ge(11) & full_counts[y].le(20),
        "count_21_plus": full_counts[y].ge(21),
    }
    output = {}
    for name, mask in buckets.items():
        rows = int(mask.sum())
        output[name] = {
            "rows": rows,
            "correct": int(prediction[mask].eq(y[mask]).sum()),
            "top1": float(prediction[mask].eq(y[mask]).float().mean()) if rows else 0.0,
        }
    return output


def standardize_on_device(logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = logits.float().to(device)
    value = value - value.mean(dim=1, keepdim=True)
    return value / value.std(dim=1, keepdim=True).clamp_min(1e-6)


def ensemble_scan(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    y: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = standardize_on_device(reference_logits, device)
    candidate = standardize_on_device(candidate_logits, device)
    y_device = y.to(device)
    dev_device = dev.to(device)
    trials = []
    alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    for topk in [5, 10, 20, 0]:
        if topk:
            reference_values, reference_indices = reference.topk(topk, dim=1)
            candidate_values = candidate.gather(1, reference_indices)
        for alpha in alphas:
            if topk:
                local = (reference_values + alpha * candidate_values).argmax(dim=1)
                prediction = reference_indices.gather(1, local[:, None]).squeeze(1)
            else:
                prediction = (reference + alpha * candidate).argmax(dim=1)
            trials.append(
                {
                    "topk": topk,
                    "alpha": alpha,
                    "dev_correct": int(prediction[dev_device].eq(y_device[dev_device]).sum()),
                }
            )
    best = max(
        trials,
        key=lambda row: (
            row["dev_correct"],
            -row["alpha"],
            -(row["topk"] if row["topk"] else 10_000),
        ),
    )
    topk = best["topk"]
    alpha = best["alpha"]
    if topk:
        reference_values, reference_indices = reference.topk(topk, dim=1)
        candidate_values = candidate.gather(1, reference_indices)
        local = (reference_values + alpha * candidate_values).argmax(dim=1)
        prediction = reference_indices.gather(1, local[:, None]).squeeze(1)
    else:
        prediction = (reference + alpha * candidate).argmax(dim=1)
    prediction = prediction.cpu()
    reference_prediction = reference_logits.argmax(dim=1)
    masks = {
        "all": torch.ones(len(y), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    return {
        "selection": "topk and alpha chosen only by deterministic dev correct count",
        "best_by_dev": best,
        "paired_vs_reference": {
            name: paired_prediction_stats(
                reference_prediction,
                prediction,
                y,
                mask,
            )
            for name, mask in masks.items()
        },
        "raw_vs_current_base_count": {
            name: int(prediction[mask].eq(y[mask]).sum()) - CURRENT_BASE_CORRECT[name]
            for name, mask in masks.items()
        },
        "trials": trials,
    }


def score_bank(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reference_top2 = reference_logits.topk(2, dim=1)
    candidate_top2 = candidate_logits.topk(2, dim=1)
    reference_margin = reference_top2.values[:, 0] - reference_top2.values[:, 1]
    candidate_margin = candidate_top2.values[:, 0] - candidate_top2.values[:, 1]
    return {
        "candidate_margin": candidate_margin,
        "delta_margin": candidate_margin - reference_margin,
        "low_reference_margin": -reference_margin,
        "hybrid": candidate_margin - 0.5 * reference_margin,
    }


def selector_scan(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    y: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> dict[str, Any]:
    reference_prediction = reference_logits.argmax(dim=1)
    candidate_prediction = candidate_logits.argmax(dim=1)
    changed = reference_prediction.ne(candidate_prediction)
    reference_correct = reference_prediction.eq(y)
    candidate_correct = candidate_prediction.eq(y)
    scores = score_bank(reference_logits.float(), candidate_logits.float())
    trials = []
    for score_name, score in scores.items():
        dev_changed_indices = torch.where(dev & changed)[0]
        for target in [25, 50, 100, 200, 400, 800]:
            count = min(target, len(dev_changed_indices))
            if count == 0:
                continue
            threshold = score[dev_changed_indices].topk(count).values[-1]
            selected = changed & score.ge(threshold)
            dev_selected = selected & dev
            wins = dev_selected & ~reference_correct & candidate_correct
            losses = dev_selected & reference_correct & ~candidate_correct
            trials.append(
                {
                    "score": score_name,
                    "target_dev": target,
                    "threshold": float(threshold),
                    "dev_selected": int(dev_selected.sum()),
                    "dev_net": int(wins.sum() - losses.sum()),
                }
            )
    best = max(
        trials,
        key=lambda row: (row["dev_net"], -row["dev_selected"], row["score"]),
    )
    selected = changed & scores[best["score"]].ge(best["threshold"])
    output = {}
    for name, mask in {
        "all": torch.ones(len(y), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }.items():
        use = selected & mask
        prediction = reference_prediction.clone()
        prediction[use] = candidate_prediction[use]
        output[name] = paired_prediction_stats(
            reference_prediction,
            prediction,
            y,
            mask,
        )
        output[name]["raw_net_vs_current_base_count"] = (
            int(prediction[mask].eq(y[mask]).sum()) - CURRENT_BASE_CORRECT[name]
        )
    return {
        "selection": "one score family and coverage chosen on deterministic dev only",
        "best_by_dev": best,
        "locked": output,
        "trials": trials,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Local Cached Contrastive Scout Analysis",
        "",
        "All models use the fixed epoch-12 checkpoint. Current-base comparisons use only",
        "the historical aggregate correct counts because exact row-level base logits are absent locally.",
        "",
        "## Fixed-Epoch Results",
        "",
        "| arm | correct | top1 | raw net vs current count | sealed net vs current count | oracle vs DINO-L reference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["arms"].items():
        fixed = row["direct"]["all"]
        sealed = row["direct"]["sealed"]
        oracle = row["paired_vs_reference"]["all"]["oracle_complement"]
        lines.append(
            f"| {name} | {fixed['correct']} | {fixed['top1']:.6f} | "
            f"{fixed['raw_net_vs_current_base_count']:+d} | "
            f"{sealed['raw_net_vs_current_base_count']:+d} | {oracle} |"
        )
    lines.extend(
        [
            "",
            "## Dev-Locked Ensembles Versus DINO-L Reference",
            "",
            "| arm | topk | alpha | dev net | sealed net | all net | all raw vs current count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["arms"].items():
        ensemble = row["ensemble_vs_reference"]
        best = ensemble["best_by_dev"]
        paired = ensemble["paired_vs_reference"]
        lines.append(
            f"| {name} | {best['topk']} | {best['alpha']} | "
            f"{paired['dev']['raw_net']:+d} | {paired['sealed']['raw_net']:+d} | "
            f"{paired['all']['raw_net']:+d} | "
            f"{ensemble['raw_vs_current_base_count']['all']:+d} |"
        )
    if report.get("family_ensembles"):
        lines.extend(
            [
                "",
                "## Fixed Two-Seed Family Averages",
                "",
                "| family | correct | net vs CE family | dev net vs CE | sealed net vs CE | dev-locked net vs reference | sealed net vs reference |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in report["family_ensembles"].items():
            direct = row["direct"]["all"]
            paired_ce = row["paired_vs_ce_family"]
            locked = row["ensemble_vs_reference"]["paired_vs_reference"]
            lines.append(
                f"| {name} | {direct['correct']} | "
                f"{paired_ce['all']['raw_net']:+d} | "
                f"{paired_ce['dev']['raw_net']:+d} | "
                f"{paired_ce['sealed']['raw_net']:+d} | "
                f"{locked['all']['raw_net']:+d} | "
                f"{locked['sealed']['raw_net']:+d} |"
            )
    lines.extend(
        [
            "",
            "The DINO-L reference is the returned frozen+adapted joint candidate, not the exact",
            "current 0.780 public base.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--full-count-cache", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference_logits)
    reference_logits = reference["logits"].float()
    y = reference["class_ids"].long()
    image_ids = list(reference["image_ids"])
    classes = list(reference["classes"])
    dev, sealed = split_dev_sealed(image_ids, y)
    masks = {
        "all": torch.ones(len(y), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    counts_payload = load(args.full_count_cache)
    full_counts = counts_payload["full_class_counts"].long()

    reference_prediction = reference_logits.argmax(dim=1)
    report: dict[str, Any] = {
        "reference": {
            "path": str(args.reference_logits),
            "direct": direct_metrics(
                reference_prediction,
                reference_logits,
                y,
                masks,
            ),
            "same_genus_errors": same_genus_error_metrics(
                reference_prediction,
                y,
                classes,
            ),
            "frequency": frequency_metrics(reference_prediction, y, full_counts),
        },
        "current_base_known_counts": CURRENT_BASE_CORRECT,
        "limitations": [
            "Exact current-base row-level logits are absent locally.",
            "Raw net versus current base is candidate correct minus a known aggregate count.",
            "Paired wins/losses/oracle and ensembles use the stronger returned DINO-L joint reference.",
        ],
        "arms": {},
    }

    run_dirs = sorted(
        path
        for path in args.run_root.iterdir()
        if path.is_dir() and path.name.startswith("arm_") and (path / "summary.json").exists()
    )
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        arm_name = run_dir.name
        candidate = load(run_dir / "val_logits.pt")
        if list(candidate["image_ids"]) != image_ids:
            raise RuntimeError(f"Image order mismatch for {run_dir}")
        candidate_logits = candidate["logits"].float()
        candidate_prediction = candidate_logits.argmax(dim=1)
        paired = {
            name: paired_prediction_stats(
                reference_prediction,
                candidate_prediction,
                y,
                mask,
            )
            for name, mask in masks.items()
        }
        history = summary["history"]
        trajectory = {
            "best_all": max(
                (
                    {
                        "epoch": row["epoch"],
                        **row["validation"]["all"],
                    }
                    for row in history
                ),
                key=lambda row: row["correct"],
            ),
            "best_dev": max(
                (
                    {
                        "epoch": row["epoch"],
                        **row["validation"]["dev"],
                        "sealed_at_epoch": row["validation"]["sealed"],
                    }
                    for row in history
                ),
                key=lambda row: row["correct"],
            ),
        }
        report["arms"][arm_name] = {
            "path": str(run_dir),
            "loss_weights": summary["loss_weights"],
            "fixed_epoch": summary["history"][-1]["epoch"],
            "direct": direct_metrics(candidate_prediction, candidate_logits, y, masks),
            "paired_vs_reference": paired,
            "same_genus_errors": same_genus_error_metrics(
                candidate_prediction,
                y,
                classes,
            ),
            "frequency": frequency_metrics(candidate_prediction, y, full_counts),
            "trajectory_diagnostic_only": trajectory,
            "ensemble_vs_reference": ensemble_scan(
                reference_logits,
                candidate_logits,
                y,
                dev,
                sealed,
            ),
            "selector_vs_reference": selector_scan(
                reference_logits,
                candidate_logits,
                y,
                dev,
                sealed,
            ),
        }

    controls_by_seed = {
        match.group(1): name
        for name in report["arms"]
        if name.startswith("arm_a_ce_")
        and (match := re.search(r"_seed(\d+)_", name)) is not None
    }
    fallback_control = next(iter(controls_by_seed.values()), None)
    for name, row in report["arms"].items():
        match = re.search(r"_seed(\d+)_", name)
        control_name = (
            controls_by_seed.get(match.group(1))
            if match is not None
            else fallback_control
        )
        if control_name is not None:
            control = load(
                Path(report["arms"][control_name]["path"]) / "val_logits.pt"
            )
            control_prediction = control["logits"].argmax(dim=1)
            candidate = load(Path(row["path"]) / "val_logits.pt")
            candidate_prediction = candidate["logits"].argmax(dim=1)
            row["ce_control"] = control_name
            row["paired_vs_ce_control"] = {
                split: paired_prediction_stats(
                    control_prediction,
                    candidate_prediction,
                    y,
                    mask,
                )
                for split, mask in masks.items()
            }

    grouped_runs: dict[str, list[str]] = defaultdict(list)
    for name in report["arms"]:
        family = re.sub(r"_seed\d+_e\d+$", "", name)
        grouped_runs[family].append(name)
    report["family_ensembles"] = {}
    for family, members in sorted(grouped_runs.items()):
        if len(members) < 2:
            continue
        family_logits = average_run_logits(sorted(members), report["arms"])
        family_prediction = family_logits.argmax(dim=1)
        report["family_ensembles"][family] = {
            "members": sorted(members),
            "aggregation": "arithmetic mean of fixed-epoch logits; no validation selection",
            "direct": direct_metrics(family_prediction, family_logits, y, masks),
            "paired_vs_reference": {
                split: paired_prediction_stats(
                    reference_prediction,
                    family_prediction,
                    y,
                    mask,
                )
                for split, mask in masks.items()
            },
            "same_genus_errors": same_genus_error_metrics(
                family_prediction,
                y,
                classes,
            ),
            "frequency": frequency_metrics(family_prediction, y, full_counts),
            "ensemble_vs_reference": ensemble_scan(
                reference_logits,
                family_logits,
                y,
                dev,
                sealed,
            ),
        }

    ce_family_name = next(
        (
            family
            for family in report["family_ensembles"]
            if family.startswith("arm_a_ce")
        ),
        None,
    )
    if ce_family_name is not None:
        ce_members = report["family_ensembles"][ce_family_name]["members"]
        ce_family_prediction = average_run_logits(
            ce_members,
            report["arms"],
        ).argmax(dim=1)
        for row in report["family_ensembles"].values():
            family_prediction = average_run_logits(
                row["members"],
                report["arms"],
            ).argmax(dim=1)
            row["paired_vs_ce_family"] = {
                split: paired_prediction_stats(
                    ce_family_prediction,
                    family_prediction,
                    y,
                    mask,
                )
                for split, mask in masks.items()
            }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
