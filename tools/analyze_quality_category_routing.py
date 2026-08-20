from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def paired_stats(base_prediction, prediction, labels, mask):
    base_correct = base_prediction.eq(labels)
    candidate_correct = prediction.eq(labels)
    changed = mask & prediction.ne(base_prediction)
    wins = changed & ~base_correct & candidate_correct
    losses = changed & base_correct & ~candidate_correct
    return {
        "rows": int(mask.sum()),
        "base_correct": int((mask & base_correct).sum()),
        "candidate_correct": int((mask & candidate_correct).sum()),
        "net": int(wins.sum() - losses.sum()),
        "changed": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "efficiency": float((wins.sum() - losses.sum()) / max(1, int(changed.sum()))),
    }


def load_flags(path: str, image_ids: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    index = {image_id: row for row, image_id in enumerate(image_ids)}
    categories_by_row: dict[int, list[str]] = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            image_id = item["image_id"]
            if image_id not in index:
                continue
            categories = [
                part.strip()
                for part in item["categories"].replace("；", ";").replace("|", ";").split(";")
                if part.strip()
            ]
            for category in categories:
                categories_by_row[index[image_id]].append(category)
    masks = {}
    rows_by_category = {}
    for category in sorted({c for values in categories_by_row.values() for c in values}):
        mask = torch.zeros(len(image_ids), dtype=torch.bool)
        rows = []
        for row, categories in categories_by_row.items():
            if category in categories:
                mask[row] = True
                rows.append(image_ids[row])
        masks[category] = mask
        rows_by_category[category] = rows
    any_mask = torch.zeros(len(image_ids), dtype=torch.bool)
    for row in categories_by_row:
        any_mask[row] = True
    masks["ANY_FLAG"] = any_mask
    rows_by_category["ANY_FLAG"] = [image_ids[row] for row in categories_by_row]
    return masks, rows_by_category


def select_blend_for_mask(
    *,
    base_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    base_prediction: torch.Tensor,
    labels: torch.Tensor,
    route_mask: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    error_gate: torch.Tensor,
    variant_name: str,
    category: str,
) -> tuple[torch.Tensor, dict]:
    alpha_values = [round(x / 100, 2) for x in range(4, 101, 4)]
    gate_values = torch.quantile(
        error_gate.float(),
        torch.tensor([0.00, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]),
    ).unique().tolist()
    gate_values = [0.0] + [float(x) for x in gate_values]
    best_key = None
    best_record = None
    best_prediction = base_prediction.clone()

    for alpha in alpha_values:
        blended = base_logits.float().mul(1.0 - alpha).add(variant_logits.float(), alpha=alpha)
        candidate_prediction = blended.argmax(dim=1).cpu()
        changed = candidate_prediction.ne(base_prediction)
        for gate_threshold in gate_values:
            use = route_mask & changed & error_gate.float().ge(float(gate_threshold))
            prediction = base_prediction.clone()
            prediction[use] = candidate_prediction[use]
            dev_stats = paired_stats(base_prediction, prediction, labels, dev)
            record = {
                "name": f"{category}:{variant_name}:alpha{alpha:.2f}:gate{gate_threshold:.4f}",
                "category": category,
                "variant": variant_name,
                "alpha": float(alpha),
                "gate_threshold": float(gate_threshold),
                "dev": dev_stats,
            }
            key = (dev_stats["net"], -dev_stats["changed"])
            if best_key is None or key > best_key:
                best_key = key
                best_record = record
                best_prediction = prediction

    assert best_record is not None
    best_record = dict(best_record)
    best_record["all"] = paired_stats(base_prediction, best_prediction, labels, torch.ones_like(dev))
    best_record["dev"] = paired_stats(base_prediction, best_prediction, labels, dev)
    best_record["sealed"] = paired_stats(base_prediction, best_prediction, labels, sealed)
    best_record["route_all"] = paired_stats(base_prediction, best_prediction, labels, route_mask)
    best_record["route_dev"] = paired_stats(base_prediction, best_prediction, labels, route_mask & dev)
    best_record["route_sealed"] = paired_stats(base_prediction, best_prediction, labels, route_mask & sealed)
    return best_prediction, best_record


def combine_category_rules(
    base_prediction: torch.Tensor,
    selected: list[tuple[torch.Tensor, dict]],
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    prediction = base_prediction.clone()
    # Apply more reliable dev-net rules first; only fill still-base rows.
    ordered = sorted(selected, key=lambda item: (item[1]["dev"]["net"], item[1]["sealed"]["net"]), reverse=True)
    applied = []
    for pred, record in ordered:
        changed = pred.ne(base_prediction)
        fill = changed & prediction.eq(base_prediction)
        if bool(fill.any()):
            prediction[fill] = pred[fill]
            applied.append(
                {
                    "name": record["name"],
                    "filled": int(fill.sum()),
                    "category": record["category"],
                    "variant": record["variant"],
                    "alpha": record["alpha"],
                    "gate_threshold": record["gate_threshold"],
                }
            )
    return prediction, {
        "name": "category_rules_fill",
        "applied": applied,
        "all": paired_stats(base_prediction, prediction, labels, torch.ones_like(dev)),
        "dev": paired_stats(base_prediction, prediction, labels, dev),
        "sealed": paired_stats(base_prediction, prediction, labels, sealed),
    }


def combine_named(
    base_prediction: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> list[dict]:
    records = []
    keys = list(predictions.keys())
    for first in keys:
        for second in keys:
            if first == second:
                continue
            output = base_prediction.clone()
            first_changed = predictions[first].ne(base_prediction)
            second_changed = predictions[second].ne(base_prediction)
            output[first_changed] = predictions[first][first_changed]
            fill = ~first_changed & second_changed
            output[fill] = predictions[second][fill]
            records.append(
                {
                    "name": f"{first}_then_{second}",
                    "all": paired_stats(base_prediction, output, labels, torch.ones_like(dev)),
                    "dev": paired_stats(base_prediction, output, labels, dev),
                    "sealed": paired_stats(base_prediction, output, labels, sealed),
                }
            )
    records.sort(key=lambda item: (item["dev"]["net"], item["sealed"]["net"]), reverse=True)
    return records


def write_report(path: Path, results: dict):
    lines = [
        "# 2026-08-07 Quality Category Routing",
        "",
        "## Result",
        "",
        (
            f"- Best category-fill route: all `{results['category_fill']['all']['net']:+d}`, "
            f"dev `{results['category_fill']['dev']['net']:+d}`, "
            f"sealed `{results['category_fill']['sealed']['net']:+d}`, "
            f"changed `{results['category_fill']['all']['changed']}`."
        ),
        (
            f"- Prior error-gated crossfit: all `{results['crossfit_gate']['all']['net']:+d}`, "
            f"dev `{results['crossfit_gate']['dev']['net']:+d}`, "
            f"sealed `{results['crossfit_gate']['sealed']['net']:+d}`."
        ),
        "",
        "No `test_seen` inference or submission was produced.",
        "",
        "## Selected category rules",
        "",
        "| Category | Variant | Alpha | Gate | Route Dev | Route Sealed | All Net | Changed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results["selected_records"]:
        lines.append(
            f"| {item['category']} | {item['variant']} | {item['alpha']:.2f} | "
            f"{item['gate_threshold']:.4f} | {item['route_dev']['net']:+d} | "
            f"{item['route_sealed']['net']:+d} | {item['all']['net']:+d} | "
            f"{item['all']['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Best single rules",
            "",
            "| Rule | Dev | Sealed | All | Changed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in results["all_records"][:20]:
        lines.append(
            f"| {item['name']} | {item['dev']['net']:+d} | {item['sealed']['net']:+d} | "
            f"{item['all']['net']:+d} | {item['all']['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Combination diagnostics",
            "",
            "| Rule | Dev | Sealed | All | Changed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in results["combination_records"][:12]:
        lines.append(
            f"| {item['name']} | {item['dev']['net']:+d} | {item['sealed']['net']:+d} | "
            f"{item['all']['net']:+d} | {item['all']['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            results["decision"],
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{results['results_json']}`",
            f"- Predictions: `{results['prediction_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-logits", default="runs/local_20260803_strong_oof_rebuild/joint_reconstruction_exact_verification/reconstructed_val_logits.pt")
    parser.add_argument("--candidate-bank", default="runs/local_20260807_seen_candidate_bank_fusion/candidate_bank_scores.pt")
    parser.add_argument("--gate-outputs", default="runs/local_20260807_error_quality_gate_scout/gate_outputs.pt")
    parser.add_argument("--flags-val", default="runs/local_20260803_quality_csv_scout/flags_val.csv")
    parser.add_argument("--quality-head-root", default="runs/local_20260803_quality_csv_scout/head_models")
    parser.add_argument("--out-dir", default="runs/local_20260807_quality_category_routing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_logits_obj = torch.load(args.base_logits, map_location="cpu", weights_only=False)
    bank = torch.load(args.candidate_bank, map_location="cpu", weights_only=False)
    gate = torch.load(args.gate_outputs, map_location="cpu", weights_only=False)
    base_logits = base_logits_obj["logits"].float()
    labels = bank["labels"].long()
    image_ids = list(bank["image_ids"])
    dev = bank["dev"].bool()
    sealed = bank["sealed"].bool()
    base_prediction = bank["top_indices"][:, 0].long()
    error_gate = gate["val_error_gate"].float()

    if list(base_logits_obj["image_ids"]) != image_ids:
        raise RuntimeError("base logits and candidate bank order differ")

    masks, _rows = load_flags(args.flags_val, image_ids)
    head_root = Path(args.quality_head_root)
    variants = ["control_all", "exclude_all_flags", "exclude_hard3", "oversample_hard3_x2"]

    all_records = []
    predictions_by_record = {}
    for category, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        for variant in variants:
            path = head_root / variant / "val_logits.pt"
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if list(obj["image_ids"]) != image_ids:
                raise RuntimeError(f"{variant} logits order differs")
            prediction, record = select_blend_for_mask(
                base_logits=base_logits,
                variant_logits=obj["logits"].float(),
                base_prediction=base_prediction,
                labels=labels,
                route_mask=mask,
                dev=dev,
                sealed=sealed,
                error_gate=error_gate,
                variant_name=variant,
                category=category,
            )
            all_records.append(record)
            predictions_by_record[record["name"]] = prediction

    all_records.sort(key=lambda item: (item["dev"]["net"], item["sealed"]["net"]), reverse=True)

    # Keep category-specific rules that are positive on route-dev, and require
    # non-negative sealed where available to avoid the obvious overfit cases.
    selected = []
    used_categories = set()
    for record in all_records:
        category = record["category"]
        if category == "ANY_FLAG" or category in used_categories:
            continue
        if record["route_dev"]["net"] > 0 and record["route_sealed"]["net"] >= 0:
            selected.append((predictions_by_record[record["name"]], record))
            used_categories.add(category)

    category_prediction, category_fill = combine_category_rules(
        base_prediction, selected, labels, dev, sealed
    )

    crossfit_gate = gate["crossfit_gated_prediction"].long()
    consensus_gate = gate["consensus_gated_prediction"].long()
    crossfit_stats = {
        "all": paired_stats(base_prediction, crossfit_gate, labels, torch.ones_like(dev)),
        "dev": paired_stats(base_prediction, crossfit_gate, labels, dev),
        "sealed": paired_stats(base_prediction, crossfit_gate, labels, sealed),
    }
    combination_records = combine_named(
        base_prediction,
        labels,
        dev,
        sealed,
        {
            "category_fill": category_prediction,
            "crossfit_gate": crossfit_gate,
            "consensus_gate": consensus_gate,
        },
    )

    best_combo = combination_records[0] if combination_records else None
    if category_fill["all"]["net"] >= 120 and category_fill["sealed"]["net"] > 0:
        decision = "Continue: category routing passed the local continuation gate."
    elif best_combo is not None and best_combo["all"]["net"] >= 120 and best_combo["sealed"]["net"] > 0:
        decision = "Continue only after stricter validation; a combination passed the nominal gate."
    else:
        decision = (
            "Stop this branch for test inference. Manual quality categories help only a small "
            "flagged subset and do not materially improve the current seen solution."
        )

    prediction_path = out_dir / "predictions.pt"
    torch.save(
        {
            "category_prediction": category_prediction,
            "crossfit_gate": crossfit_gate,
            "consensus_gate": consensus_gate,
            "base_prediction": base_prediction,
            "labels": labels,
            "dev": dev,
            "sealed": sealed,
            "selected_record_names": [record["name"] for _pred, record in selected],
        },
        prediction_path,
    )

    results = {
        "category_fill": category_fill,
        "crossfit_gate": crossfit_stats,
        "selected_records": [record for _pred, record in selected],
        "all_records": all_records,
        "combination_records": combination_records,
        "decision": decision,
        "results_json": str(out_dir / "results.json"),
        "prediction_path": str(prediction_path),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(out_dir / "EXPERIMENT_REPORT.md", results)
    print(
        json.dumps(
            {
                "category_fill": category_fill,
                "crossfit_gate": crossfit_stats,
                "best_combo": best_combo,
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
