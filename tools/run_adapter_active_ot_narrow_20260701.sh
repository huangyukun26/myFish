#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/adapter_active_ot_narrow_20260701}
mkdir -p "$OUT_ROOT"
TEXT=${TEXT:-work/clip_text_features/all_bioclip25_taxon.pt}

run_eval() {
  local name=$1
  local species_feat=$2
  local genus_feat=$3
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  for split in species43 species44; do
    local seed=${split#species}
    local cand="work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json"
    local eval_dir="$out_dir/eval_${split}"
    [[ -f "$eval_dir/summary.json" ]] || python3 tools/transductive_active_sinkhorn.py \
      --image-features "$species_feat" \
      --text-features "$TEXT" \
      --candidate-classes "$cand" \
      --out-dir "$eval_dir" \
      --score-batch-size 256 \
      --active-count-grid 4000,6000,9000,11598 \
      --active-mode-grid max,logsumexp \
      --union-topk-grid 0,2 \
      --tau-grid 0.035,0.04 \
      --blend-grid 1,2 \
      --prior-mode-grid logsumexp \
      --prior-alpha-grid 0.25,0.5 \
      --prior-uniform-mix-grid 0.95
  done
  for split in genus43 genus44; do
    local seed=${split#genus}
    local cand="work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json"
    local eval_dir="$out_dir/eval_${split}"
    [[ -f "$eval_dir/summary.json" ]] || python3 tools/transductive_active_sinkhorn.py \
      --image-features "$genus_feat" \
      --text-features "$TEXT" \
      --candidate-classes "$cand" \
      --out-dir "$eval_dir" \
      --score-batch-size 256 \
      --active-count-grid 4000,6000,9000,11598 \
      --active-mode-grid max,logsumexp \
      --union-topk-grid 0,2 \
      --tau-grid 0.035,0.04 \
      --blend-grid 1,2 \
      --prior-mode-grid logsumexp \
      --prior-alpha-grid 0.25,0.5 \
      --prior-uniform-mix-grid 0.95
  done
}

run_eval id20_ep30 \
  runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_species_adapted.pt \
  runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_genus_adapted.pt

run_eval ens_ep30_id30_w50 \
  runs/global_visual_adapter_ensemble_20260630/ep30_id30_w50/pseudo_species_adapted.pt \
  runs/global_visual_adapter_ensemble_20260630/ep30_id30_w50/pseudo_genus_adapted.pt

python3 - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("runs/adapter_active_ot_narrow_20260701/*/eval_*/summary.json")):
    config = f.split("/")[-3]
    split = os.path.basename(os.path.dirname(f)).replace("eval_", "")
    best = json.load(open(f, encoding="utf-8"))["best"]
    rows.append({
        "config": config,
        "split": split,
        "top1": best["top1"],
        "base_top1": best["base_top1"],
        "gain": best["top1"] - best["base_top1"],
        "net": best["net"],
        "wins": best["wins"],
        "losses": best["losses"],
        "changed": best["changed"],
        "active_count": best["active_count"],
        "active_actual": best["active_actual"],
        "active_mode": best["active_mode"],
        "union_topk": best["union_topk"],
        "tau": best["tau"],
        "blend": best["blend"],
        "prior_alpha": best["prior_alpha"],
        "prior_uniform_mix": best["prior_uniform_mix"],
    })
with open("runs/adapter_active_ot_narrow_20260701/best_by_split.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = []
for config in sorted({r["config"] for r in rows}):
    group = [r for r in rows if r["config"] == config]
    summary.append({
        "config": config,
        "avg_top1": sum(r["top1"] for r in group) / len(group),
        "min_gain": min(r["gain"] for r in group),
        "avg_net": sum(r["net"] for r in group) / len(group),
        "min_net": min(r["net"] for r in group),
        "wins": sum(r["wins"] for r in group),
        "losses": sum(r["losses"] for r in group),
        "changed": sum(r["changed"] for r in group),
    })
summary.sort(key=lambda r: (r["avg_top1"], r["min_gain"], r["avg_net"]), reverse=True)
with open("runs/adapter_active_ot_narrow_20260701/aggregate_summary.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
PY
