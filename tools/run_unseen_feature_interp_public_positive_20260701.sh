#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/unseen_feature_interp_public_positive_20260701}
mkdir -p "$OUT_ROOT/features"

TAXON=work/clip_text_features/all_bioclip25_taxon.pt

make_feature() {
  local split=$1
  local weight_name=$2
  local weights=$3
  local out=$4
  [[ -f "$out" ]] && return 0
  if [[ "$split" == species ]]; then
    local orig=runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt
    local adapt=runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_species_adapted.pt
  elif [[ "$split" == genus ]]; then
    local orig=runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt
    local adapt=runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_genus_adapted.pt
  else
    local orig=runs/bioclip25_image_cache_3090/test_unseen_hflip.pt
    local adapt=runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/test_unseen_adapted.pt
  fi
  python3 tools/average_feature_files.py --inputs "$orig,$adapt" --weights "$weights" --out "$out"
}

run_eval() {
  local name=$1
  local img=$2
  local cand=$3
  local out=$4
  [[ -f "$out/summary.json" ]] && return 0
  python3 tools/transductive_active_sinkhorn.py \
    --image-features "$img" \
    --text-features "$TAXON" \
    --candidate-classes "$cand" \
    --out-dir "$out" \
    --score-batch-size 256 \
    --active-count-grid 11598 \
    --active-mode-grid max \
    --union-topk-grid 0 \
    --tau-grid 0.015,0.02,0.025,0.03 \
    --blend-grid 3,5,7 \
    --prior-mode-grid logsumexp \
    --prior-alpha-grid 0.25,0.5,0.75 \
    --prior-uniform-mix-grid 0.93,0.95,0.97
}

weights=(
  "orig95_adapt05|0.95,0.05"
  "orig90_adapt10|0.90,0.10"
  "orig80_adapt20|0.80,0.20"
  "orig70_adapt30|0.70,0.30"
)

for pair in "${weights[@]}"; do
  IFS='|' read -r name w <<<"$pair"
  species_feat="$OUT_ROOT/features/${name}_pseudo_species.pt"
  genus_feat="$OUT_ROOT/features/${name}_pseudo_genus.pt"
  make_feature species "$name" "$w" "$species_feat"
  make_feature genus "$name" "$w" "$genus_feat"
  for seed in 43 44; do
    run_eval "$name" "$species_feat" "work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json" "$OUT_ROOT/${name}_species${seed}"
    run_eval "$name" "$genus_feat" "work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json" "$OUT_ROOT/${name}_genus${seed}"
  done
done

python3 - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("runs/unseen_feature_interp_public_positive_20260701/*/summary.json")):
    name = os.path.basename(os.path.dirname(f))
    for suffix in ["species43", "species44", "genus43", "genus44"]:
        if name.endswith("_" + suffix):
            config = name[:-(len(suffix)+1)]
            split = suffix
            break
    else:
        continue
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
        "tau": best["tau"],
        "blend": best["blend"],
        "prior_alpha": best["prior_alpha"],
        "prior_uniform_mix": best["prior_uniform_mix"],
    })
with open("runs/unseen_feature_interp_public_positive_20260701/best_by_split.csv", "w", newline="", encoding="utf-8") as fp:
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
with open("runs/unseen_feature_interp_public_positive_20260701/aggregate_summary.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
PY
