#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/global_visual_adapter_ensemble_20260630}
mkdir -p "$OUT_ROOT"

TEXT=${TEXT:-work/clip_text_features/all_bioclip25_taxon.pt}

make_ensemble() {
  local name=$1
  local weights=$2
  local species_inputs=$3
  local genus_inputs=$4
  local public_inputs=${5:-}
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  [[ -f "$dir/pseudo_species_adapted.pt" ]] || python3 tools/average_feature_files.py --inputs "$species_inputs" --weights "$weights" --out "$dir/pseudo_species_adapted.pt" >/dev/null
  [[ -f "$dir/pseudo_genus_adapted.pt" ]] || python3 tools/average_feature_files.py --inputs "$genus_inputs" --weights "$weights" --out "$dir/pseudo_genus_adapted.pt" >/dev/null
  if [[ -n "$public_inputs" ]]; then
    [[ -f "$dir/test_unseen_adapted.pt" ]] || python3 tools/average_feature_files.py --inputs "$public_inputs" --weights "$weights" --out "$dir/test_unseen_adapted.pt" >/dev/null
  fi
}

run_eval_one() {
  local feature_path=$1
  local cand_path=$2
  local out_dir=$3
  local blend=$4
  local alpha=$5
  local mix=$6
  python3 tools/transductive_active_sinkhorn.py \
    --image-features "$feature_path" \
    --text-features "$TEXT" \
    --candidate-classes "$cand_path" \
    --out-dir "$out_dir" \
    --score-batch-size 256 \
    --active-count-grid 11598 \
    --active-mode-grid max \
    --union-topk-grid 0 \
    --tau-grid 0.04 \
    --blend-grid "$blend" \
    --prior-mode-grid logsumexp \
    --prior-alpha-grid "$alpha" \
    --prior-uniform-mix-grid "$mix" >/dev/null
}

run_evals() {
  local name=$1
  local dir="$OUT_ROOT/$name"
  for split in species43 species44; do
    local seed=${split#species}
    local cand="work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json"
    [[ -f "$dir/eval_${split}_raw/summary.json" ]] || run_eval_one "$dir/pseudo_species_adapted.pt" "$cand" "$dir/eval_${split}_raw" 0 0.5 0.95
    [[ -f "$dir/eval_${split}_b1/summary.json" ]] || run_eval_one "$dir/pseudo_species_adapted.pt" "$cand" "$dir/eval_${split}_b1" 1 0.5 0.95
    [[ -f "$dir/eval_${split}_b2/summary.json" ]] || run_eval_one "$dir/pseudo_species_adapted.pt" "$cand" "$dir/eval_${split}_b2" 2 0.25 0.95
  done
  for split in genus43 genus44; do
    local seed=${split#genus}
    local cand="work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json"
    [[ -f "$dir/eval_${split}_raw/summary.json" ]] || run_eval_one "$dir/pseudo_genus_adapted.pt" "$cand" "$dir/eval_${split}_raw" 0 0.5 0.95
    [[ -f "$dir/eval_${split}_b1/summary.json" ]] || run_eval_one "$dir/pseudo_genus_adapted.pt" "$cand" "$dir/eval_${split}_b1" 1 0.5 0.95
    [[ -f "$dir/eval_${split}_b2/summary.json" ]] || run_eval_one "$dir/pseudo_genus_adapted.pt" "$cand" "$dir/eval_${split}_b2" 2 0.25 0.95
  done
}

EP20_S="runs/global_visual_adapter_grid_20260630/taxon_id20_ep20/pseudo_species_adapted.pt"
EP20_G="runs/global_visual_adapter_grid_20260630/taxon_id20_ep20/pseudo_genus_adapted.pt"
EP20_U="runs/global_visual_adapter_grid_20260630/taxon_id20_ep20/test_unseen_adapted.pt"
EP30_S="runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_species_adapted.pt"
EP30_G="runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_genus_adapted.pt"
EP30_U="runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/test_unseen_adapted.pt"
ID30_S="runs/global_visual_adapter_followup_20260630/taxon_id30_ep30/pseudo_species_adapted.pt"
ID30_G="runs/global_visual_adapter_followup_20260630/taxon_id30_ep30/pseudo_genus_adapted.pt"
ID30_U="runs/global_visual_adapter_followup_20260630/taxon_id30_ep30/test_unseen_adapted.pt"

make_ensemble ep30_id30_w50 "0.5,0.5" "$EP30_S,$ID30_S" "$EP30_G,$ID30_G" "$EP30_U,$ID30_U"
make_ensemble ep30_id30_w73 "0.7,0.3" "$EP30_S,$ID30_S" "$EP30_G,$ID30_G" "$EP30_U,$ID30_U"
make_ensemble ep30_id30_w37 "0.3,0.7" "$EP30_S,$ID30_S" "$EP30_G,$ID30_G" "$EP30_U,$ID30_U"
make_ensemble ep20_ep30_w37 "0.3,0.7" "$EP20_S,$EP30_S" "$EP20_G,$EP30_G" "$EP20_U,$EP30_U"
make_ensemble ep20_ep30_id30_w2525 "0.25,0.5,0.25" "$EP20_S,$EP30_S,$ID30_S" "$EP20_G,$EP30_G,$ID30_G" "$EP20_U,$EP30_U,$ID30_U"

for name in ep30_id30_w50 ep30_id30_w73 ep30_id30_w37 ep20_ep30_w37 ep20_ep30_id30_w2525; do
  run_evals "$name"
done

python3 - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("runs/global_visual_adapter_ensemble_20260630/*/eval_*/summary.json")):
    config = f.split("/")[-3]
    eval_name = os.path.basename(os.path.dirname(f)).replace("eval_", "")
    mode = "raw" if eval_name.endswith("_raw") else ("b1" if eval_name.endswith("_b1") else "b2")
    best = json.load(open(f, encoding="utf-8"))["best"]
    rows.append({
        "config": config,
        "eval": eval_name,
        "mode": mode,
        "top1": best["top1"],
        "base_top1": best["base_top1"],
        "gain": best["top1"] - best["base_top1"],
        "net": best["net"],
        "wins": best["wins"],
        "losses": best["losses"],
        "changed": best["changed"],
    })
with open("runs/global_visual_adapter_ensemble_20260630/aggregate.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = []
for config in sorted({r["config"] for r in rows}):
    for mode in ["raw", "b1", "b2"]:
        group = [r for r in rows if r["config"] == config and r["mode"] == mode]
        if group:
            summary.append({
                "config": config,
                "mode": mode,
                "avg_top1": sum(r["top1"] for r in group) / len(group),
                "min_gain": min(r["gain"] for r in group),
                "avg_net": sum(r["net"] for r in group) / len(group),
                "min_net": min(r["net"] for r in group),
                "wins": sum(r["wins"] for r in group),
                "losses": sum(r["losses"] for r in group),
                "changed": sum(r["changed"] for r in group),
            })
summary.sort(key=lambda r: (r["avg_top1"], r["min_gain"], r["avg_net"]), reverse=True)
with open("runs/global_visual_adapter_ensemble_20260630/aggregate_summary.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary[:20]:
    print(row)
PY
