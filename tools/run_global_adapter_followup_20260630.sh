#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/global_visual_adapter_followup_20260630}
mkdir -p "$OUT_ROOT"

TRAIN_FEATURES=${TRAIN_FEATURES:-runs/bioclip25_image_cache_3090/trainval_hflip.pt}
PSEUDO_SPECIES=${PSEUDO_SPECIES:-runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt}
PSEUDO_GENUS=${PSEUDO_GENUS:-runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt}
TEXT=${TEXT:-work/clip_text_features/all_bioclip25_taxon.pt}
SEEN_CLASSES=${SEEN_CLASSES:-work/full_manifests/seen_class_to_idx.json}

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

run_adapter() {
  local name=$1
  local epochs=$2
  local identity_weight=$3
  local lr=$4
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  if [[ ! -f "$dir/summary.json" ]]; then
    python3 tools/train_global_visual_adapter.py \
      --train-features "$TRAIN_FEATURES" \
      --text-features "$TEXT" \
      --seen-classes "$SEEN_CLASSES" \
      --eval-features "$PSEUDO_SPECIES,$PSEUDO_GENUS" \
      --eval-outs "$dir/pseudo_species_adapted.pt,$dir/pseudo_genus_adapted.pt" \
      --out-dir "$dir" \
      --epochs "$epochs" \
      --batch-size 1024 \
      --lr "$lr" \
      --weight-decay 1e-4 \
      --identity-weight "$identity_weight"
  fi
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

run_adapter taxon_id20_ep25 25 20 1e-4
run_adapter taxon_id20_ep30 30 20 1e-4
run_adapter taxon_id30_ep30 30 30 1e-4
run_adapter taxon_id20_ep30_lr5e5 30 20 5e-5

python3 - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("runs/global_visual_adapter_followup_20260630/*/eval_*/summary.json")):
    config = f.split("/")[-3]
    eval_name = os.path.basename(os.path.dirname(f)).replace("eval_", "")
    if eval_name.endswith("_raw"):
        mode = "raw"
    elif eval_name.endswith("_b1"):
        mode = "b1"
    elif eval_name.endswith("_b2"):
        mode = "b2"
    else:
        mode = eval_name.rsplit("_", 1)[-1]
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
out = "runs/global_visual_adapter_followup_20260630/aggregate.csv"
with open(out, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = []
for config in sorted({r["config"] for r in rows}):
    for mode in ["raw", "b1", "b2"]:
        group = [r for r in rows if r["config"] == config and r["mode"] == mode]
        if not group:
            continue
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
out = "runs/global_visual_adapter_followup_20260630/aggregate_summary.csv"
with open(out, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary[:20]:
    print(row)
PY
