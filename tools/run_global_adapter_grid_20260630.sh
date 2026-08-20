#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/global_visual_adapter_grid_20260630}
mkdir -p "$OUT_ROOT"

TRAIN_FEATURES=${TRAIN_FEATURES:-runs/bioclip25_image_cache_3090/trainval_hflip.pt}
PSEUDO_SPECIES=${PSEUDO_SPECIES:-runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt}
PSEUDO_GENUS=${PSEUDO_GENUS:-runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt}
SEEN_CLASSES=${SEEN_CLASSES:-work/full_manifests/seen_class_to_idx.json}

run_eval_one() {
  local feature_path=$1
  local text_path=$2
  local cand_path=$3
  local out_dir=$4
  local blend=$5

  python3 tools/transductive_active_sinkhorn.py \
    --image-features "$feature_path" \
    --text-features "$text_path" \
    --candidate-classes "$cand_path" \
    --out-dir "$out_dir" \
    --score-batch-size 256 \
    --active-count-grid 11598 \
    --active-mode-grid max \
    --union-topk-grid 0 \
    --tau-grid 0.04 \
    --blend-grid "$blend" \
    --prior-mode-grid logsumexp \
    --prior-alpha-grid 0.5 \
    --prior-uniform-mix-grid 0.95
}

run_adapter() {
  local name=$1
  local text_path=$2
  local epochs=$3
  local identity_weight=$4
  local lr=${5:-1e-4}

  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  if [[ ! -f "$dir/summary.json" ]]; then
    python3 tools/train_global_visual_adapter.py \
      --train-features "$TRAIN_FEATURES" \
      --text-features "$text_path" \
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

  for mode in raw b1; do
    local blend=0
    if [[ "$mode" == "b1" ]]; then
      blend=1
    fi
    for split in species43 species44; do
      local seed=${split#species}
      local cand="work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json"
      local eval_dir="$dir/eval_${split}_${mode}"
      if [[ ! -f "$eval_dir/summary.json" ]]; then
        run_eval_one "$dir/pseudo_species_adapted.pt" "$text_path" "$cand" "$eval_dir" "$blend"
      fi
    done
    for split in genus43 genus44; do
      local seed=${split#genus}
      local cand="work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json"
      local eval_dir="$dir/eval_${split}_${mode}"
      if [[ ! -f "$eval_dir/summary.json" ]]; then
        run_eval_one "$dir/pseudo_genus_adapted.pt" "$text_path" "$cand" "$eval_dir" "$blend"
      fi
    done
  done
}

run_adapter taxon_id20_ep15 work/clip_text_features/all_bioclip25_taxon.pt 15 20
run_adapter taxon_id10_ep15 work/clip_text_features/all_bioclip25_taxon.pt 15 10
run_adapter taxon_id5_ep12 work/clip_text_features/all_bioclip25_taxon.pt 12 5
run_adapter taxon_id20_ep20 work/clip_text_features/all_bioclip25_taxon.pt 20 20
run_adapter fish_id20_ep10 work/clip_text_features/all_bioclip25_fish.pt 10 20
run_adapter fishtaxon_id20_ep10 work/clip_text_features/all_bioclip25_fish_taxon_avg.pt 10 20
run_adapter fish01taxon09_id20_ep10 work/clip_text_features/all_bioclip25_fish01_taxon09_avg.pt 10 20

python3 - <<'PY'
import csv, glob, json, os

rows = []
for summary_path in sorted(glob.glob("runs/global_visual_adapter_grid_20260630/*/eval_*/summary.json")):
    parts = summary_path.split("/")
    config = parts[-3]
    eval_name = parts[-2].replace("eval_", "")
    data = json.load(open(summary_path, encoding="utf-8"))
    best = data.get("best", {})
    rows.append({
        "config": config,
        "eval": eval_name,
        "top1": best.get("top1", 0.0),
        "base_top1": best.get("base_top1", 0.0),
        "changed": best.get("changed", 0),
        "wins": best.get("wins", 0),
        "losses": best.get("losses", 0),
        "net": best.get("net", 0),
    })

out = "runs/global_visual_adapter_grid_20260630/aggregate.csv"
with open(out, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=["config", "eval", "top1", "base_top1", "changed", "wins", "losses", "net"])
    writer.writeheader()
    writer.writerows(rows)

agg = {}
for row in rows:
    config, mode = row["config"], row["eval"].rsplit("_", 1)[-1]
    key = (config, mode)
    agg.setdefault(key, []).append(row)
summary_rows = []
for (config, mode), group in agg.items():
    summary_rows.append({
        "config": config,
        "mode": mode,
        "avg_top1": sum(float(r["top1"]) for r in group) / len(group),
        "min_gain": min(float(r["top1"]) - float(r["base_top1"]) for r in group),
        "avg_net": sum(float(r["net"]) for r in group) / len(group),
        "min_net": min(float(r["net"]) for r in group),
        "total_wins": sum(int(r["wins"]) for r in group),
        "total_losses": sum(int(r["losses"]) for r in group),
    })
summary_rows.sort(key=lambda r: (r["avg_top1"], r["min_gain"], r["avg_net"]), reverse=True)
out2 = "runs/global_visual_adapter_grid_20260630/aggregate_summary.csv"
with open(out2, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=["config", "mode", "avg_top1", "min_gain", "avg_net", "min_net", "total_wins", "total_losses"])
    writer.writeheader()
    writer.writerows(summary_rows)
for row in summary_rows[:20]:
    print(row)
PY
