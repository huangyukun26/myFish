#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/selective_hybrid_h8192_vs_active_20260701}
mkdir -p "$OUT_ROOT"

TAXON=work/clip_text_features/all_bioclip25_taxon.pt

run_topk() {
  local image_features=$1
  local candidates=$2
  local out_dir=$3
  local active_count=$4
  local active_mode=$5
  local union_topk=$6
  local tau=$7
  local blend=$8
  [[ -f "$out_dir/topk.csv" ]] && return 0
  python3 tools/export_transductive_topk.py \
    --image-features "$image_features" \
    --text-features "$TAXON" \
    --candidate-classes "$candidates" \
    --out-dir "$out_dir" \
    --score-batch-size 256 \
    --active-count "$active_count" \
    --active-mode "$active_mode" \
    --union-topk "$union_topk" \
    --tau "$tau" \
    --blend "$blend" \
    --prior-mode logsumexp \
    --prior-alpha 0.5 \
    --prior-uniform-mix 0.95 \
    --topk 20
}

for split in species43 species44 genus43 genus44; do
  if [[ "$split" == species* ]]; then
    seed=${split#species}
    base_img=runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt
    cand_img=runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_species_adapted.pt
    cand_json=work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json
  else
    seed=${split#genus}
    base_img=runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt
    cand_img=runs/global_visual_adapter_followup_20260630/taxon_id20_ep30/pseudo_genus_adapted.pt
    cand_json=work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json
  fi

  base_dir="$OUT_ROOT/base_h8192_${split}"
  cand_dir="$OUT_ROOT/cand_active9000_${split}"
  gate_dir="$OUT_ROOT/gate_${split}"

  run_topk "$base_img" "$cand_json" "$base_dir" 11598 max 0 0.02 5
  run_topk "$cand_img" "$cand_json" "$cand_dir" 9000 logsumexp 2 0.04 2

  python3 tools/selective_override_predictions.py \
    --base-csv "$base_dir/predictions.csv" \
    --candidate-csv "$cand_dir/predictions.csv" \
    --base-topk-csv "$base_dir/topk.csv" \
    --candidate-topk-csv "$cand_dir/topk.csv" \
    --label-csv "$base_dir/topk.csv" \
    --base-margin-max-grid "0.02,0.05,0.1,0.2,0.3,0.5,1,2,5,10" \
    --candidate-margin-min-grid "0,0.02,0.05,0.1,0.2,0.3,0.5,1,2,5" \
    --out-dir "$gate_dir"
done

python3 - <<'PY'
import csv, glob, json, os
rows = []
for path in sorted(glob.glob("runs/selective_hybrid_h8192_vs_active_20260701/gate_*/summary.json")):
    split = os.path.basename(os.path.dirname(path)).replace("gate_", "")
    best = json.load(open(path, encoding="utf-8"))["best"]
    rows.append({"split": split, **best})
with open("runs/selective_hybrid_h8192_vs_active_20260701/best_by_split.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "avg_new_top1": sum(r["new_top1"] for r in rows) / len(rows),
    "avg_base_top1": sum(r["base_top1"] for r in rows) / len(rows),
    "min_gain": min(r["new_top1"] - r["base_top1"] for r in rows),
    "avg_net_wins": sum(r["net_wins"] for r in rows) / len(rows),
    "min_net_wins": min(r["net_wins"] for r in rows),
    "wins": sum(r["wins"] for r in rows),
    "losses": sum(r["losses"] for r in rows),
    "changed": sum(r["changed"] for r in rows),
}
json.dump({"rows": rows, "summary": summary}, open("runs/selective_hybrid_h8192_vs_active_20260701/summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
