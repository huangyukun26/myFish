#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/selective_hybrid_h8192_vs_orig90_20260701}
mkdir -p "$OUT_ROOT"

TAXON=work/clip_text_features/all_bioclip25_taxon.pt

run_topk() {
  local img=$1
  local cand=$2
  local out=$3
  [[ -f "$out/topk.csv" ]] && return 0
  python3 tools/export_transductive_topk.py \
    --image-features "$img" \
    --text-features "$TAXON" \
    --candidate-classes "$cand" \
    --out-dir "$out" \
    --score-batch-size 256 \
    --active-count 11598 \
    --active-mode max \
    --union-topk 0 \
    --tau 0.03 \
    --blend 5 \
    --prior-mode logsumexp \
    --prior-alpha 0.75 \
    --prior-uniform-mix 0.93 \
    --topk 20
}

for split in species43 species44 genus43 genus44; do
  if [[ "$split" == species* ]]; then
    seed=${split#species}
    cand_json=work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json
    cand_img=runs/unseen_feature_interp_public_positive_20260701/features/orig90_adapt10_pseudo_species.pt
  else
    seed=${split#genus}
    cand_json=work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json
    cand_img=runs/unseen_feature_interp_public_positive_20260701/features/orig90_adapt10_pseudo_genus.pt
  fi

  base_dir=runs/selective_hybrid_h8192_vs_active_20260701/base_h8192_${split}
  cand_dir="$OUT_ROOT/cand_orig90_${split}"
  gate_dir="$OUT_ROOT/gate_${split}"
  run_topk "$cand_img" "$cand_json" "$cand_dir"

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
import csv, glob, os
from collections import defaultdict
agg = defaultdict(list)
for f in glob.glob("runs/selective_hybrid_h8192_vs_orig90_20260701/gate_*/sweep.csv"):
    split = os.path.basename(os.path.dirname(f)).replace("gate_", "")
    for row in csv.DictReader(open(f, encoding="utf-8")):
        key = (float(row["base_margin_max"]), float(row["candidate_margin_min"]))
        item = {k: float(v) for k, v in row.items()}
        item["split"] = split
        agg[key].append(item)
rows = []
for (bm, cm), items in agg.items():
    if len(items) != 4:
        continue
    wins = sum(int(x["wins"]) for x in items)
    losses = sum(int(x["losses"]) for x in items)
    rows.append({
        "base_margin_max": bm,
        "candidate_margin_min": cm,
        "avg_base_top1": sum(x["base_top1"] for x in items) / 4,
        "avg_new_top1": sum(x["new_top1"] for x in items) / 4,
        "min_gain": min(x["new_top1"] - x["base_top1"] for x in items),
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "min_net": min(int(x["net_wins"]) for x in items),
        "changed": sum(int(x["changed"]) for x in items),
        "triggered": sum(int(x["triggered"]) for x in items),
        "win_loss_ratio": wins / max(1, losses),
    })
rows.sort(key=lambda x: (x["min_net"], x["net"], x["win_loss_ratio"], -x["changed"]), reverse=True)
with open("runs/selective_hybrid_h8192_vs_orig90_20260701/threshold_aggregate.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
for row in rows[:20]:
    print(row)
PY
