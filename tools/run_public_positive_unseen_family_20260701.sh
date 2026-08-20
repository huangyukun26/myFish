#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

OUT_ROOT=${OUT_ROOT:-runs/public_positive_unseen_family_20260701}
mkdir -p "$OUT_ROOT"

IMG_SPECIES=runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt
IMG_GENUS=runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt
TAXON=work/clip_text_features/all_bioclip25_taxon.pt
FISH=work/clip_text_features/all_bioclip25_fish.pt
DESC=work/clip_text_features/all_bioclip25_desc_short.pt
VISUAL=work/clip_text_features/all_bioclip25_visual_traits.pt

run_one() {
  local name=$1
  local text=$2
  local extra=$3
  local weights=$4
  local norm=$5
  local img=$6
  local cand=$7
  local out=$8
  [[ -f "$out/summary.json" ]] && return 0
  local extra_args=()
  if [[ -n "$extra" ]]; then
    extra_args+=(--extra-text-features "$extra" --text-weights "$weights" --logit-normalization "$norm")
  fi
  python3 tools/transductive_active_sinkhorn.py \
    --image-features "$img" \
    --text-features "$text" \
    "${extra_args[@]}" \
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

configs=(
  "taxon|$TAXON|||none"
  "taxon95_fish05|$TAXON|$FISH|0.95,0.05|none"
  "taxon90_fish10|$TAXON|$FISH|0.90,0.10|none"
  "taxon95_desc05|$TAXON|$DESC|0.95,0.05|none"
  "taxon95_visual05|$TAXON|$VISUAL|0.95,0.05|none"
  "taxon90_fish05_desc05|$TAXON|$FISH,$DESC|0.90,0.05,0.05|none"
  "taxon90_fish10_z|$TAXON|$FISH|0.90,0.10|zscore"
)

for cfg in "${configs[@]}"; do
  IFS='|' read -r name text extra weights norm <<<"$cfg"
  for split in species43 species44; do
    seed=${split#species}
    cand="work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json"
    run_one "$name" "$text" "$extra" "$weights" "$norm" "$IMG_SPECIES" "$cand" "$OUT_ROOT/${name}_${split}"
  done
  for split in genus43 genus44; do
    seed=${split#genus}
    cand="work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json"
    run_one "$name" "$text" "$extra" "$weights" "$norm" "$IMG_GENUS" "$cand" "$OUT_ROOT/${name}_${split}"
  done
done

python3 - <<'PY'
import csv, glob, json, os
rows = []
for f in sorted(glob.glob("runs/public_positive_unseen_family_20260701/*/summary.json")):
    name = os.path.basename(os.path.dirname(f))
    config, split = name.rsplit("_", 1)
    # handle names ending in species43/genus43: split by suffix instead.
    for suffix in ["species43", "species44", "genus43", "genus44"]:
        if name.endswith("_" + suffix):
            config = name[:-(len(suffix)+1)]
            split = suffix
            break
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
with open("runs/public_positive_unseen_family_20260701/best_by_split.csv", "w", newline="", encoding="utf-8") as fp:
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
with open("runs/public_positive_unseen_family_20260701/aggregate_summary.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
PY
