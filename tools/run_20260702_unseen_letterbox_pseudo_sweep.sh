#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs/letterbox_bioclip25_20260702}"
MODEL="${MODEL:-local-dir:work/hf_models/bioclip-2.5-vith14}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-512}"

mkdir -p "${RUN_ROOT}"

cache_letterbox() {
  local split="$1"
  local manifest
  local out
  local shard_dir

  if [[ "${split}" == "species" ]]; then
    manifest="work/pseudo_unseen/species_1000_seed42/manifest.csv"
    out="${RUN_ROOT}/pseudo_species_letterbox.pt"
    shard_dir="${RUN_ROOT}/pseudo_species_letterbox_shards"
  elif [[ "${split}" == "genus" ]]; then
    manifest="work/pseudo_unseen/genus_1000_seed42/manifest.csv"
    out="${RUN_ROOT}/pseudo_genus_letterbox.pt"
    shard_dir="${RUN_ROOT}/pseudo_genus_letterbox_shards"
  else
    echo "Unknown split: ${split}" >&2
    exit 2
  fi

  if [[ -f "${out}" ]]; then
    echo "Using existing ${out}"
    return
  fi

  python tools/cache_clip_image_features_sharded.py \
    --manifest "${manifest}" \
    --image-root dataset/images \
    --out "${out}" \
    --shard-dir "${shard_dir}" \
    --model "${MODEL}" \
    --pretrained none \
    --device cuda \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --tta-crops none \
    --preprocess-mode letterbox \
    --clip-precision fp16 \
    --shard-size 500 \
    --resume
}

average_hflip_letterbox() {
  local split="$1"
  local hflip
  local letterbox
  local out

  if [[ "${split}" == "species" ]]; then
    hflip="runs/bioclip25_image_cache_3090/pseudo_species1000_hflip.pt"
    letterbox="${RUN_ROOT}/pseudo_species_letterbox.pt"
    out="${RUN_ROOT}/pseudo_species_hflip_letterbox_avg.pt"
  elif [[ "${split}" == "genus" ]]; then
    hflip="runs/bioclip25_image_cache_3090/pseudo_genus1000_hflip.pt"
    letterbox="${RUN_ROOT}/pseudo_genus_letterbox.pt"
    out="${RUN_ROOT}/pseudo_genus_hflip_letterbox_avg.pt"
  else
    echo "Unknown split: ${split}" >&2
    exit 2
  fi

  if [[ -f "${out}" ]]; then
    echo "Using existing ${out}"
    return
  fi

  python tools/average_feature_caches.py \
    --inputs "${hflip},${letterbox}" \
    --out "${out}"
}

run_sinkhorn() {
  local split="$1"
  local seed="$2"
  local feature_name="$3"
  local text_name="$4"
  local feature_path
  local candidate_classes
  local out_dir
  shift 4

  if [[ "${split}" == "species" ]]; then
    candidate_classes="work/pseudo_unseen/species_1000_seed42/candidate_classes_11598_seed${seed}.json"
  elif [[ "${split}" == "genus" ]]; then
    candidate_classes="work/pseudo_unseen/genus_1000_seed42/candidate_classes_11598_seed${seed}.json"
  else
    echo "Unknown split: ${split}" >&2
    exit 2
  fi

  case "${feature_name}" in
    letterbox)
      feature_path="${RUN_ROOT}/pseudo_${split}_letterbox.pt"
      ;;
    hflip_letterbox_avg)
      feature_path="${RUN_ROOT}/pseudo_${split}_hflip_letterbox_avg.pt"
      ;;
    *)
      echo "Unknown feature_name: ${feature_name}" >&2
      exit 2
      ;;
  esac

  out_dir="${RUN_ROOT}/${split}${seed}_${feature_name}_${text_name}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "Using existing ${out_dir}/summary.json"
    return
  fi

  python tools/transductive_active_sinkhorn.py \
    --image-features "${feature_path}" \
    --text-features work/clip_text_features/all_bioclip25_taxon.pt \
    --candidate-classes "${candidate_classes}" \
    --out-dir "${out_dir}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --apply-active-count 11598 \
    --apply-active-mode max \
    --apply-union-topk 0 \
    --apply-tau 0.02 \
    --apply-blend 5.0 \
    --apply-prior-mode logsumexp \
    --apply-prior-alpha 0.5 \
    --apply-prior-uniform-mix 0.95 \
    "$@"
}

for split in species genus; do
  cache_letterbox "${split}"
  average_hflip_letterbox "${split}"
done

for split in species genus; do
  for seed in 43 44; do
    run_sinkhorn "${split}" "${seed}" letterbox taxon
    run_sinkhorn "${split}" "${seed}" letterbox taxon95_fish05_z \
      --extra-text-features work/clip_text_features/all_bioclip25_fish.pt \
      --text-weights 0.95,0.05 \
      --logit-normalization zscore
    run_sinkhorn "${split}" "${seed}" letterbox taxon90_fish10_z \
      --extra-text-features work/clip_text_features/all_bioclip25_fish.pt \
      --text-weights 0.90,0.10 \
      --logit-normalization zscore
    run_sinkhorn "${split}" "${seed}" hflip_letterbox_avg taxon
    run_sinkhorn "${split}" "${seed}" hflip_letterbox_avg taxon95_fish05_z \
      --extra-text-features work/clip_text_features/all_bioclip25_fish.pt \
      --text-weights 0.95,0.05 \
      --logit-normalization zscore
    run_sinkhorn "${split}" "${seed}" hflip_letterbox_avg taxon90_fish10_z \
      --extra-text-features work/clip_text_features/all_bioclip25_fish.pt \
      --text-weights 0.90,0.10 \
      --logit-normalization zscore
  done
done

python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("runs/letterbox_bioclip25_20260702")
rows = []
for summary_path in sorted(root.glob("*_*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    best = data.get("best", {})
    name = summary_path.parent.name
    parts = name.split("_", 2)
    split_seed = parts[0]
    rows.append({
        "name": name,
        "top1": best.get("top1", 0.0),
        "base_top1": best.get("base_top1", 0.0),
        "changed": best.get("changed", 0),
        "wins": best.get("wins", 0),
        "losses": best.get("losses", 0),
        "net": best.get("net", 0),
        "active_actual": best.get("active_actual", 0),
        "tau": best.get("tau", ""),
        "blend": best.get("blend", ""),
        "prior_mode": best.get("prior_mode", ""),
        "prior_alpha": best.get("prior_alpha", ""),
        "prior_uniform_mix": best.get("prior_uniform_mix", ""),
    })

out = root / "pseudo_sweep_summary.csv"
with out.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["name"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out} rows={len(rows)}")
PY
