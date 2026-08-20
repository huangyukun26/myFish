#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs/letterbox_bioclip25_20260701}"
MODEL="${MODEL:-local-dir:work/hf_models/bioclip-2.5-vith14}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLIP_PRECISION="${CLIP_PRECISION:-fp16}"
BASE_SUBMISSION="${BASE_SUBMISSION:-runs/current_best_20260630_overall046956_unseen011067/submission.zip}"

PUBLIC_FEATURES="${RUN_ROOT}/public_unseen_letterbox.pt"
PUBLIC_SHARDS="${RUN_ROOT}/public_unseen_letterbox_shards"
PUBLIC_PRED_DIR="${RUN_ROOT}/public_unseen_letterbox_h8192_config_taxon"
SUBMISSION_DIR="runs/submission_20260701_bestseen_unseen_letterbox_h8192config_taxon"

python tools/cache_clip_image_features_sharded.py \
  --manifest work/full_manifests/test_unseen.csv \
  --image-root dataset/images \
  --out "${PUBLIC_FEATURES}" \
  --shard-dir "${PUBLIC_SHARDS}" \
  --model "${MODEL}" \
  --pretrained none \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --tta-crops none \
  --preprocess-mode letterbox \
  --clip-precision "${CLIP_PRECISION}" \
  --shard-size 1000 \
  --resume

python tools/transductive_active_sinkhorn.py \
  --image-features "${PUBLIC_FEATURES}" \
  --text-features work/clip_text_features/all_bioclip25_taxon.pt \
  --candidate-classes work/full_manifests/unseen_candidate_classes.json \
  --out-dir "${PUBLIC_PRED_DIR}" \
  --score-batch-size 512 \
  --apply-active-count 11598 \
  --apply-active-mode max \
  --apply-union-topk 0 \
  --apply-tau 0.02 \
  --apply-blend 5.0 \
  --apply-prior-mode logsumexp \
  --apply-prior-alpha 0.5 \
  --apply-prior-uniform-mix 0.95

python tools/override_submission_with_csv.py \
  --base "${BASE_SUBMISSION}" \
  --override-csv "${PUBLIC_PRED_DIR}/predictions.csv" \
  --submission-keys work/full_manifests/submission_keys.csv \
  --out-dir "${SUBMISSION_DIR}"

echo "Wrote ${SUBMISSION_DIR}/submission.zip"
