#!/usr/bin/env bash
set -euo pipefail

cd "${FISHNET_ROOT:-/root/fishnet}"

MODEL="${BIOCLIP25_MODEL:-local-dir:work/hf_models/bioclip-2.5-vith14}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/images}"
OUT_DIR="${OUT_DIR:-runs/bioclip25_image_cache_3090}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
TTA_CROPS="${TTA_CROPS:-hflip}"

mkdir -p "$OUT_DIR"

run_cache() {
  local split="$1"
  local manifest="$2"
  echo "CACHE ${split} manifest=${manifest}"
  python tools/cache_clip_image_features_sharded.py \
    --manifest "$manifest" \
    --image-root "$IMAGE_ROOT" \
    --out "${OUT_DIR}/${split}_${TTA_CROPS}.pt" \
    --shard-dir "${OUT_DIR}/${split}_${TTA_CROPS}_shards" \
    --model "$MODEL" \
    --pretrained none \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --tta-crops "$TTA_CROPS" \
    --clip-precision fp16 \
    --shard-size "$SHARD_SIZE" \
    --resume
}

run_cache train work/seen_image_distribution_split_seed2027_frac20/train.csv
run_cache val work/seen_image_distribution_split_seed2027_frac20/val.csv
run_cache test_seen work/full_manifests/test_seen.csv
run_cache test_unseen work/full_manifests/test_unseen.csv

