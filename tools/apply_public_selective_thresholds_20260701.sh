#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/fishnet}
cd "$ROOT"

BASE_SUB=runs/current_best_20260630_overall046956_unseen011067/submission.zip
BASE_CSV=runs/topk_public_20260701/h8192_taxon_tau002_b5/predictions.csv
CAND_CSV=runs/topk_public_20260701/adapter_id20ep30_active9000/predictions.csv
BASE_TOPK=runs/topk_public_20260701/h8192_taxon_tau002_b5/topk.csv
CAND_TOPK=runs/topk_public_20260701/adapter_id20ep30_active9000/topk.csv

run_one() {
  local name=$1
  local base_margin=$2
  local cand_margin=$3
  local unseen_dir="runs/public_unseen_selective_20260701/h8192_active_${name}"
  local sub_dir="runs/submission_20260701_bestseen_unseen_h8192_active_${name}"
  python3 tools/apply_selective_override_by_margin.py \
    --base-csv "$BASE_CSV" \
    --candidate-csv "$CAND_CSV" \
    --base-topk-csv "$BASE_TOPK" \
    --candidate-topk-csv "$CAND_TOPK" \
    --base-margin-max "$base_margin" \
    --candidate-margin-min "$cand_margin" \
    --out-dir "$unseen_dir"
  python3 tools/override_submission_with_csv.py \
    --base "$BASE_SUB" \
    --override-csv "$unseen_dir/predictions.csv" \
    --out-dir "$sub_dir"
}

run_one bm0p1_cm0p5 0.1 0.5
run_one bm0p5_cm1 0.5 1.0
run_one bm0p5_cm0p5 0.5 0.5
run_one bm0p2_cm0p3 0.2 0.3
run_one bm5_cm1 5.0 1.0
