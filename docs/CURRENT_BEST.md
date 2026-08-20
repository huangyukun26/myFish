# Current Best

Updated: 2026-08-08

## Public score

Current online-best ZIP supplied by the user:

- overall: about `0.51`
- exact seen / unseen split scores: not recorded in this file yet

## Recovery archive

- archive: `runs/current_best_online_20260808_overall051/`
- ready-to-submit ZIP: `runs/current_best_online_20260808_overall051/submission/submission.zip`
- prediction: `runs/current_best_online_20260808_overall051/submission/prediction.json`
- checksums and lineage: `runs/current_best_online_20260808_overall051/ARCHIVE_MANIFEST.json`

The ZIP contains exactly one root `prediction.json`, has `35665` rows, matches
the submission-key order, and uses only known labels.

## Seen method

The public-positive branch is:

`joint_ens_2027_2028_2029 / joint_margin_0p281_or_dino_agree`

The local prediction bytes match:

`runs/cloud_20260716_seen_dinov3l_return/submissions_agreement/joint_ens_2027_2028_2029/joint_margin_0p281_or_dino_agree/prediction.json`

Relative to the superseded 2026-07-30 archive, it changes:

- seen rows: `701`
- unseen rows: `0`

## Experiment policy after this checkpoint

- All future submissions must be generated as overlays on
  `runs/current_best_online_20260808_overall051/submission/prediction.json`.
- Before public submission, run `tools/guard_submission_against_online_best.py`.
- Do not submit candidates based on
  `runs/current_best_archive_20260730_seen078046/`; that archive is superseded.
- Do not reuse the 2026-08-08 external-data safe / balanced / push /
  aggressive packages.
