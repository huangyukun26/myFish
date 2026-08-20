# FishNet Challenge Workspace

This GitHub export contains the reusable source code, configuration, and a small set of current experiment notes. Competition data, model weights, predictions, submissions, logs, and generated results are intentionally kept out of the repository; see `.gitignore` for the full exclusion list.

This workspace contains a lightweight experiment scaffold for the CV4Ecology Fish Species Recognition Challenge.

## Current Status

- Metadata is available under `dataset/`.
- `dataset/images.zip` is complete and verified.
- Images have been extracted to `dataset/images/`.
- A smoke subset has been extracted to `work/smoke_subset/` from complete ZIP entries already present in the partial archive.
- The first smoke run completed under `runs/smoke_20260618_115806/`.
- GPU smoke training works in the `sam_wvi` conda environment: `torch 2.5.1+cu121`, `device=cuda`.
- First supervised baseline completed under `runs/supervised_resnet18_20260618_153208/`.
- First baseline submission is available at `runs/submission_baseline/submission.zip`.
- Current best seen model is ResNet50 224 under `runs/supervised_resnet50_20260618_172225/`.
- Submitted BioCLIP2.5 candidate is `runs/submission_bioclip25_20260619/submission.zip` with reported public score `overall=0.40`, `seen=0.65`, `unseen=0.08`.
- Latest experimental candidate is `runs/submission_bioclip25_fish_taxon_20260619/submission.zip`, which keeps the same seen predictions and replaces unseen with a BioCLIP2.5 fish+taxon prompt ensemble.
- Latest rerank candidate is `runs/submission_bioclip25_taxon09_desc_sentence_rerank_20260623/submission.zip`, which keeps the same seen predictions and uses BioCLIP2.5 fish/taxon plus top20 `desc_sentence` reranking for unseen.
- Latest diagnostic report is `docs/RUN_REPORT_20260623_SEEN_DIST_AND_SCORE_ACCOUNTING.md`; no new submission was generated from that run because seen needed a clean training rerun and unseen weak rerank did not pass the multi-seed gate.
- Current seen-upgraded candidate is `runs/submission_seen_resnet50_4ep_hflip_tw065_keep_unseen_20260624/submission.zip`, which replaces seen predictions with ResNet50 4-epoch + BioCLIP2.5 hflip fusion and keeps the latest unseen predictions.

## Useful Commands

```powershell
python tools\check_env.py
python tools\inspect_dataset.py --dataset-root dataset --json-out work\dataset_summary.json
python tools\verify_images_zip.py --dataset-root dataset --json-out work\images_zip_verify.json
python tools\prepare_full_manifests.py --dataset-root dataset --output work\full_manifests --overwrite
python tools\extract_smoke_subset.py --dataset-root dataset --output work\smoke_subset --overwrite
python tools\smoke_train.py --data-root work\smoke_subset --run-root runs --epochs 3
python tools\monitor_gpu.py --out runs\gpu_monitor.csv --interval-sec 10
```

For GPU runs on this machine, prefer:

```powershell
conda run -n sam_wvi python tools\check_env.py
conda run -n sam_wvi python tools\smoke_train.py --data-root work\smoke_subset --run-root runs --epochs 5 --image-size 160
```

Read [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md) before starting long experiments.

Latest run summaries:

- [docs/RUN_REPORT_20260619_BIOCLIP25.md](docs/RUN_REPORT_20260619_BIOCLIP25.md)
- [docs/RUN_REPORT_20260619_PSEUDO_UNSEEN_TOPK.md](docs/RUN_REPORT_20260619_PSEUDO_UNSEEN_TOPK.md)
- [docs/RUN_REPORT_20260623_LARGE_CANDIDATE_RERANK.md](docs/RUN_REPORT_20260623_LARGE_CANDIDATE_RERANK.md)
- [docs/RUN_REPORT_20260623_SEEN_DIST_AND_SCORE_ACCOUNTING.md](docs/RUN_REPORT_20260623_SEEN_DIST_AND_SCORE_ACCOUNTING.md)
- [docs/RUN_NOTE_20260624_SEEN_CANDIDATE_AND_DESCRIPTION.md](docs/RUN_NOTE_20260624_SEEN_CANDIDATE_AND_DESCRIPTION.md)
