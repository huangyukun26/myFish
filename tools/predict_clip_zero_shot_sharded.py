from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot
from fishnet.image_preprocess import apply_preprocess_mode

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_candidate_classes(path: Optional[Path], text_classes: List[str]) -> List[str]:
    if path is None:
        return text_classes
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def row_zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def parse_csv_paths(value: str) -> List[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def parse_weights(value: str, count: int) -> List[float]:
    if not value:
        return [1.0 / count for _ in range(count)]
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != count:
        raise ValueError(f"Expected {count} text weights, got {len(weights)}")
    total = sum(weights)
    if total == 0:
        raise ValueError("Text weights must not sum to zero")
    return [weight / total for weight in weights]


def make_pil_variants(image: Image.Image, mode: str) -> List[Image.Image]:
    if mode == "none":
        return [image.copy()]
    if mode == "hflip":
        return [image.copy(), ImageOps.mirror(image)]
    if mode in {"fivecrop", "tencrop"}:
        width, height = image.size
        crop_size = min(width, height)
        boxes = [
            (0, 0, crop_size, crop_size),
            (width - crop_size, 0, width, crop_size),
            (0, height - crop_size, crop_size, height),
            (width - crop_size, height - crop_size, width, height),
            ((width - crop_size) // 2, (height - crop_size) // 2, (width + crop_size) // 2, (height + crop_size) // 2),
        ]
        crops = [image.crop(box).copy() for box in boxes]
        if mode == "fivecrop":
            return crops
        return crops + [ImageOps.mirror(crop) for crop in crops]
    raise ValueError(f"Unknown tta_crops: {mode}")


class PredictionRowsDataset(Dataset):
    def __init__(self, rows: List[dict], image_root: Path, preprocess, tta_crops: str, preprocess_mode: str):
        self.rows = rows
        self.image_root = image_root
        self.preprocess = preprocess
        self.tta_crops = tta_crops
        self.preprocess_mode = preprocess_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        label = row.get("label", "")
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            variants = [apply_preprocess_mode(variant, self.preprocess_mode) for variant in variants]
            x = torch.stack([self.preprocess(variant) for variant in variants])
        return x, image_id, label


def shard_paths(shard_dir: Path, shard_index: int) -> tuple[Path, Path, Path]:
    stem = f"shard_{shard_index:05d}"
    return shard_dir / f"{stem}.csv", shard_dir / f"{stem}.topk.jsonl", shard_dir / f"{stem}.summary.json"


def valid_completed_shard(summary_path: Path, expected_rows: int, args_hash: dict) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("rows") == expected_rows and summary.get("args_hash") == args_hash


def write_prediction_shard(
    *,
    rows: List[dict],
    shard_index: int,
    shard_dir: Path,
    image_root: Path,
    preprocess,
    model,
    candidate_feature_sets: List[torch.Tensor],
    text_weights: List[float],
    rerank_feature_set: Optional[torch.Tensor],
    rerank_weight: float,
    score_normalization: str,
    candidates: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_crops: str,
    preprocess_mode: str,
    topk: int,
    amp: bool,
    args_hash: dict,
) -> dict:
    csv_path, topk_path, summary_path = shard_paths(shard_dir, shard_index)
    ds = PredictionRowsDataset(rows, image_root, preprocess, tta_crops, preprocess_mode)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    k = min(topk, len(candidates))
    shard_stats = {
        "shard_index": shard_index,
        "rows": 0,
        "labeled": 0,
        "rank_known": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "topk_correct": 0,
        "args_hash": args_hash,
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_fp, topk_path.open("w", encoding="utf-8") as topk_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=["image_id", "prediction", "score", "label", "true_rank"])
        writer.writeheader()
        with torch.inference_mode():
            for x, image_ids, labels in tqdm(loader, desc=f"shard_{shard_index:05d}", leave=False):
                current_batch = len(image_ids)
                crop_count = x.shape[1]
                x = x.to(device, non_blocking=True).flatten(0, 1)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    image_features = model.encode_image(x)
                image_features = normalize_features(image_features.float()).view(current_batch, crop_count, -1)
                image_features = normalize_features(image_features.mean(dim=1))
                logits = None
                for weight, candidate_features in zip(text_weights, candidate_feature_sets):
                    part = image_features @ candidate_features.T
                    if score_normalization == "zscore":
                        part = row_zscore(part)
                    logits = part.mul(weight) if logits is None else logits + part.mul(weight)
                if logits is None:
                    raise RuntimeError("No text feature sets were provided")
                top_scores, top_indices = logits.topk(k, dim=1)

                reranked_within_topk = rerank_feature_set is not None and rerank_weight != 0.0
                if reranked_within_topk:
                    rerank_logits = image_features @ rerank_feature_set.T
                    rerank_top_scores = torch.gather(rerank_logits, 1, top_indices)
                    rerank_top_scores = row_zscore(rerank_top_scores)
                    final_top_scores = top_scores + rerank_weight * rerank_top_scores
                    order = final_top_scores.argsort(dim=1, descending=True)
                    top_scores = torch.gather(final_top_scores, 1, order)
                    top_indices = torch.gather(top_indices, 1, order)

                top_scores_cpu = top_scores.cpu()
                top_indices_cpu = top_indices.cpu()
                logits_cpu = logits.cpu() if not reranked_within_topk else None

                for row_index, (image_id, label) in enumerate(zip(image_ids, labels)):
                    pred_idx = int(top_indices_cpu[row_index, 0].item())
                    pred = candidates[pred_idx]
                    score = float(top_scores_cpu[row_index, 0].item())
                    true_rank = ""
                    if label:
                        shard_stats["labeled"] += 1
                        true_idx = class_to_idx.get(label)
                        if true_idx is not None:
                            shard_stats["rank_known"] += 1
                            if reranked_within_topk:
                                matches = (top_indices_cpu[row_index] == true_idx).nonzero(as_tuple=False)
                                rank = int(matches[0, 0].item()) + 1 if matches.numel() else k + 1
                            else:
                                true_score = logits_cpu[row_index, true_idx]
                                rank = int((logits_cpu[row_index] > true_score).sum().item()) + 1
                            true_rank = rank
                            shard_stats["top1_correct"] += int(rank == 1)
                            shard_stats["top5_correct"] += int(rank <= 5)
                            shard_stats["topk_correct"] += int(rank <= k)
                    predictions = [candidates[int(idx)] for idx in top_indices_cpu[row_index].tolist()]
                    scores = [float(value) for value in top_scores_cpu[row_index].tolist()]
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "prediction": pred,
                            "score": score,
                            "label": label,
                            "true_rank": true_rank,
                        }
                    )
                    topk_fp.write(
                        json.dumps(
                            {
                                "image_id": image_id,
                                "label": label,
                                "predictions": predictions,
                                "scores": scores,
                                "true_rank": true_rank,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    shard_stats["rows"] += 1

    summary_path.write_text(json.dumps(shard_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return shard_stats


def merge_shards(shard_dir: Path, shard_count: int, out_csv: Path, out_topk_jsonl: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as out_fp:
        writer = csv.DictWriter(out_fp, fieldnames=["image_id", "prediction", "score", "label", "true_rank"])
        writer.writeheader()
        for shard_index in range(shard_count):
            csv_path, _topk_path, _summary_path = shard_paths(shard_dir, shard_index)
            with csv_path.open("r", encoding="utf-8", newline="") as shard_fp:
                for row in csv.DictReader(shard_fp):
                    writer.writerow(row)

    with out_topk_jsonl.open("w", encoding="utf-8") as out_fp:
        for shard_index in range(shard_count):
            _csv_path, topk_path, _summary_path = shard_paths(shard_dir, shard_index)
            with topk_path.open("r", encoding="utf-8") as shard_fp:
                for line in shard_fp:
                    out_fp.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", default="", help="Comma-separated extra text feature files to ensemble.")
    parser.add_argument("--text-weights", default="", help="Comma-separated weights for primary plus extra text feature files.")
    parser.add_argument("--rerank-text-features", type=Path, default=None, help="Optional text features used only to rerank the first-stage topK.")
    parser.add_argument("--rerank-weight", type=float, default=0.0, help="Weight added to topK z-scored rerank scores.")
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--preprocess-mode", choices=["model", "letterbox"], default="model")
    parser.add_argument("--clip-precision", default="fp32")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-on-cpu", action="store_true", help="Load checkpoint on CPU before moving model to CUDA.")
    parser.add_argument(
        "--load-safetensors-bytes",
        action="store_true",
        help="Opt-in workaround for Windows pagefile mmap failures when loading safetensors.",
    )
    parser.add_argument(
        "--preload-safetensors-state",
        action="store_true",
        help="Load local-dir safetensors state dict before model construction, then create model with load_weights=False.",
    )
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    import open_clip

    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError(f"No rows in {args.manifest}")

    text_feature_paths = [args.text_features] + parse_csv_paths(args.extra_text_features)
    text_weights = parse_weights(args.text_weights, len(text_feature_paths))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in text_feature_paths]
    text_classes = payloads[0]["classes"]
    candidates = load_candidate_classes(args.candidate_classes, text_classes)
    candidate_feature_sets = []
    for path, payload in zip(text_feature_paths, payloads):
        class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
        missing = [name for name in candidates if name not in class_to_idx]
        if missing:
            raise RuntimeError(f"{len(missing)} candidate classes missing from {path}; first={missing[:5]}")
        candidate_indices = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
        candidate_feature_sets.append(normalize_features(payload["features"].float()[candidate_indices]))
    rerank_feature_set = None
    if args.rerank_text_features is not None:
        rerank_payload = torch.load(args.rerank_text_features, map_location="cpu", weights_only=False)
        rerank_class_to_idx = {name: idx for idx, name in enumerate(rerank_payload["classes"])}
        missing = [name for name in candidates if name not in rerank_class_to_idx]
        if missing:
            raise RuntimeError(f"{len(missing)} candidate classes missing from {args.rerank_text_features}; first={missing[:5]}")
        candidate_indices = torch.tensor([rerank_class_to_idx[name] for name in candidates], dtype=torch.long)
        rerank_feature_set = normalize_features(rerank_payload["features"].float()[candidate_indices])

    model_name = args.model or payloads[0]["model"]
    pretrained = args.pretrained or payloads[0]["pretrained"]
    pretrained_for_open_clip = None if str(pretrained).lower() in {"none", "null", ""} else pretrained
    if args.load_safetensors_bytes:
        from fishnet.safetensors_compat import patch_open_clip_safetensors_load_file

        patch_open_clip_safetensors_load_file()
    preloaded_state = None
    if args.preload_safetensors_state:
        if not str(model_name).startswith("local-dir:"):
            raise ValueError("--preload-safetensors-state currently supports only local-dir: models")
        from fishnet.safetensors_compat import load_state_dict_without_mmap

        model_dir = Path(str(model_name).split("local-dir:", 1)[1])
        checkpoint_path = model_dir / "open_clip_model.safetensors"
        preloaded_state = load_state_dict_without_mmap(checkpoint_path)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_device = torch.device("cpu") if args.load_on_cpu else device
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_for_open_clip,
        precision=args.clip_precision,
        device=load_device,
        load_weights=not args.preload_safetensors_state,
    )
    if load_device != device:
        model = model.to(device)
    if preloaded_state is not None:
        model.load_state_dict(preloaded_state, strict=True)
        del preloaded_state
    model = model.eval()
    candidate_feature_sets = [features.to(device) for features in candidate_feature_sets]
    if rerank_feature_set is not None:
        rerank_feature_set = rerank_feature_set.to(device)
    amp = (not args.no_amp) and device.type == "cuda"

    args_hash = {
        "manifest": str(args.manifest),
        "text_feature_paths": [str(path) for path in text_feature_paths],
        "text_weights": text_weights,
        "rerank_text_features": str(args.rerank_text_features) if args.rerank_text_features else None,
        "rerank_weight": args.rerank_weight,
        "score_normalization": args.score_normalization,
        "candidate_classes": str(args.candidate_classes),
        "model": model_name,
        "pretrained": pretrained,
        "device": args.device,
        "clip_precision": args.clip_precision,
        "tta_crops": args.tta_crops,
        "preprocess_mode": args.preprocess_mode,
        "topk": args.topk,
        "load_on_cpu": args.load_on_cpu,
        "load_safetensors_bytes": args.load_safetensors_bytes,
        "preload_safetensors_state": args.preload_safetensors_state,
    }
    shard_dir = args.out_dir / "shards"
    shard_count = math.ceil(len(rows) / args.shard_size)
    shard_summaries = []
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        _csv_path, _topk_path, summary_path = shard_paths(shard_dir, shard_index)
        if args.resume and valid_completed_shard(summary_path, len(shard_rows), args_hash):
            shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(json.dumps({"skip_shard": shard_index, "rows": len(shard_rows)}, ensure_ascii=False), flush=True)
            continue
        shard_summaries.append(
            write_prediction_shard(
                rows=shard_rows,
                shard_index=shard_index,
                shard_dir=shard_dir,
                image_root=args.image_root,
                preprocess=preprocess_val,
                model=model,
                candidate_feature_sets=candidate_feature_sets,
                text_weights=text_weights,
                rerank_feature_set=rerank_feature_set,
                rerank_weight=args.rerank_weight,
                score_normalization=args.score_normalization,
                candidates=candidates,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tta_crops=args.tta_crops,
                preprocess_mode=args.preprocess_mode,
                topk=args.topk,
                amp=amp,
                args_hash=args_hash,
            )
        )

    out_csv = args.out_dir / "predictions.csv"
    out_topk_jsonl = args.out_dir / "topk.jsonl"
    merge_shards(shard_dir, shard_count, out_csv, out_topk_jsonl)

    labeled = sum(item["labeled"] for item in shard_summaries)
    rank_known = sum(item["rank_known"] for item in shard_summaries)
    top1_correct = sum(item["top1_correct"] for item in shard_summaries)
    top5_correct = sum(item["top5_correct"] for item in shard_summaries)
    topk_correct = sum(item["topk_correct"] for item in shard_summaries)
    summary = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "candidate_classes": len(candidates),
        "text_feature_paths": [str(path) for path in text_feature_paths],
        "text_weights": text_weights,
        "rerank_text_features": str(args.rerank_text_features) if args.rerank_text_features else None,
        "rerank_weight": args.rerank_weight,
        "score_normalization": args.score_normalization,
        "model": model_name,
        "pretrained": pretrained,
        "clip_precision": args.clip_precision,
        "amp": amp,
        "tta_crops": args.tta_crops,
        "preprocess_mode": args.preprocess_mode,
        "topk": args.topk,
        "shard_size": args.shard_size,
        "shards": shard_count,
        "labeled": labeled,
        "rank_known": rank_known,
        "top1": (top1_correct / rank_known) if rank_known else None,
        "top5": (top5_correct / rank_known) if rank_known else None,
        "topk_accuracy": (topk_correct / rank_known) if rank_known else None,
        "out_csv": str(out_csv),
        "out_topk_jsonl": str(out_topk_jsonl),
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
