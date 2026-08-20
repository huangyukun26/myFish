from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import ImageOps
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot
from train_supervised import build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


class FusionPredictionDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        supervised_transform,
        clip_transform,
        max_samples: int = 0,
        tta_crops: str = "none",
        start_index: int = 0,
        end_index: int | None = None,
    ):
        self.image_root = image_root
        self.supervised_transform = supervised_transform
        self.clip_transform = clip_transform
        self.tta_crops = tta_crops
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.rows = self.rows[start_index:end_index]
        if not self.rows:
            raise RuntimeError(f"No rows in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            supervised_x = torch.stack([self.supervised_transform(variant) for variant in variants])
            clip_x = torch.stack([self.clip_transform(variant) for variant in variants])
        return supervised_x, clip_x, image_id, row.get("label", "")


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


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def row_zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def write_prediction_outputs(
    rows: list[dict],
    topk_rows: list[dict],
    out_csv: Path,
    out_json: Path | None = None,
    topk_csv: Path | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "score"])
        writer.writeheader()
        writer.writerows(rows)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps({row["image_id"]: row["prediction"] for row in rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if topk_csv:
        topk_csv.parent.mkdir(parents=True, exist_ok=True)
        with topk_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["image_id", "true_label", "prediction", "margin_top1_top2", "top_classes", "top_scores"],
            )
            writer.writeheader()
            writer.writerows(topk_rows)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--supervised-checkpoint", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--supervised-image-size", type=int, default=224)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    parser.add_argument("--text-weight", type=float, default=0.38)
    parser.add_argument("--supervised-temp", type=float, default=1.0)
    parser.add_argument("--clip-temp", type=float, default=1.0)
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--clip-precision", default="fp32")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--topk-csv", type=Path, default=None)
    parser.add_argument("--topk-k", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from torchvision import transforms
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_classes(args.class_map)

    checkpoint = torch.load(args.supervised_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    supervised_model_name = ckpt_args.get("model", "resnet18")
    num_classes = int(ckpt_args.get("num_classes", len(class_names)))
    supervised_model = build_model(supervised_model_name, num_classes, pretrained=False)
    supervised_model.load_state_dict(checkpoint["model"])
    supervised_model = supervised_model.to(device).eval()

    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    text_classes = text_payload["classes"]
    text_class_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    missing = [name for name in class_names if name not in text_class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} class names missing from text features; first={missing[:5]}")
    text_indices = torch.tensor([text_class_to_idx[name] for name in class_names], dtype=torch.long)
    text_features = text_payload["features"].float()[text_indices]
    text_features = normalize_features(text_features).to(device)

    clip_model_name = args.clip_model or text_payload["model"]
    clip_pretrained = args.clip_pretrained or text_payload["pretrained"]
    clip_pretrained_arg = None if str(clip_pretrained).lower() in {"none", "null", ""} else clip_pretrained
    clip_model, _clip_preprocess_train, clip_preprocess_val = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained_arg,
        precision=args.clip_precision,
        device=device,
    )
    clip_model = clip_model.eval()

    supervised_transform = transforms.Compose(
        [
            transforms.Resize(int(args.supervised_image_size * 1.15)),
            transforms.CenterCrop(args.supervised_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    amp = (not args.no_amp) and device.type == "cuda"
    supervised_weight = 1.0 - args.text_weight

    def predict_slice(start_index: int, end_index: int, desc: str) -> tuple[list[dict], list[dict]]:
        ds = FusionPredictionDataset(
            args.manifest,
            args.image_root,
            supervised_transform,
            clip_preprocess_val,
            max_samples=args.max_samples,
            tta_crops=args.tta_crops,
            start_index=start_index,
            end_index=end_index,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        rows: list[dict] = []
        topk_rows: list[dict] = []
        with torch.inference_mode():
            iterator = tqdm(loader, desc=desc)
            for supervised_x, clip_x, image_ids, labels in iterator:
                batch_size, crop_count = supervised_x.shape[:2]
                supervised_x = supervised_x.to(device, non_blocking=True)
                clip_x = clip_x.to(device, non_blocking=True)
                supervised_x = supervised_x.flatten(0, 1)
                clip_x = clip_x.flatten(0, 1)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    supervised_scores = supervised_model(supervised_x).float().view(batch_size, crop_count, -1).mean(dim=1)
                    clip_image_features = clip_model.encode_image(clip_x)
                clip_image_features = normalize_features(clip_image_features.float()).view(batch_size, crop_count, -1)
                clip_image_features = normalize_features(clip_image_features.mean(dim=1))
                clip_scores = clip_image_features @ text_features.T
                if args.score_normalization == "zscore":
                    supervised_scores = row_zscore(supervised_scores)
                    clip_scores = row_zscore(clip_scores)
                supervised_scores = supervised_scores / args.supervised_temp
                clip_scores = clip_scores / args.clip_temp
                fused_scores = supervised_weight * supervised_scores + args.text_weight * clip_scores
                scores, pred = fused_scores.max(dim=1)
                for image_id, idx, score in zip(image_ids, pred.cpu().tolist(), scores.cpu().tolist()):
                    rows.append({"image_id": image_id, "prediction": class_names[int(idx)], "score": score})
                if args.topk_csv:
                    k = min(args.topk_k, fused_scores.shape[1])
                    top_scores, top_indices = fused_scores.topk(k, dim=1)
                    for image_id, label, indices, values in zip(
                        image_ids,
                        labels,
                        top_indices.cpu().tolist(),
                        top_scores.cpu().tolist(),
                    ):
                        margin = values[0] - values[1] if len(values) > 1 else 0.0
                        top_classes = [class_names[int(idx)] for idx in indices]
                        topk_rows.append(
                            {
                                "image_id": image_id,
                                "true_label": label,
                                "prediction": top_classes[0],
                                "margin_top1_top2": f"{margin:.8f}",
                                "top_classes": "|".join(top_classes),
                                "top_scores": "|".join(f"{float(value):.8f}" for value in values),
                            }
                        )
        return rows, topk_rows

    total_rows = sum(1 for _ in csv.DictReader(args.manifest.open("r", encoding="utf-8", newline="")))
    if args.max_samples:
        total_rows = min(total_rows, args.max_samples)

    if args.shard_size > 0:
        shard_dir = args.shard_dir or (args.out_csv.parent / "seen_fusion_shards")
        shard_dir.mkdir(parents=True, exist_ok=True)
        expected_shards = []
        for shard_index, start_index in enumerate(range(0, total_rows, args.shard_size)):
            end_index = min(start_index + args.shard_size, total_rows)
            shard_csv = shard_dir / f"shard_{shard_index:05d}.csv"
            shard_json = shard_dir / f"shard_{shard_index:05d}.json"
            shard_topk = shard_dir / f"shard_{shard_index:05d}_top{args.topk_k}.csv" if args.topk_csv else None
            expected_shards.append((shard_csv, shard_json, shard_topk))
            if args.resume and shard_csv.exists() and (shard_topk is None or shard_topk.exists()):
                print(f"Skipping existing shard {shard_index:05d} rows {start_index}:{end_index}", flush=True)
                continue
            shard_rows, shard_topk_rows = predict_slice(
                start_index,
                end_index,
                desc=f"fusion_predict[{shard_index:05d}]",
            )
            write_prediction_outputs(shard_rows, shard_topk_rows, shard_csv, shard_json, shard_topk)
            print(
                json.dumps(
                    {
                        "shard": shard_index,
                        "start_index": start_index,
                        "end_index": end_index,
                        "rows": len(shard_rows),
                        "out_csv": str(shard_csv),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        rows = []
        topk_rows = []
        for shard_csv, _shard_json, shard_topk in expected_shards:
            if not shard_csv.exists():
                raise RuntimeError(f"Missing shard prediction file: {shard_csv}")
            rows.extend(read_csv_rows(shard_csv))
            if args.topk_csv:
                if shard_topk is None or not shard_topk.exists():
                    raise RuntimeError(f"Missing shard topk file: {shard_topk}")
                topk_rows.extend(read_csv_rows(shard_topk))
    else:
        rows, topk_rows = predict_slice(0, total_rows, desc="fusion_predict")

    write_prediction_outputs(rows, topk_rows, args.out_csv, args.out_json, args.topk_csv)

    summary = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "supervised_checkpoint": str(args.supervised_checkpoint),
        "supervised_model": supervised_model_name,
        "text_features": str(args.text_features),
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_precision": args.clip_precision,
        "text_weight": args.text_weight,
        "supervised_weight": supervised_weight,
        "supervised_temp": args.supervised_temp,
        "clip_temp": args.clip_temp,
        "tta_crops": args.tta_crops,
        "score_normalization": args.score_normalization,
        "out_csv": str(args.out_csv),
        "out_json": str(args.out_json) if args.out_json else None,
        "topk_csv": str(args.topk_csv) if args.topk_csv else None,
        "topk_k": args.topk_k if args.topk_csv else None,
        "shard_size": args.shard_size,
        "shard_dir": str(args.shard_dir) if args.shard_dir else None,
        "resume": bool(args.resume),
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    (args.out_csv.parent / "seen_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
