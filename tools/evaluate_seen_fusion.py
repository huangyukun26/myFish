from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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


class FusionDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        supervised_transform,
        clip_transform,
        max_samples: int = 0,
        tta_crops: str = "none",
    ):
        self.image_root = image_root
        self.supervised_transform = supervised_transform
        self.clip_transform = clip_transform
        self.tta_crops = tta_crops
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError(f"No rows in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        label = row.get("label", "")
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            supervised_x = torch.stack([self.supervised_transform(variant) for variant in variants])
            clip_x = torch.stack([self.clip_transform(variant) for variant in variants])
        return supervised_x, clip_x, image_id, label


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


def topk_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    top5 = scores.topk(min(5, scores.shape[1]), dim=1).indices
    return {
        "top1": (top5[:, 0] == labels).float().mean().item(),
        "top5": (top5 == labels[:, None]).any(dim=1).float().mean().item(),
    }


def topk_summary(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> Dict[str, float]:
    k = min(k, scores.shape[1])
    topk = scores.topk(k, dim=1).indices
    top1_correct = topk[:, 0] == labels
    topk_correct = (topk == labels[:, None]).any(dim=1)
    return {
        "top1": top1_correct.float().mean().item(),
        f"top{k}": topk_correct.float().mean().item(),
        f"correct_in_top{k}_not_top1": (topk_correct & ~top1_correct).float().mean().item(),
        f"correct_in_top{k}_not_top1_count": int((topk_correct & ~top1_correct).sum().item()),
        "samples": int(labels.numel()),
    }


def prediction_sets(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> Dict[str, torch.Tensor]:
    k = min(k, scores.shape[1])
    topk = scores.topk(k, dim=1).indices
    return {
        "top1": topk[:, 0] == labels,
        f"top{k}": (topk == labels[:, None]).any(dim=1),
    }


def write_topk_csv(
    path: Path,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_ids: Sequence[str],
    class_names: Sequence[str],
    k: int = 5,
) -> None:
    k = min(k, scores.shape[1])
    top_scores, top_indices = scores.topk(k, dim=1)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "image_id",
            "true_label",
            "hit_top1",
            f"hit_top{k}",
            "margin_top1_top2",
            "top_classes",
            "top_scores",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, image_id in enumerate(image_ids):
            true_idx = int(labels[row_idx].item())
            indices = [int(value) for value in top_indices[row_idx].tolist()]
            values = [float(value) for value in top_scores[row_idx].tolist()]
            margin = values[0] - values[1] if len(values) > 1 else 0.0
            writer.writerow(
                {
                    "image_id": image_id,
                    "true_label": class_names[true_idx],
                    "hit_top1": int(indices[0] == true_idx),
                    f"hit_top{k}": int(true_idx in indices),
                    "margin_top1_top2": f"{margin:.8f}",
                    "top_classes": "|".join(class_names[idx] for idx in indices),
                    "top_scores": "|".join(f"{value:.8f}" for value in values),
                }
            )


def parse_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("work/supervised_splits/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--supervised-checkpoint", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--supervised-image-size", type=int, default=224)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--text-weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--supervised-temps", default="1.0")
    parser.add_argument("--clip-temps", default="1.0")
    parser.add_argument("--score-normalization", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--clip-precision", default="fp32")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-topk", action="store_true", help="Save topK CSV files and branch complement summary.")
    parser.add_argument("--topk-k", type=int, default=5)
    args = parser.parse_args()

    from torchvision import transforms
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_classes(args.class_map)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

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
    ds = FusionDataset(
        args.manifest,
        args.image_root,
        supervised_transform,
        clip_preprocess_val,
        max_samples=args.max_samples,
        tta_crops=args.tta_crops,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    amp = (not args.no_amp) and device.type == "cuda"
    supervised_batches: List[torch.Tensor] = []
    clip_batches: List[torch.Tensor] = []
    label_indices: List[int] = []
    image_ids: List[str] = []

    with torch.inference_mode():
        for supervised_x, clip_x, batch_image_ids, labels in tqdm(loader, desc="score_images"):
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
            supervised_batches.append(supervised_scores.cpu())
            clip_batches.append(clip_scores.cpu())
            image_ids.extend(batch_image_ids)
            label_indices.extend(class_to_idx[label] for label in labels)

    supervised_scores = torch.cat(supervised_batches, dim=0)
    clip_scores = torch.cat(clip_batches, dim=0)
    labels = torch.tensor(label_indices, dtype=torch.long)
    if args.score_normalization == "zscore":
        supervised_base = row_zscore(supervised_scores)
        clip_base = row_zscore(clip_scores)
    else:
        supervised_base = supervised_scores
        clip_base = clip_scores

    rows = []
    for supervised_temp in parse_floats(args.supervised_temps):
        for clip_temp in parse_floats(args.clip_temps):
            supervised_scaled = supervised_base / supervised_temp
            clip_scaled = clip_base / clip_temp
            for text_weight in parse_floats(args.text_weights):
                combined = (1.0 - text_weight) * supervised_scaled + text_weight * clip_scaled
                metrics = topk_metrics(combined, labels)
                row = {
                    "text_weight": text_weight,
                    "supervised_weight": 1.0 - text_weight,
                    "supervised_temp": supervised_temp,
                    "clip_temp": clip_temp,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "samples": len(labels),
                    "score_normalization": args.score_normalization,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    best_top1 = max(rows, key=lambda row: (row["top1"], row["top5"]))
    best_top5 = max(rows, key=lambda row: (row["top5"], row["top1"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "text_weight",
                "supervised_weight",
                "supervised_temp",
                "clip_temp",
                "top1",
                "top5",
                "samples",
                "score_normalization",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(args.manifest),
        "supervised_checkpoint": str(args.supervised_checkpoint),
        "supervised_model": supervised_model_name,
        "text_features": str(args.text_features),
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_precision": args.clip_precision,
        "tta_crops": args.tta_crops,
        "score_normalization": args.score_normalization,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "samples": len(labels),
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    if args.save_topk:
        best_text_weight = float(best_top1["text_weight"])
        best_supervised_temp = float(best_top1["supervised_temp"])
        best_clip_temp = float(best_top1["clip_temp"])
        supervised_for_best = supervised_base / best_supervised_temp
        clip_for_best = clip_base / best_clip_temp
        fusion_best = (1.0 - best_text_weight) * supervised_for_best + best_text_weight * clip_for_best

        k = args.topk_k
        variants = {
            "supervised": supervised_for_best,
            "clip": clip_for_best,
            "fusion_best_top1": fusion_best,
        }
        topk_report: Dict[str, Dict[str, float]] = {}
        sets = {name: prediction_sets(scores, labels, k=k) for name, scores in variants.items()}
        for name, scores in variants.items():
            write_topk_csv(args.out_dir / f"top{k}_{name}.csv", scores, labels, image_ids, class_names, k=k)
            topk_report[name] = topk_summary(scores, labels, k=k)

        sup_top1 = sets["supervised"]["top1"]
        clip_top1 = sets["clip"]["top1"]
        fusion_top1 = sets["fusion_best_top1"]["top1"]
        sup_topk = sets["supervised"][f"top{k}"]
        clip_topk = sets["clip"][f"top{k}"]
        fusion_topk = sets["fusion_best_top1"][f"top{k}"]
        topk_report["complement"] = {
            "supervised_top1_clip_wrong_count": int((sup_top1 & ~clip_top1).sum().item()),
            "clip_top1_supervised_wrong_count": int((clip_top1 & ~sup_top1).sum().item()),
            "both_top1_correct_count": int((sup_top1 & clip_top1).sum().item()),
            "neither_top1_correct_count": int((~sup_top1 & ~clip_top1).sum().item()),
            "fusion_top1_correct_when_either_branch_top1_correct_count": int(
                (fusion_top1 & (sup_top1 | clip_top1)).sum().item()
            ),
            "fusion_top1_wrong_when_either_branch_top1_correct_count": int(
                (~fusion_top1 & (sup_top1 | clip_top1)).sum().item()
            ),
            f"supervised_top{k}_clip_miss_count": int((sup_topk & ~clip_topk).sum().item()),
            f"clip_top{k}_supervised_miss_count": int((clip_topk & ~sup_topk).sum().item()),
            f"fusion_top{k}_correct_count": int(fusion_topk.sum().item()),
        }
        summary["topk_report"] = topk_report
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
