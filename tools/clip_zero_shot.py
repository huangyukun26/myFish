from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import ImageOps
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PredictionDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, preprocess, max_samples: int = 0, tta_crops: str = "none"):
        self.manifest = manifest
        self.image_root = image_root
        self.preprocess = preprocess
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
        path = self.image_root / image_id
        with Image.open(path) as image:
            image = image.convert("RGB")
            variants = make_pil_variants(image, self.tta_crops)
            x = torch.stack([self.preprocess(variant) for variant in variants])
        label = row.get("label", "")
        return x, image_id, label


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


def load_candidate_classes(path: Optional[Path], text_classes: List[str]) -> List[str]:
    if path is None:
        return text_classes
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Override model name; defaults to value stored in text feature file.")
    parser.add_argument("--pretrained", default=None, help="Override pretrained tag; defaults to value stored in text feature file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tta-crops", choices=["none", "hflip", "fivecrop", "tencrop"], default="none")
    parser.add_argument("--clip-precision", default="fp32")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import open_clip

    payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    text_classes = payload["classes"]
    text_features = payload["features"].float()
    class_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    candidates = load_candidate_classes(args.candidate_classes, text_classes)
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidate classes missing from text features; first={missing[:5]}")
    candidate_indices = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    candidate_features = text_features[candidate_indices]
    candidate_features = normalize_features(candidate_features)

    model_name = args.model or payload["model"]
    pretrained = args.pretrained or payload["pretrained"]
    pretrained_for_open_clip = None if str(pretrained).lower() in {"none", "null", ""} else pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_for_open_clip,
        precision=args.clip_precision,
        device=device,
    )
    model = model.eval()
    candidate_features = candidate_features.to(device)

    ds = PredictionDataset(args.manifest, args.image_root, preprocess_val, max_samples=args.max_samples, tta_crops=args.tta_crops)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    correct = 0
    labeled = 0
    amp = (not args.no_amp) and device.type == "cuda"
    with torch.inference_mode():
        for x, image_ids, labels in tqdm(loader, desc="clip_predict"):
            batch_size, crop_count = x.shape[:2]
            x = x.to(device, non_blocking=True)
            x = x.flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                image_features = model.encode_image(x)
            image_features = normalize_features(image_features.float()).view(batch_size, crop_count, -1)
            image_features = normalize_features(image_features.mean(dim=1))
            logits = image_features @ candidate_features.T
            scores, pred_idx = logits.max(dim=1)
            pred_idx = pred_idx.cpu().tolist()
            scores = scores.cpu().tolist()
            for image_id, label, idx, score in zip(image_ids, labels, pred_idx, scores):
                pred = candidates[idx]
                rows.append({"image_id": image_id, "prediction": pred, "score": score, "label": label})
                if label:
                    labeled += 1
                    correct += int(pred == label)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "score", "label"])
        writer.writeheader()
        writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({row["image_id"]: row["prediction"] for row in rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "manifest": str(args.manifest),
        "rows": len(rows),
        "candidate_classes": len(candidates),
        "labeled": labeled,
        "accuracy": (correct / labeled) if labeled else None,
        "out_csv": str(args.out_csv),
        "out_json": str(args.out_json) if args.out_json else None,
        "tta_crops": args.tta_crops,
        "clip_precision": args.clip_precision,
        "amp": amp,
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    (args.out_csv.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
