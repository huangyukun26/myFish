from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot
from train_supervised import build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestImageDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, image_size: int, tta: str, max_samples: int = 0):
        self.image_root = image_root
        self.tta = tta
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples:
            self.rows = self.rows[:max_samples]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        with Image.open(self.image_root / image_id) as image:
            x = self.transform(image.convert("RGB"))
        if self.tta == "none":
            x = x.unsqueeze(0)
        elif self.tta == "hflip":
            x = torch.stack([x, TF.hflip(x)], dim=0)
        else:
            raise ValueError(f"Unknown TTA mode: {self.tta}")
        return x, image_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-topk-csv", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    class_to_idx = json.loads(args.class_map.read_text(encoding="utf-8"))
    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = payload.get("args", {})
    model_name = ckpt_args.get("model", "resnet18")
    num_classes = int(ckpt_args.get("num_classes", len(idx_to_class)))
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()

    ds = TestImageDataset(args.manifest, args.image_root, args.image_size, args.tta, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    rows = []
    topk_rows = []
    with torch.inference_mode():
        for x, image_ids in tqdm(loader, desc="supervised_predict"):
            batch, views = x.shape[:2]
            x = x.view(batch * views, *x.shape[2:]).to(device, non_blocking=True)
            logits = model(x).view(batch, views, -1).mean(dim=1)
            prob = torch.softmax(logits, dim=1)
            scores, pred = prob.max(dim=1)
            for image_id, idx, score in zip(image_ids, pred.cpu().tolist(), scores.cpu().tolist()):
                rows.append({"image_id": image_id, "prediction": idx_to_class[int(idx)], "score": score})
            if args.out_topk_csv:
                k = min(args.topk, logits.shape[1])
                top_scores, top_indices = logits.topk(k, dim=1)
                for image_id, indices, values in zip(image_ids, top_indices.cpu().tolist(), top_scores.cpu().tolist()):
                    classes = [idx_to_class[int(idx)] for idx in indices]
                    margin = float(values[0] - values[1]) if len(values) > 1 else 0.0
                    topk_rows.append(
                        {
                            "image_id": image_id,
                            "prediction": classes[0],
                            "margin_top1_top2": margin,
                            "top_classes": "|".join(classes),
                            "top_scores": "|".join(f"{float(value):.8f}" for value in values),
                        }
                    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "score"])
        writer.writeheader()
        writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({row["image_id"]: row["prediction"] for row in rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.out_topk_csv:
        args.out_topk_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_topk_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["image_id", "prediction", "margin_top1_top2", "top_classes", "top_scores"],
            )
            writer.writeheader()
            writer.writerows(topk_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "rows": len(rows),
        "out_csv": str(args.out_csv),
        "out_json": str(args.out_json) if args.out_json else None,
        "out_topk_csv": str(args.out_topk_csv) if args.out_topk_csv else None,
        "topk": args.topk,
        "tta": args.tta,
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    (args.out_csv.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
