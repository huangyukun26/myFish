from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--manifest", type=Path, default=Path("work/supervised_splits/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--prompt", default="a photo of Amphiprion ocellaris, a fish species.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import open_clip

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pretrained = None if str(args.pretrained).lower() in {"none", "null", ""} else args.pretrained

    print(json.dumps({"stage": "load_start", "model": args.model, "precision": args.precision, "device": str(device)}), flush=True)
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        precision=args.precision,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval()
    print(json.dumps({"stage": "load_done", "gpu": gpu_snapshot()}), flush=True)

    with args.manifest.open("r", encoding="utf-8", newline="") as fp:
        row = next(csv.DictReader(fp))
    image_path = args.image_root / row["image_id"]
    with Image.open(image_path) as image:
        x = preprocess_val(image.convert("RGB")).unsqueeze(0).to(device)
    tokens = tokenizer([args.prompt]).to(device)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        image_features = model.encode_image(x)
        text_features = model.encode_text(tokens)
    image_features = image_features.float()
    text_features = text_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = float((image_features @ text_features.T).item())

    summary = {
        "stage": "encode_done",
        "image_id": row["image_id"],
        "label": row.get("label", ""),
        "prompt": args.prompt,
        "similarity": similarity,
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
