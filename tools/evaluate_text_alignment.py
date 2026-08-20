from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def make_prompts(class_name: str, description: str, mode: str, desc_words: int) -> List[str]:
    desc = " ".join((description or "").split())
    sent = first_sentence(desc)
    short_desc = truncate_words(desc, desc_words)
    genus = class_name.split()[0] if class_name.split() else class_name

    if mode == "name":
        return [class_name]
    if mode == "taxon":
        return [
            class_name,
            f"a photo of {class_name}",
            f"a photo of the species {class_name}",
        ]
    if mode == "fish":
        return [
            f"a photo of {class_name}, a fish species.",
            f"a close-up photo of the fish species {class_name}.",
            f"an underwater image of {class_name}.",
            f"{class_name}.",
        ]
    if mode == "genus":
        return [
            class_name,
            f"a fish in the genus {genus}.",
            f"a photo of a {genus} fish.",
        ]
    if mode == "desc_sentence":
        prompts = [class_name]
        if sent:
            prompts.append(f"{class_name}. {sent}")
            prompts.append(f"a fish species with these visual traits: {sent}")
        return prompts
    if mode == "desc_short":
        prompts = [class_name]
        if short_desc:
            prompts.append(f"{class_name}, a fish species. {short_desc}")
            prompts.append(f"a photo of a fish with these traits: {short_desc}")
        return prompts
    if mode == "all":
        prompts: List[str] = []
        for inner in ["name", "taxon", "fish", "genus", "desc_sentence", "desc_short"]:
            prompts.extend(make_prompts(class_name, desc, inner, desc_words))
        return list(dict.fromkeys(prompts))
    raise ValueError(f"Unknown prompt mode: {mode}")


class ImageLabelDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, preprocess, max_samples: int = 0):
        self.image_root = image_root
        self.preprocess = preprocess
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
            x = self.preprocess(image.convert("RGB"))
        return x, image_id, label


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_images(model, loader, device: torch.device, amp: bool) -> tuple[torch.Tensor, List[str], List[str]]:
    features: List[torch.Tensor] = []
    image_ids: List[str] = []
    labels: List[str] = []
    with torch.inference_mode():
        for x, batch_image_ids, batch_labels in tqdm(loader, desc="encode_images"):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model.encode_image(x)
            features.append(normalize(encoded.float()).cpu())
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)
    return torch.cat(features, dim=0), image_ids, labels


def encode_text_features(
    model,
    tokenizer,
    classes: Sequence[str],
    descriptions: Dict[str, str],
    mode: str,
    desc_words: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    sum_features = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    prompt_buffer: List[str] = []
    index_buffer: List[int] = []

    for class_index, class_name in enumerate(tqdm(classes, desc=f"prepare_{mode}")):
        prompts = make_prompts(class_name, descriptions.get(class_name, ""), mode, desc_words)
        prompt_buffer.extend(prompts)
        index_buffer.extend([class_index] * len(prompts))

    with torch.inference_mode():
        for start in tqdm(range(0, len(prompt_buffer), batch_size), desc=f"encode_text_{mode}"):
            prompts = prompt_buffer[start : start + batch_size]
            class_indices = index_buffer[start : start + batch_size]
            tokens = tokenizer(prompts).to(device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                encoded = model.encode_text(tokens)
            encoded = normalize(encoded.float()).cpu()
            if sum_features is None:
                sum_features = torch.zeros((len(classes), encoded.shape[1]), dtype=torch.float32)
            for row_index, class_index in enumerate(class_indices):
                sum_features[class_index] += encoded[row_index]
                counts[class_index] += 1

    if sum_features is None:
        raise RuntimeError("No text prompts were encoded")
    features = sum_features / counts[:, None].clamp_min(1)
    return normalize(features)


def evaluate(image_features: torch.Tensor, labels: Sequence[str], text_features: torch.Tensor, classes: Sequence[str]) -> Dict[str, float]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    label_indices = torch.tensor([class_to_idx.get(label, -1) for label in labels], dtype=torch.long)
    valid = label_indices >= 0
    if not bool(valid.any()):
        return {"top1": 0.0, "top5": 0.0, "labeled": 0}
    logits = image_features @ text_features.T
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    valid_top5 = top5[valid]
    valid_labels = label_indices[valid]
    top1 = (valid_top5[:, 0] == valid_labels).float().mean().item()
    top5_acc = (valid_top5 == valid_labels[:, None]).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": top5_acc, "labeled": int(valid.sum().item())}


def parse_modes(value: str) -> List[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise ValueError("At least one mode is required")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("work/supervised_splits/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--classes-json", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--modes", default="name,taxon,fish,desc_sentence,desc_short,all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--desc-words", type=int, default=45)
    parser.add_argument("--clip-precision", default="fp32")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    import open_clip

    classes = load_classes(args.classes_json)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8")) if args.descriptions.exists() else {}
    modes = parse_modes(args.modes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = None if str(args.pretrained).lower() in {"none", "null", ""} else args.pretrained

    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        precision=args.clip_precision,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval()

    ds = ImageLabelDataset(args.manifest, args.image_root, preprocess_val, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    amp = (not args.no_amp) and device.type == "cuda"
    image_features, image_ids, labels = encode_images(model, loader, device, amp)

    rows = []
    for mode in modes:
        text_features = encode_text_features(
            model,
            tokenizer,
            classes,
            descriptions,
            mode,
            args.desc_words,
            args.text_batch_size,
            device,
            amp,
        )
        metrics = evaluate(image_features, labels, text_features, classes)
        row = {
            "mode": mode,
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "labeled": metrics["labeled"],
            "classes": len(classes),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["mode", "top1", "top5", "labeled", "classes"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(args.manifest),
        "classes_json": str(args.classes_json),
        "model": args.model,
        "pretrained": args.pretrained,
        "clip_precision": args.clip_precision,
        "modes": modes,
        "rows": rows,
        "images": len(image_ids),
        "device": str(device),
        "gpu": gpu_snapshot(),
        "env": environment_report(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
