from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rows(path: Path, max_samples: int = 0) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def truncate_words(text: str, max_words: int) -> str:
    return " ".join((text or "").split()[:max_words])


def make_prompts(class_name: str, description: str, mode: str, desc_words: int) -> list[str]:
    genus = class_name.split()[0] if class_name.split() else class_name
    desc = " ".join((description or "").split())
    sent = first_sentence(desc)
    short = truncate_words(desc, desc_words)
    if mode == "name":
        return [class_name]
    if mode == "fish":
        return [
            f"a photo of {class_name}, a fish species.",
            f"a close-up photo of the fish species {class_name}.",
            f"an underwater image of {class_name}.",
            class_name,
        ]
    if mode == "taxon":
        return [class_name, f"a photo of {class_name}", f"a photo of the species {class_name}"]
    if mode == "genus":
        return [class_name, f"a fish in the genus {genus}.", f"a photo of a {genus} fish."]
    if mode == "desc_sentence":
        prompts = [class_name]
        if sent:
            prompts.extend([f"{class_name}. {sent}", f"a fish species with these visual traits: {sent}"])
        return prompts
    if mode == "desc_short":
        prompts = [class_name]
        if short:
            prompts.extend([f"{class_name}, a fish species. {short}", f"a photo of a fish with these traits: {short}"])
        return prompts
    raise ValueError(f"Unknown prompt mode: {mode}")


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise ValueError("At least one prompt mode is required")
    return modes


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class ImageRows(Dataset):
    def __init__(self, rows: Sequence[dict[str, str]], image_root: Path):
        self.rows = list(rows)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
        return image, image_id, row.get("label", "")


def pil_collate(batch):
    images, image_ids, labels = zip(*batch)
    return list(images), list(image_ids), list(labels)


def move_to_device(batch, device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def call_image_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        try:
            return model.get_image_features(**inputs)
        except TypeError:
            return model.get_image_features(pixel_values=inputs["pixel_values"])
    outputs = model(**inputs)
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0]
    raise RuntimeError("Could not extract image features from model outputs")


def call_text_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        try:
            return model.get_text_features(**inputs)
        except TypeError:
            keep = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "position_ids"}}
            return model.get_text_features(**keep)
    outputs = model(**inputs)
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0]
    raise RuntimeError("Could not extract text features from model outputs")


def encode_text(
    *,
    model,
    processor,
    classes: Sequence[str],
    descriptions: dict[str, str],
    modes: Sequence[str],
    desc_words: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    prompts: list[str] = []
    prompt_class_indices: list[int] = []
    for class_idx, class_name in enumerate(classes):
        seen_prompts: dict[str, None] = {}
        for mode in modes:
            for prompt in make_prompts(class_name, descriptions.get(class_name, ""), mode, desc_words):
                seen_prompts.setdefault(prompt, None)
        prompts.extend(seen_prompts.keys())
        prompt_class_indices.extend([class_idx] * len(seen_prompts))

    sums: torch.Tensor | None = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    for start in tqdm(range(0, len(prompts), batch_size), desc="encode_text"):
        batch_prompts = prompts[start : start + batch_size]
        class_indices = prompt_class_indices[start : start + batch_size]
        encoded = processor(text=batch_prompts, padding=True, truncation=True, return_tensors="pt")
        encoded = move_to_device(encoded, device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            feats = call_text_features(model, encoded)
        feats = l2_normalize(feats.float()).cpu()
        if sums is None:
            sums = torch.zeros((len(classes), feats.shape[1]), dtype=torch.float32)
        for row_idx, class_idx in enumerate(class_indices):
            sums[class_idx] += feats[row_idx]
            counts[class_idx] += 1
    if sums is None:
        raise RuntimeError("No text features encoded")
    return l2_normalize(sums / counts[:, None].clamp_min(1))


def encode_images(
    *,
    model,
    processor,
    rows: Sequence[dict[str, str]],
    image_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, list[str], list[str]]:
    ds = ImageRows(rows, image_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pil_collate)
    features: list[torch.Tensor] = []
    image_ids: list[str] = []
    labels: list[str] = []
    for images, batch_image_ids, batch_labels in tqdm(loader, desc="encode_images"):
        encoded = processor(images=images, return_tensors="pt")
        encoded = move_to_device(encoded, device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            feats = call_image_features(model, encoded)
        features.append(l2_normalize(feats.float()).cpu())
        image_ids.extend(batch_image_ids)
        labels.extend(batch_labels)
    return torch.cat(features, dim=0), image_ids, labels


def write_topk(
    *,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_ids: Sequence[str],
    labels: Sequence[str],
    classes: Sequence[str],
    out_path: Path,
    topk: int,
    score_batch_size: int,
) -> dict[str, float | int | str]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    k = min(topk, len(classes))
    total = 0
    labeled = 0
    top1 = 0
    top5 = 0
    top20 = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text_t = text_features.T.contiguous()
    with out_path.open("w", encoding="utf-8") as fp:
        for start in tqdm(range(0, len(image_ids), score_batch_size), desc="score_topk"):
            end = min(start + score_batch_size, len(image_ids))
            logits = image_features[start:end] @ text_t
            scores, indices = logits.topk(k, dim=1)
            for row_idx in range(end - start):
                image_id = image_ids[start + row_idx]
                label = labels[start + row_idx]
                preds = [classes[int(idx)] for idx in indices[row_idx].tolist()]
                row_scores = [float(v) for v in scores[row_idx].tolist()]
                true_rank: int | str = ""
                if label:
                    labeled += 1
                    true_idx = class_to_idx.get(label)
                    if true_idx is not None:
                        matches = (indices[row_idx] == true_idx).nonzero(as_tuple=False)
                        true_rank = int(matches[0, 0].item()) + 1 if matches.numel() else k + 1
                        top1 += int(true_rank == 1)
                        top5 += int(true_rank <= 5)
                        top20 += int(true_rank <= 20)
                total += 1
                fp.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "label": label,
                            "predictions": preds,
                            "scores": row_scores,
                            "true_rank": true_rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    summary: dict[str, float | int | str] = {
        "rows": total,
        "labeled": labeled,
        "top1": top1 / labeled if labeled else 0.0,
        "top5": top5 / labeled if labeled else 0.0,
        "top20": top20 / labeled if labeled else 0.0,
        "out": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-modes", default="fish")
    parser.add_argument("--desc-words", type=int, default=45)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    rows = load_rows(args.manifest, args.max_samples)
    classes = load_classes(args.candidate_classes)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8")) if args.descriptions.exists() else {}
    modes = parse_modes(args.prompt_modes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    text_features = encode_text(
        model=model,
        processor=processor,
        classes=classes,
        descriptions=descriptions,
        modes=modes,
        desc_words=args.desc_words,
        batch_size=args.text_batch_size,
        device=device,
        amp=not args.no_amp,
    )
    image_features, image_ids, labels = encode_images(
        model=model,
        processor=processor,
        rows=rows,
        image_root=args.image_root,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=not args.no_amp,
    )
    out_path = args.out_dir / "primary_topk.jsonl"
    summary = write_topk(
        image_features=image_features,
        text_features=text_features,
        image_ids=image_ids,
        labels=labels,
        classes=classes,
        out_path=out_path,
        topk=args.topk,
        score_batch_size=args.score_batch_size,
    )
    summary.update(
        {
            "model_id": args.model_id,
            "prompt_modes": modes,
            "candidate_classes": str(args.candidate_classes),
            "manifest": str(args.manifest),
        }
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
