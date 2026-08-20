from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_hf_zeroshot_topk import (
    call_image_features,
    call_text_features,
    l2_normalize,
    make_prompts,
    move_to_device,
    parse_modes,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_topk(path: Path, max_rows: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


class TopkImages(Dataset):
    def __init__(self, rows: Sequence[dict], image_root: Path):
        self.rows = list(rows)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_id = row["image_id"]
        with Image.open(self.image_root / image_id) as image:
            image = image.convert("RGB")
        return image, image_id


def pil_collate(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)


def encode_unique_text(
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
) -> tuple[torch.Tensor, dict[str, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    prompt_buffer: list[str] = []
    index_buffer: list[int] = []
    for class_name in classes:
        seen: dict[str, None] = {}
        for mode in modes:
            for prompt in make_prompts(class_name, descriptions.get(class_name, ""), mode, desc_words):
                seen.setdefault(prompt, None)
        prompt_buffer.extend(seen.keys())
        index_buffer.extend([class_to_idx[class_name]] * len(seen))

    sums: torch.Tensor | None = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    for start in tqdm(range(0, len(prompt_buffer), batch_size), desc="encode_text_topk"):
        prompts = prompt_buffer[start : start + batch_size]
        indices = index_buffer[start : start + batch_size]
        batch = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
        batch = move_to_device(batch, device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            feats = call_text_features(model, batch)
        feats = l2_normalize(feats.float()).cpu()
        if sums is None:
            sums = torch.zeros((len(classes), feats.shape[1]), dtype=torch.float32)
        for row_idx, class_idx in enumerate(indices):
            sums[class_idx] += feats[row_idx]
            counts[class_idx] += 1
    if sums is None:
        raise RuntimeError("No text features encoded")
    return l2_normalize(sums / counts[:, None].clamp_min(1)), class_to_idx


def encode_image_map(
    *,
    model,
    processor,
    rows: Sequence[dict],
    image_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> dict[str, torch.Tensor]:
    ds = TopkImages(rows, image_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pil_collate)
    image_map: dict[str, torch.Tensor] = {}
    for images, image_ids in tqdm(loader, desc="encode_images_topk"):
        batch = processor(images=images, return_tensors="pt")
        batch = move_to_device(batch, device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            feats = call_image_features(model, batch)
        feats = l2_normalize(feats.float()).cpu()
        for image_id, feat in zip(image_ids, feats):
            image_map[image_id] = feat
    return image_map


def metrics(indices: torch.Tensor, labels: Sequence[str], predictions: Sequence[Sequence[str]]) -> dict[str, float | int]:
    top1 = 0
    top5 = 0
    top20 = 0
    changed = 0
    wins = 0
    losses = 0
    labeled = 0
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        labeled += 1
        preds = list(predictions[row_idx])
        base = preds[0]
        final = preds[int(indices[row_idx, 0].item())]
        changed += int(base != final)
        base_ok = base == label
        final_ok = final == label
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [preds[int(i)] for i in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
        top20 += int(rank <= 20)
    return {
        "labeled": labeled,
        "top1": top1 / labeled if labeled else 0.0,
        "top5": top5 / labeled if labeled else 0.0,
        "top20": top20 / labeled if labeled else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-modes", default="fish")
    parser.add_argument("--desc-words", type=int, default=45)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--weight-grid", default="-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    rows = load_topk(args.topk_jsonl, args.max_rows)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8")) if args.descriptions.exists() else {}
    modes = parse_modes(args.prompt_modes)
    classes = sorted({pred for row in rows for pred in row["predictions"]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    text_features, class_to_idx = encode_unique_text(
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
    image_map = encode_image_map(
        model=model,
        processor=processor,
        rows=rows,
        image_root=args.image_root,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=not args.no_amp,
    )

    image_ids = [row["image_id"] for row in rows]
    labels = [row.get("label", "") for row in rows]
    predictions = [list(row["predictions"]) for row in rows]
    base_scores = torch.tensor([row["scores"] for row in rows], dtype=torch.float32)
    adapter_scores = torch.zeros_like(base_scores)
    for row_idx, row in enumerate(tqdm(rows, desc="score_base_topk")):
        image_feat = image_map[row["image_id"]]
        cand_idx = torch.tensor([class_to_idx[pred] for pred in row["predictions"]], dtype=torch.long)
        adapter_scores[row_idx] = image_feat @ text_features[cand_idx].T

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "adapter_topk_scores.pt"
    torch.save(
        {
            "image_ids": image_ids,
            "predictions": predictions,
            "base_scores": base_scores,
            "adapter_scores": adapter_scores,
            "labels": labels,
        },
        score_path,
    )

    base_indices = base_scores.argsort(dim=1, descending=True)
    sweep = []
    for weight in [float(part.strip()) for part in args.weight_grid.split(",") if part.strip()]:
        final = base_scores + weight * row_zscore(adapter_scores)
        indices = final.argsort(dim=1, descending=True)
        row = {"weight": weight, **metrics(indices, labels, predictions)}
        sweep.append(row)
    best = max(sweep, key=lambda item: (item["top1"], item["net"], -item["changed"]))
    summary = {
        "topk_jsonl": str(args.topk_jsonl),
        "model_id": args.model_id,
        "prompt_modes": modes,
        "rows": len(rows),
        "unique_topk_classes": len(classes),
        "base": metrics(base_indices, labels, predictions),
        "best": best,
        "sweep": sweep,
        "score_file": str(score_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
