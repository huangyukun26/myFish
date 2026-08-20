"""Encode official FishNet candidate text with a local SigLIP2 checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.keys()) if isinstance(data, dict) else list(data)


def first_sentence(value: str) -> str:
    value = " ".join((value or "").split())
    return re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0] if value else ""


def prompts_for(name: str, description: str, mode: str) -> list[str]:
    if mode == "name":
        return [name]
    if mode == "taxon":
        return [name, f"a photo of {name}", f"a photo of the species {name}"]
    if mode == "fish":
        return [f"a photo of {name}, a fish species.", f"an underwater photo of {name}.", name]
    if mode == "desc_sentence":
        sentence = first_sentence(description)
        return [name, f"a photo of {name}, a fish species.", f"{name}. {sentence}"] if sentence else [name]
    raise ValueError(f"Unknown prompt mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--prompt-mode", choices=["name", "taxon", "fish", "desc_sentence"], default="desc_sentence")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    classes = load_classes(args.classes)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    prompts: list[str] = []
    owners: list[int] = []
    for idx, name in enumerate(classes):
        text_list = prompts_for(name, descriptions.get(name, ""), args.prompt_mode)
        prompts.extend(text_list)
        owners.extend([idx] * len(text_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    sums: torch.Tensor | None = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="siglip2_text"):
            batch = prompts[start : start + args.batch_size]
            inputs = processor(text=batch, padding="max_length", max_length=64, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            encoded = F.normalize(model.get_text_features(**inputs).float(), dim=1).cpu()
            if sums is None:
                sums = torch.zeros((len(classes), encoded.shape[1]), dtype=torch.float32)
            for row, owner in enumerate(owners[start : start + len(batch)]):
                sums[owner] += encoded[row]
                counts[owner] += 1
    if sums is None:
        raise RuntimeError("No prompts were encoded")
    features = F.normalize(sums / counts[:, None].clamp_min(1), dim=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classes": classes,
            "features": features,
            "model": str(args.model),
            "prompt_mode": args.prompt_mode,
            "prompts": len(prompts),
        },
        args.out,
    )
    print(json.dumps({"out": str(args.out), "classes": len(classes), "prompts": len(prompts), "dim": int(features.shape[1])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
