from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from fishnet.env import environment_report


TRAIT_KEYS = [
    "body_shape",
    "color_pattern",
    "fins_tail",
    "head_mouth_eye",
    "diagnostic_marks",
    "sex_age_variation",
]


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def load_traits(path: Path) -> Dict[str, dict]:
    traits: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            class_name = obj.get("class_name") or obj.get("custom_id")
            if not class_name:
                raise ValueError(f"missing class_name at {path}:{line_no}")
            traits[class_name] = obj
    return traits


def compact_trait_text(trait: dict, max_words: int) -> str:
    parts = []
    for key in TRAIT_KEYS:
        value = " ".join(str(trait.get(key, "") or "").split())
        if value:
            parts.append(value)
    text = "; ".join(parts)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def make_prompts(class_name: str, trait: dict, max_words: int) -> List[str]:
    trait_text = compact_trait_text(trait, max_words=max_words)
    prompts = [class_name, f"a photo of {class_name}, a fish species."]
    if trait_text:
        prompts.extend(
            [
                f"{class_name}, a fish species with visible morphology: {trait_text}",
                f"a close-up fish photo showing these visible traits: {trait_text}",
            ]
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--traits-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="local-dir:work/hf_models/bioclip-2.5-vith14")
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-trait-words", type=int, default=55)
    parser.add_argument("--clip-precision", default="fp16")
    args = parser.parse_args()

    import open_clip

    classes = load_classes(args.classes_json)
    traits = load_traits(args.traits_jsonl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = None if args.pretrained.lower() in {"none", "null", ""} else args.pretrained
    model, _preprocess_train, _preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        precision=args.clip_precision,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval()

    prompt_buffer: List[str] = []
    index_buffer: List[int] = []
    missing_traits = 0
    for class_index, class_name in enumerate(tqdm(classes, desc="prepare_trait_prompts")):
        trait = traits.get(class_name)
        if trait is None:
            missing_traits += 1
            trait = {"class_name": class_name}
        prompts = make_prompts(class_name, trait, max_words=args.max_trait_words)
        prompt_buffer.extend(prompts)
        index_buffer.extend([class_index] * len(prompts))

    sum_features = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompt_buffer), args.batch_size), desc="encode_trait_text"):
            prompts = prompt_buffer[start : start + args.batch_size]
            class_indices = index_buffer[start : start + args.batch_size]
            tokens = tokenizer(prompts).to(device)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            encoded = encoded.cpu()
            if sum_features is None:
                sum_features = torch.zeros((len(classes), encoded.shape[1]), dtype=torch.float32)
            for row_index, class_index in enumerate(class_indices):
                sum_features[class_index] += encoded[row_index]
                counts[class_index] += 1

    if sum_features is None:
        raise RuntimeError("No text features encoded")
    features = sum_features / counts[:, None].clamp_min(1)
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    payload = {
        "classes": classes,
        "features": features,
        "traits_jsonl": str(args.traits_jsonl),
        "model": args.model,
        "pretrained": args.pretrained,
        "clip_precision": args.clip_precision,
        "max_trait_words": args.max_trait_words,
        "prompts": len(prompt_buffer),
        "missing_traits": missing_traits,
        "env": environment_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "classes": len(classes),
                "dim": int(features.shape[1]),
                "prompts": len(prompt_buffer),
                "missing_traits": missing_traits,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
