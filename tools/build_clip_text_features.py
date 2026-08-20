from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import re
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from fishnet.env import environment_report


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


VISUAL_KEYWORDS = {
    "body",
    "shape",
    "deep",
    "elongated",
    "fusiform",
    "compressed",
    "oval",
    "short",
    "color",
    "colour",
    "silvery",
    "blue",
    "green",
    "red",
    "orange",
    "yellow",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "stripe",
    "stripes",
    "spot",
    "spots",
    "blotch",
    "blotches",
    "bar",
    "bars",
    "band",
    "bands",
    "saddle",
    "pattern",
    "fin",
    "fins",
    "dorsal",
    "caudal",
    "pectoral",
    "pelvic",
    "anal",
    "tail",
    "forked",
    "truncate",
    "rounded",
    "mouth",
    "eye",
    "eyes",
    "forehead",
    "snout",
    "head",
    "jaw",
    "scales",
    "scale",
    "margin",
    "margins",
}

NON_VISUAL_HINTS = {
    "mtdna",
    "dna",
    "cleavage",
    "phenotype",
    "phenotypes",
    "gill raker",
    "gill rakers",
    "rays",
    "vertebrae",
    "lateral-line",
    "pored",
    "habitat",
    "distribution",
    "occurs",
    "found",
    "endemic",
    "depth",
    "meters",
    "metres",
}


def visual_trait_text(text: str, max_words: int) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: List[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(hint in lower for hint in NON_VISUAL_HINTS):
            continue
        if any(keyword in lower for keyword in VISUAL_KEYWORDS):
            selected.append(sentence.strip())
        if len(selected) >= 3:
            break
    if not selected:
        selected = sentences[:1]
    return truncate_words(" ".join(selected), max_words)


def make_prompts(class_name: str, description: str, include_description: bool, prompt_mode: str, desc_words: int) -> List[str]:
    desc = " ".join((description or "").split())
    sent = first_sentence(desc)
    short_desc = truncate_words(desc, desc_words)
    visual_desc = visual_trait_text(desc, desc_words)
    genus = class_name.split()[0] if class_name.split() else class_name
    if prompt_mode == "name":
        prompts = [class_name]
    elif prompt_mode == "taxon":
        prompts = [
            class_name,
            f"a photo of {class_name}",
            f"a photo of the species {class_name}",
        ]
    elif prompt_mode == "fish":
        prompts = [
            f"a photo of {class_name}, a fish species.",
            f"a close-up photo of the fish species {class_name}.",
            f"an underwater image of {class_name}.",
            f"{class_name}.",
        ]
    elif prompt_mode == "genus":
        prompts = [
            class_name,
            f"a fish in the genus {genus}.",
            f"a photo of a {genus} fish.",
        ]
    elif prompt_mode == "desc_sentence":
        prompts = [class_name]
        if sent:
            prompts.extend([f"{class_name}. {sent}", f"a fish species with these visual traits: {sent}"])
    elif prompt_mode == "desc_short":
        prompts = [class_name]
        if short_desc:
            prompts.extend([f"{class_name}, a fish species. {short_desc}", f"a photo of a fish with these traits: {short_desc}"])
    elif prompt_mode == "visual_traits":
        prompts = [class_name]
        if visual_desc:
            prompts.extend(
                [
                    f"{class_name}, a fish species with visible traits: {visual_desc}",
                    f"a photo of a fish with these visual identification traits: {visual_desc}",
                ]
            )
    elif prompt_mode == "all":
        prompts = []
        for inner in ["name", "taxon", "fish", "genus", "desc_sentence", "desc_short", "visual_traits"]:
            prompts.extend(make_prompts(class_name, desc, False, inner, desc_words))
        prompts = list(dict.fromkeys(prompts))
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    if include_description and desc and prompt_mode not in {"desc_sentence", "desc_short", "all"}:
        prompts.append(f"{class_name}, a fish species. {short_desc}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out", type=Path, default=Path("work/clip_text_features/all_classes_vitb32_openai.pt"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-description", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        default="fish",
        choices=["name", "taxon", "fish", "genus", "desc_sentence", "desc_short", "visual_traits", "all"],
    )
    parser.add_argument("--desc-words", type=int, default=45)
    parser.add_argument("--clip-precision", default="fp32")
    args = parser.parse_args()

    import open_clip

    classes = load_classes(args.classes_json)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8")) if args.descriptions.exists() else {}
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

    sum_features = None
    counts = torch.zeros(len(classes), dtype=torch.float32)
    prompt_buffer: List[str] = []
    index_buffer: List[int] = []
    with torch.inference_mode():
        for class_index, class_name in enumerate(tqdm(classes, desc="prepare_prompts")):
            prompts = make_prompts(class_name, descriptions.get(class_name, ""), args.include_description, args.prompt_mode, args.desc_words)
            prompt_buffer.extend(prompts)
            index_buffer.extend([class_index] * len(prompts))

        for start in tqdm(range(0, len(prompt_buffer), args.batch_size), desc="encode_text"):
            prompts = prompt_buffer[start : start + args.batch_size]
            class_indices = index_buffer[start : start + args.batch_size]
            tokens = tokenizer(prompts).to(device)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            encoded = encoded.cpu()
            if sum_features is None:
                sum_features = torch.zeros((len(classes), encoded.shape[1]), dtype=torch.float32)
            for row_index, class_index in enumerate(class_indices):
                sum_features[class_index] += encoded[row_index]
                counts[class_index] += 1

    if sum_features is None:
        raise RuntimeError("No text features encoded")
    features = sum_features / counts[:, None].clamp_min(1)
    features = features / features.norm(dim=-1, keepdim=True)

    out = {
        "classes": classes,
        "features": features,
        "model": args.model,
        "pretrained": args.pretrained,
        "clip_precision": args.clip_precision,
        "include_description": args.include_description,
        "prompt_mode": args.prompt_mode,
        "desc_words": args.desc_words,
        "prompts": len(prompt_buffer),
        "env": environment_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"out": str(args.out), "classes": len(classes), "dim": int(out["features"].shape[1])}, indent=2))


if __name__ == "__main__":
    main()
