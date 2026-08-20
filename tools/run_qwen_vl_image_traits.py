from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You describe only visible morphology of the main fish in an image. "
    "Do not identify the species. Do not use outside biological knowledge. "
    "Return exactly one compact JSON object and no explanation."
)


USER_PROMPT = (
    "Inspect the main fish in the image. Return JSON only with these keys: "
    "body_shape, dominant_colors, pattern, fins_tail, head_mouth_eye, fish_count, image_domain, quality, uncertainty. "
    "Use short phrases. Mention only visible evidence such as elongated/deep/compressed body, stripes/bars/bands/spots/blotches, "
    "tail shape, fin colors, mouth/head/eye traits, whether the image is underwater/aquarium/specimen/market/angler, and if the fish is small/occluded. "
    "Do not guess a species name. If unsure, say uncertainty='high'."
)


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = obj.get("image_id")
            if image_id:
                done.add(str(image_id))
    return done


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object found: {text[:200]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("decoded JSON is not an object")
    return obj


def normalize_obj(obj: dict[str, Any]) -> dict[str, str]:
    keys = [
        "body_shape",
        "dominant_colors",
        "pattern",
        "fins_tail",
        "head_mouth_eye",
        "fish_count",
        "image_domain",
        "quality",
        "uncertainty",
    ]
    out: dict[str, str] = {}
    for key in keys:
        value = obj.get(key, "")
        if value is None:
            value = ""
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(part) for part in value)
        out[key] = " ".join(str(value).split())
    return out


def read_topk(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], mode: str, max_samples: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        scores = [float(v) for v in row.get("scores", [])]
        margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        true_rank = row.get("true_rank", "")
        try:
            true_rank_int = int(true_rank)
        except Exception:
            true_rank_int = 0
        if mode == "all":
            keep = True
        elif mode == "hard_top20":
            keep = true_rank_int > 1 and true_rank_int <= len(row.get("predictions", []))
        elif mode == "low_margin":
            keep = margin <= 0.02
        elif mode == "hard_or_low_margin":
            keep = (true_rank_int > 1 and true_rank_int <= len(row.get("predictions", []))) or margin <= 0.02
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        if keep:
            item = dict(row)
            item["margin"] = margin
            selected.append(item)
        if max_samples and len(selected) >= max_samples:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf_models/vlm_cache"))
    parser.add_argument("--selection-mode", default="hard_top20")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-pixels", type=int, default=640 * 640)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = select_rows(read_topk(args.topk_jsonl), args.selection_mode, args.max_samples)
    done = read_done(args.out_jsonl) if args.resume else set()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        trust_remote_code=True,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    written = 0
    errors = 0
    with args.out_jsonl.open("a", encoding="utf-8") as out_fp:
        for row in tqdm(rows, desc="qwen_vl_traits"):
            image_id = str(row["image_id"])
            if image_id in done:
                continue
            image_path = args.image_root / image_id
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                response_ids = generated[0, inputs["input_ids"].shape[1] :]
                response = processor.decode(response_ids, skip_special_tokens=True)
                traits = normalize_obj(extract_json_object(response))
                out = {
                    "image_id": image_id,
                    "label": row.get("label", ""),
                    "base_prediction": row.get("predictions", [""])[0],
                    "true_rank": row.get("true_rank", ""),
                    "margin": row.get("margin", ""),
                    **traits,
                    "raw_response": response,
                }
            except Exception as exc:
                errors += 1
                out = {
                    "image_id": image_id,
                    "label": row.get("label", ""),
                    "base_prediction": row.get("predictions", [""])[0],
                    "true_rank": row.get("true_rank", ""),
                    "margin": row.get("margin", ""),
                    "error": str(exc),
                }
            out_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_fp.flush()
            written += 1
    print(
        json.dumps(
            {
                "topk_jsonl": str(args.topk_jsonl),
                "out_jsonl": str(args.out_jsonl),
                "selected": len(rows),
                "written": written,
                "skipped_done": len(done),
                "errors": errors,
                "model": args.model,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
