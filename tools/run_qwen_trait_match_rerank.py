from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You match visible fish-image traits to official fish species descriptions. "
    "Use only the provided image traits and official candidate descriptions. "
    "Do not use outside biological knowledge. Choose only from the listed candidates. "
    "Return exactly one compact JSON object and no explanation."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("image_id"):
                out.add(str(row["image_id"]))
    return out


def compact_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


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
        raise ValueError(f"no JSON object found: {text[:240]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("decoded JSON is not an object")
    return obj


def normalize_choice(value: Any, topk: int) -> int | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text[0].isalpha():
        idx = ord(text[0]) - ord("A")
        return idx if 0 <= idx < topk else None
    match = re.search(r"\d+", text)
    if not match:
        return None
    idx = int(match.group(0)) - 1
    return idx if 0 <= idx < topk else None


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "high" in text:
        return "high"
    if "medium" in text or "moderate" in text:
        return "medium"
    if "low" in text:
        return "low"
    return text[:32] if text else "unknown"


def trait_text(row: dict[str, Any]) -> str:
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
    parts = []
    for key in keys:
        value = " ".join(str(row.get(key, "")).split())
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def build_prompt(
    *,
    topk_row: dict[str, Any],
    trait_row: dict[str, Any],
    descriptions: dict[str, str],
    candidate_count: int,
    max_desc_chars: int,
) -> str:
    preds = list(topk_row.get("predictions", []))[:candidate_count]
    lines = [
        "Image traits extracted by a vision model:",
        trait_text(trait_row),
        "",
        "Select the candidate whose official description best matches these visible traits.",
        "Prioritize morphology that can be visible in a photo: body shape, colors, stripes/bars/spots/blotches, fins, tail, head, mouth, eyes.",
        "Ignore geography, distribution, genetics, and habitat unless directly visible in the image traits.",
        "If the traits are too generic or conflicting, keep the closest candidate but set confidence to low.",
        "Return JSON only: {\"choice\":\"A\", \"confidence\":\"low|medium|high\", \"trait_match\":\"short reason\"}.",
        "",
        "Candidates:",
    ]
    for idx, pred in enumerate(preds):
        letter = chr(ord("A") + idx)
        lines.append(f"{letter}. {pred}: {compact_text(descriptions.get(pred, ''), max_desc_chars)}")
    return "\n".join(lines)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row.get("label")]
    if not known:
        return {}
    wins = losses = changed = 0
    base_ok = final_ok = 0
    for row in known:
        label = row["label"]
        base = row.get("base_prediction", "")
        pred = row.get("prediction", "")
        base_hit = base == label
        final_hit = pred == label
        base_ok += int(base_hit)
        final_ok += int(final_hit)
        changed += int(base != pred)
        wins += int((not base_hit) and final_hit)
        losses += int(base_hit and (not final_hit))
    return {
        "known": len(known),
        "base_top1": base_ok / len(known),
        "new_top1": final_ok / len(known),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--image-traits-jsonl", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="work/aria2_qwen25vl3b")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf_models/vlm_cache"))
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--max-desc-chars", type=int, default=360)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    topk_by_id = {row["image_id"]: row for row in read_jsonl(args.topk_jsonl)}
    trait_rows = [row for row in read_jsonl(args.image_traits_jsonl) if "error" not in row and row.get("image_id") in topk_by_id]
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    done = read_done(args.out_jsonl) if args.resume else set()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=str(args.cache_dir), trust_remote_code=True)
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

    written_rows = []
    errors = 0
    with args.out_jsonl.open("a", encoding="utf-8") as out_fp:
        for trait_row in tqdm(trait_rows, desc="qwen_trait_match"):
            image_id = str(trait_row["image_id"])
            if image_id in done:
                continue
            topk_row = topk_by_id[image_id]
            preds = list(topk_row.get("predictions", []))[: args.candidate_count]
            try:
                prompt = build_prompt(
                    topk_row=topk_row,
                    trait_row=trait_row,
                    descriptions=descriptions,
                    candidate_count=len(preds),
                    max_desc_chars=args.max_desc_chars,
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], padding=True, return_tensors="pt")
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                response_ids = generated[0, inputs["input_ids"].shape[1] :]
                response = processor.decode(response_ids, skip_special_tokens=True)
                obj = extract_json_object(response)
                choice_idx = normalize_choice(obj.get("choice"), len(preds))
                prediction = preds[choice_idx] if choice_idx is not None else preds[0]
                out = {
                    "image_id": image_id,
                    "label": topk_row.get("label", ""),
                    "base_prediction": preds[0] if preds else "",
                    "prediction": prediction,
                    "choice_index": choice_idx,
                    "confidence": normalize_confidence(obj.get("confidence")),
                    "trait_match": " ".join(str(obj.get("trait_match", "")).split())[:500],
                    "candidates": preds,
                    "raw_response": response,
                }
            except Exception as exc:
                errors += 1
                out = {
                    "image_id": image_id,
                    "label": topk_row.get("label", ""),
                    "base_prediction": preds[0] if preds else "",
                    "prediction": preds[0] if preds else "",
                    "error": str(exc),
                    "candidates": preds,
                }
            out_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_fp.flush()
            written_rows.append(out)
    summary = {
        "topk_jsonl": str(args.topk_jsonl),
        "image_traits_jsonl": str(args.image_traits_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "selected": len(trait_rows),
        "written": len(written_rows),
        "skipped_done": len(done),
        "errors": errors,
        "candidate_count": args.candidate_count,
        **metrics(written_rows),
    }
    (args.out_jsonl.parent / (args.out_jsonl.stem + ".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
