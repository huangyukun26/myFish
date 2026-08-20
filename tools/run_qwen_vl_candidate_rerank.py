from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a visual morphology judge for fish images. "
    "Use only the visible fish image and the provided official candidate descriptions. "
    "Do not use outside biological knowledge. Choose only from the listed candidates. "
    "Return exactly one compact JSON object and no explanation."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if image_id:
                done.add(str(image_id))
    return done


def compact_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."


def row_margin(row: dict[str, Any]) -> float:
    scores = [float(v) for v in row.get("scores", [])]
    return scores[0] - scores[1] if len(scores) >= 2 else 0.0


def true_rank(row: dict[str, Any]) -> int:
    try:
        return int(row.get("true_rank", 0))
    except Exception:
        return 0


def select_rows(rows: list[dict[str, Any]], mode: str, max_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        rank = true_rank(row)
        margin = row_margin(row)
        if mode == "all":
            keep = True
        elif mode == "hard_topk":
            keep = rank > 1 and rank <= len(row.get("predictions", []))
        elif mode == "base_correct":
            keep = rank == 1
        elif mode == "low_margin":
            keep = margin <= 0.02
        elif mode == "hard_or_low_margin":
            keep = (rank > 1 and rank <= len(row.get("predictions", []))) or margin <= 0.02
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        if keep:
            item = dict(row)
            item["margin"] = margin
            selected.append(item)
        if max_samples and len(selected) >= max_samples:
            break
    return selected


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


def build_user_prompt(row: dict[str, Any], descriptions: dict[str, str], candidate_count: int, max_desc_chars: int) -> str:
    preds = list(row.get("predictions", []))[:candidate_count]
    lines = [
        "Select the candidate species whose official description best matches the visible main fish.",
        "Prioritize visible body shape, color, pattern, fins, tail, head, mouth, eye, and image domain.",
        "Ignore non-visible geography, genetics, counts, and distribution unless the image visibly supports it.",
        "If evidence is ambiguous, keep the visually closest candidate and set confidence to low.",
        "Return JSON only: {\"choice\":\"A\", \"confidence\":\"low|medium|high\", \"visual_evidence\":\"short visible evidence\"}.",
        "",
        "Candidates:",
    ]
    for idx, pred in enumerate(preds):
        letter = chr(ord("A") + idx)
        desc = compact_text(descriptions.get(pred, ""), max_desc_chars)
        lines.append(f"{letter}. {pred}: {desc}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="work/aria2_qwen25vl3b")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf_models/vlm_cache"))
    parser.add_argument("--selection-mode", default="hard_topk")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=-1)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--max-desc-chars", type=int, default=420)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-pixels", type=int, default=640 * 640)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = select_rows(read_jsonl(args.topk_jsonl), args.selection_mode, 0)
    if args.shuffle_seed >= 0:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
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
        for row in tqdm(rows, desc="qwen_candidate_rerank"):
            image_id = str(row["image_id"])
            if image_id in done:
                continue
            preds = list(row.get("predictions", []))[: args.candidate_count]
            image_path = args.image_root / image_id
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {
                            "type": "text",
                            "text": build_user_prompt(row, descriptions, len(preds), args.max_desc_chars),
                        },
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
                obj = extract_json_object(response)
                choice_idx = normalize_choice(obj.get("choice"), len(preds))
                prediction = preds[choice_idx] if choice_idx is not None else preds[0]
                out = {
                    "image_id": image_id,
                    "label": row.get("label", ""),
                    "base_prediction": preds[0] if preds else "",
                    "prediction": prediction,
                    "choice_index": choice_idx,
                    "confidence": normalize_confidence(obj.get("confidence")),
                    "visual_evidence": " ".join(str(obj.get("visual_evidence", "")).split())[:500],
                    "true_rank": row.get("true_rank", ""),
                    "margin": row.get("margin", row_margin(row)),
                    "candidates": preds,
                    "raw_response": response,
                }
            except Exception as exc:
                errors += 1
                out = {
                    "image_id": image_id,
                    "label": row.get("label", ""),
                    "base_prediction": preds[0] if preds else "",
                    "prediction": preds[0] if preds else "",
                    "error": str(exc),
                    "true_rank": row.get("true_rank", ""),
                    "margin": row.get("margin", row_margin(row)),
                    "candidates": preds,
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
                "candidate_count": args.candidate_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
