from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REQUIRED_KEYS = [
    "class_name",
    "body_shape",
    "color_pattern",
    "fins_tail",
    "head_mouth_eye",
    "diagnostic_marks",
    "sex_age_variation",
    "confidence",
]

PLACEHOLDER_PATTERNS = [
    "visible body shape",
    "visible colors",
    "visible fin",
    "visible head",
    "other visible diagnostic",
    "sex_age_variation",
    "male/female/juvenile variation",
    "max 18 words",
    "max 24 words",
    "max 28 words",
    "not mentioned",
    "no explicit description",
]


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
            class_name = obj.get("class_name") or obj.get("custom_id")
            if class_name:
                done.add(class_name)
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
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("decoded JSON is not an object")
    return obj


def normalize_trait(obj: dict[str, Any], fallback_class_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in REQUIRED_KEYS:
        value = obj.get(key, "")
        if value is None:
            value = ""
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(part) for part in value)
        cleaned = " ".join(str(value).split())
        if key not in {"class_name", "confidence"}:
            lower = cleaned.lower()
            if any(pattern in lower for pattern in PLACEHOLDER_PATTERNS):
                cleaned = ""
        out[key] = cleaned
    out["class_name"] = fallback_class_name
    if out["confidence"].lower() not in {"high", "medium", "low"}:
        out["confidence"] = "low"
    return out


def truncate_user_prompt_preserving_rules(user: str, max_chars: int) -> str:
    if len(user) <= max_chars:
        return user
    marker = "\n\nReturn one JSON object"
    if marker not in user:
        return user[:max_chars]
    head, tail = user.split(marker, 1)
    desc_marker = "Official description:\n"
    if desc_marker not in head:
        keep_head = max(0, max_chars - len(marker) - len(tail) - 80)
        return head[:keep_head] + marker + tail
    prefix, desc = head.split(desc_marker, 1)
    budget = max_chars - len(prefix) - len(desc_marker) - len(marker) - len(tail)
    budget = max(300, budget)
    truncated_desc = desc[:budget].rsplit(" ", 1)[0]
    return prefix + desc_marker + truncated_desc + marker + tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf_models/llm_cache"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=2600)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.dtype == "float16" or (args.dtype == "auto" and device == "cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(args.cache_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    done = read_done(args.out_jsonl) if args.resume else set()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0
    with args.tasks_jsonl.open("r", encoding="utf-8") as in_fp, args.out_jsonl.open("a", encoding="utf-8") as out_fp:
        tasks = [json.loads(line) for line in in_fp if line.strip()]
        if args.max_samples:
            tasks = tasks[: args.max_samples]
        for task in tqdm(tasks, desc="extract_traits"):
            class_name = task["class_name"]
            if class_name in done:
                continue
            system = task["system"]
            user = truncate_user_prompt_preserving_rules(task["user"], args.max_input_chars)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_ids = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            try:
                obj = normalize_trait(extract_json_object(response), fallback_class_name=class_name)
            except Exception as exc:
                errors += 1
                obj = {
                    "class_name": class_name,
                    "body_shape": "",
                    "color_pattern": "",
                    "fins_tail": "",
                    "head_mouth_eye": "",
                    "diagnostic_marks": "",
                    "sex_age_variation": "",
                    "confidence": "low",
                    "error": str(exc),
                    "raw_response": response[:1000],
                }
            out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_fp.flush()
            written += 1
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "written": written,
                "skipped_done": len(done),
                "errors": errors,
                "out_jsonl": str(args.out_jsonl),
                "model": args.model,
                "device": device,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
