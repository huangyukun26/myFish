from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


SYSTEM_PROMPT = (
    "You extract visible fish morphology from official species descriptions. "
    "Use only the provided description. Ignore genetics, counts, geography, habitat, depth, behavior, and taxonomy unless visibly diagnostic. "
    "If a visible trait is not explicitly described, return an empty string for that field. "
    "Never use field names as placeholder content. Never infer sex or age variation unless the description explicitly states male, female, juvenile, or breeding coloration. "
    "Return exactly one compact JSON object. Do not explain."
)


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def build_user_prompt(class_name: str, description: str) -> str:
    return (
        f"Species: {class_name}\n"
        f"Official description:\n{description}\n\n"
        "Return JSON only, beginning with { and ending with }.\n"
        "Keys: class_name, body_shape, color_pattern, fins_tail, head_mouth_eye, diagnostic_marks, sex_age_variation, confidence.\n"
        "For each trait key, write a short visible morphology phrase if the official description explicitly supports it; otherwise use an empty string.\n"
        "Use confidence high, medium, or low. Use class_name exactly as given.\n"
        "Rules:\n"
        "- Do not add knowledge not present in the description.\n"
        "- Do not copy meristic counts such as fin rays, scale counts, gill rakers, vertebrae, or pored lateral-line counts.\n"
        "- Do not write generic placeholders such as 'visible head traits' or 'other diagnostic marks'.\n"
        "- Do not mention male/female/juvenile unless the description explicitly describes visible variation."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-classes", type=int, default=0)
    args = parser.parse_args()

    classes = load_classes(args.classes_json)
    if args.max_classes:
        classes = classes[: args.max_classes]
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with args.out.open("w", encoding="utf-8") as fp:
        for class_name in classes:
            desc = " ".join((descriptions.get(class_name, "") or "").split())
            if not desc:
                missing += 1
            item = {
                "custom_id": class_name,
                "class_name": class_name,
                "system": SYSTEM_PROMPT,
                "user": build_user_prompt(class_name, desc),
            }
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "classes": len(classes),
                "missing_descriptions": missing,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
