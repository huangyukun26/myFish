from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ALLOWED_CATEGORIES = {
    "shape",
    "structure",
    "coloration",
    "markings",
    "diagnostic",
    "absence",
    "meristic_or_measurement",
}

CATEGORY_WEIGHT = {
    "markings": 5.0,
    "coloration": 5.0,
    "shape": 4.0,
    "structure": 4.0,
    "diagnostic": 3.0,
    "absence": 1.0,
    "meristic_or_measurement": 0.0,
}

GENERIC_PHRASES = (
    "normal",
    "unnoted",
    "unspecified",
    "assist in identifying",
    "aiding in differentiation",
    "distinctive features absent",
    "not described",
    "not provided",
)


def normalize_match_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", (value or "").lower()))


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if not isinstance(payload, dict) else list(payload.keys())


def trait_score(row: dict[str, Any]) -> float:
    value = str(row.get("normalized_value") or "")
    score = CATEGORY_WEIGHT.get(str(row.get("trait_category")), 0.0)
    score += 3.0 if row.get("diagnostic_strength") == "diagnostic" else 0.0
    score += 2.0 if row.get("visibility_for_image") == "high" else 0.0
    score += min(2.0, len(value.split()) / 5.0)
    if any(phrase in value.lower() for phrase in GENERIC_PHRASES):
        score -= 5.0
    return score


def keep_trait(row: dict[str, Any]) -> bool:
    if row.get("visibility_for_image") not in {"high", "medium"}:
        return False
    if row.get("trait_category") not in ALLOWED_CATEGORIES:
        return False
    value = str(row.get("normalized_value") or "").strip()
    if len(value.split()) < 2:
        return False
    if any(phrase in value.lower() for phrase in GENERIC_PHRASES):
        return False
    return True


def build_document(class_name: str, rows: list[dict[str, Any]], max_traits: int) -> tuple[str, list[dict[str, Any]]]:
    selected = []
    seen = set()
    for row in sorted(rows, key=trait_score, reverse=True):
        if not keep_trait(row):
            continue
        value = " ".join(str(row["normalized_value"]).split())
        key = (str(row.get("body_region") or ""), value.lower())
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= max_traits:
            break
    genus = class_name.split(maxsplit=1)[0]
    pieces = [f"species {class_name}", f"genus {genus}"]
    for row in selected:
        region = str(row.get("body_region") or "whole_body").replace("_", " ")
        category = str(row.get("trait_category") or "visual").replace("_", " ")
        value = " ".join(str(row["normalized_value"]).split())
        pieces.append(f"{region} {category}: {value}")
    return ". ".join(pieces), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traits-jsonl", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-traits", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=75000)
    parser.add_argument("--svd-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    description_items = list(descriptions.items())
    normalized_descriptions = [normalize_match_text(text) for _, text in description_items]
    target_classes = load_classes(args.classes)
    target_set = set(target_classes)
    rows_by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pointer = 0
    mapped_rows = 0
    with args.traits_jsonl.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, 1):
            row = json.loads(line)
            evidence = normalize_match_text(str(row.get("raw_evidence") or ""))
            if not evidence:
                raise RuntimeError(f"Empty raw_evidence at line {line_number}")
            if evidence not in normalized_descriptions[pointer]:
                pointer += 1
                while pointer < len(description_items) and evidence not in normalized_descriptions[pointer]:
                    pointer += 1
                if pointer >= len(description_items):
                    raise RuntimeError(f"Could not map line {line_number}: {row.get('raw_evidence')}")
            species = description_items[pointer][0]
            mapped_rows += 1
            if species in target_set:
                row["species"] = species
                rows_by_species[species].append(row)
    if pointer != len(description_items) - 1:
        raise RuntimeError(f"Trait stream ended at description {pointer}, expected {len(description_items) - 1}")
    missing = [name for name in target_classes if name not in rows_by_species]
    if missing:
        raise RuntimeError(f"{len(missing)} target classes have no mapped traits; first={missing[:5]}")

    documents = []
    selected_payload: dict[str, Any] = {}
    selected_counts = []
    category_counts: Counter[str] = Counter()
    for class_name in target_classes:
        document, selected = build_document(class_name, rows_by_species[class_name], args.max_traits)
        documents.append(document)
        selected_counts.append(len(selected))
        category_counts.update(str(row.get("trait_category")) for row in selected)
        selected_payload[class_name] = {
            "document": document,
            "traits": [
                {
                    "body_region": row.get("body_region"),
                    "body_part": row.get("body_part"),
                    "trait_category": row.get("trait_category"),
                    "normalized_value": row.get("normalized_value"),
                    "visibility_for_image": row.get("visibility_for_image"),
                    "diagnostic_strength": row.get("diagnostic_strength"),
                    "life_stage": row.get("life_stage"),
                    "sex": row.get("sex"),
                    "condition": row.get("condition"),
                }
                for row in selected
            ],
        }

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(documents)
    effective_dim = min(args.svd_dim, sparse.shape[0] - 1, sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components=effective_dim, n_iter=7, random_state=args.seed)
    features = torch.from_numpy(svd.fit_transform(sparse)).float()
    features = F.normalize(features, dim=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "structured_trait_tfidf_svd.pt"
    torch.save(
        {
            "classes": target_classes,
            "features": features,
            "source_traits": str(args.traits_jsonl),
            "source_descriptions": str(args.descriptions),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "vocabulary_size": len(vectorizer.vocabulary_),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        },
        feature_path,
    )
    (args.out_dir / "structured_trait_documents.json").write_text(
        json.dumps(selected_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "traits_jsonl": str(args.traits_jsonl),
        "descriptions": str(args.descriptions),
        "mapped_rows": mapped_rows,
        "descriptions_mapped": len(description_items),
        "target_classes": len(target_classes),
        "target_trait_rows": sum(len(rows_by_species[name]) for name in target_classes),
        "selected_traits": sum(selected_counts),
        "selected_traits_min": min(selected_counts),
        "selected_traits_median": float(np.median(selected_counts)),
        "selected_traits_max": max(selected_counts),
        "selected_category_counts": category_counts,
        "tfidf_shape": list(sparse.shape),
        "svd_dim": effective_dim,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "feature_path": str(feature_path),
        "documents_path": str(args.out_dir / "structured_trait_documents.json"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
