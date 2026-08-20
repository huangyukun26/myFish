from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


CORE_CATEGORIES = {
    "shape",
    "structure",
    "coloration",
    "markings",
    "diagnostic",
}


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if not isinstance(payload, dict) else list(payload.keys())


def normalized_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", value.lower()))


def clean_traits(payload: dict[str, Any], max_traits: int) -> list[str]:
    pieces: list[str] = []
    seen_values: set[str] = set()
    for row in payload.get("traits", []):
        category = str(row.get("trait_category") or "")
        if category not in CORE_CATEGORIES:
            continue
        if row.get("visibility_for_image") not in {"high", "medium"}:
            continue
        value = " ".join(str(row.get("normalized_value") or "").split())
        key = normalized_key(value)
        if len(key.split()) < 2 or key in seen_values:
            continue
        seen_values.add(key)
        region = str(row.get("body_region") or "whole_body").replace("_", " ")
        pieces.append(f"{region} {category}: {value}")
        if len(pieces) >= max_traits:
            break
    return pieces


def build_documents(
    classes: list[str],
    source: dict[str, Any],
    variant: str,
    max_traits: int,
) -> tuple[list[str], list[int]]:
    documents: list[str] = []
    trait_counts: list[int] = []
    for class_name in classes:
        genus = class_name.split(maxsplit=1)[0]
        traits = clean_traits(source[class_name], max_traits)
        if variant == "names_only":
            pieces = [f"species {class_name}", f"genus {genus}"]
        elif variant == "traits_only_clean":
            pieces = traits or ["visual traits unavailable"]
        elif variant == "names_traits_clean":
            pieces = [f"species {class_name}", f"genus {genus}", *traits]
        else:
            raise ValueError(f"Unknown variant: {variant}")
        documents.append(". ".join(pieces))
        trait_counts.append(len(traits))
    return documents, trait_counts


def encode_documents(
    documents: list[str],
    max_features: int,
    svd_dim: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(documents)
    effective_dim = min(svd_dim, sparse.shape[0] - 1, sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components=effective_dim, n_iter=7, random_state=seed)
    features = F.normalize(torch.from_numpy(svd.fit_transform(sparse)).float(), dim=1)
    metrics = {
        "tfidf_shape": list(sparse.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "svd_dim": effective_dim,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    return features, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="names_only,traits_only_clean,names_traits_clean",
    )
    parser.add_argument("--max-traits", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=75000)
    parser.add_argument("--svd-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    classes = load_classes(args.classes)
    source = json.loads(args.documents.read_text(encoding="utf-8"))
    missing = [name for name in classes if name not in source]
    if missing:
        raise RuntimeError(f"{len(missing)} classes missing from documents; first={missing[:5]}")

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "source_documents": str(args.documents),
        "classes": len(classes),
        "max_traits": args.max_traits,
        "variants": {},
    }
    for variant in variants:
        documents, trait_counts = build_documents(classes, source, variant, args.max_traits)
        features, metrics = encode_documents(
            documents,
            args.max_features,
            args.svd_dim,
            args.seed,
        )
        out_path = args.out_dir / f"{variant}_tfidf_svd.pt"
        torch.save(
            {
                "classes": classes,
                "features": features,
                "variant": variant,
                "source_documents": str(args.documents),
                "max_traits": args.max_traits,
                **metrics,
            },
            out_path,
        )
        summary["variants"][variant] = {
            **metrics,
            "trait_count_min": min(trait_counts),
            "trait_count_median": float(np.median(trait_counts)),
            "trait_count_max": max(trait_counts),
            "out": str(out_path),
        }
        print(json.dumps({variant: summary["variants"][variant]}, indent=2), flush=True)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
