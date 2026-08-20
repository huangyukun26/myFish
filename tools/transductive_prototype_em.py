from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from transductive_active_sinkhorn import load_classes, load_text_features, normalize


def topk_metrics(pred: torch.Tensor, labels: list[str], candidates: list[str]) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    known = correct = 0
    for i, label in enumerate(labels):
        if not label:
            continue
        true = class_to_idx.get(label)
        if true is None:
            continue
        known += 1
        correct += int(int(pred[i]) == true)
    return {} if not known else {"known": known, "top1": correct / known}


def masked_softmax(logits: torch.Tensor, topm: int, temp: float) -> torch.Tensor:
    if topm > 0 and topm < logits.shape[1]:
        vals, idx = logits.topk(topm, dim=1)
        masked = torch.full_like(logits, -10000.0)
        masked.scatter_(1, idx, vals)
        logits = masked
    return torch.softmax(logits / temp, dim=1)


def run_em(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    topm: int,
    temp: float,
    beta: float,
    iters: int,
    blend_text: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    proto = text_features.clone()
    text = text_features
    x = image_features
    for _ in range(iters):
        logits = x @ proto.T
        probs = masked_softmax(logits, topm, temp)
        mass = probs.sum(dim=0).clamp_min(1e-6)
        visual = probs.T @ x
        visual = visual / mass[:, None]
        proto = F.normalize(beta * text + visual, dim=1)
    final_logits = x @ proto.T
    if blend_text:
        final_logits = final_logits + blend_text * (x @ text.T)
    pred = final_logits.argmax(dim=1).cpu()
    return pred, final_logits


def parse_grid(value: str, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def write_predictions(path: Path, image_ids: list[str], candidates: list[str], pred: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, idx in zip(image_ids, pred.tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topm-grid", default="20,50,100,200")
    parser.add_argument("--temp-grid", default="0.01,0.02,0.03,0.05")
    parser.add_argument("--beta-grid", default="1,3,10,30")
    parser.add_argument("--iters-grid", default="1,2,4")
    parser.add_argument("--blend-text-grid", default="0,0.5,1,2")
    parser.add_argument("--score-batch-size", type=int, default=0)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))
    image_features = normalize(image_payload["features"].float())

    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_features = load_text_features(args.text_features, candidates)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_row = None
    best_pred = None
    for topm in parse_grid(args.topm_grid, int):
        for temp in parse_grid(args.temp_grid, float):
            for beta in parse_grid(args.beta_grid, float):
                for iters in parse_grid(args.iters_grid, int):
                    for blend_text in parse_grid(args.blend_text_grid, float):
                        pred, _ = run_em(
                            image_features,
                            text_features,
                            topm=topm,
                            temp=temp,
                            beta=beta,
                            iters=iters,
                            blend_text=blend_text,
                        )
                        row = {
                            "topm": topm,
                            "temp": temp,
                            "beta": beta,
                            "iters": iters,
                            "blend_text": blend_text,
                            **topk_metrics(pred, labels, candidates),
                        }
                        rows.append(row)
                        if row.get("top1") is not None and (
                            best_row is None or row["top1"] > best_row.get("top1", -1)
                        ):
                            best_row = row
                            best_pred = pred

    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best_pred is not None:
        write_predictions(args.out_dir / "predictions.csv", image_ids, candidates, best_pred)
    summary = {
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "candidate_classes": str(args.candidate_classes),
        "best": best_row,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
