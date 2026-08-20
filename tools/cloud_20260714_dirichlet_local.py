from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def load_classes(path: Path) -> list[str]:
    x = json.loads(path.read_text(encoding="utf-8"))
    return list(x) if not isinstance(x, dict) else list(x.keys())


def align_text(path: Path, classes: list[str]) -> torch.Tensor:
    x = torch.load(path, map_location="cpu", weights_only=False)
    pos = {name: i for i, name in enumerate(x["classes"])}
    return normalize(x["features"][[pos[name] for name in classes]])


def score(image: torch.Tensor, text: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts = []
    text = text.to(device)
    with torch.inference_mode():
        for start in range(0, len(image), 256):
            parts.append((image[start:start + 256].to(device) @ text.T).cpu())
    return torch.cat(parts)


def update_alpha(alpha: torch.Tensor, y: torch.Tensor, steps: int) -> torch.Tensor:
    zero = torch.polygamma(1, torch.ones(1, device=alpha.device))
    for _ in range(steps):
        digam = torch.polygamma(0, alpha + 1)
        curv = torch.where(
            alpha > 1e-11,
            (2 * (digam * alpha - torch.lgamma(alpha + 1)) / alpha.square()).abs(),
            zero,
        ).clamp_min(1e-8)
        b = digam - torch.polygamma(0, alpha.sum(-1))[:, None] - curv * alpha - y
        alpha = (-b + torch.sqrt(b.square() + 4 * curv)) / (2 * curv)
    return alpha


def em_batch(prob: torch.Tensor, scale: float, outer: int, mm: int) -> torch.Tensor:
    eps = 1e-12
    n, k = prob.shape
    u = prob.clone()
    alpha = torch.ones(k, k, device=prob.device)
    logq = torch.log(prob.clamp_min(eps))
    for _ in range(outer):
        cluster = u.sum(0).clamp_min(eps)
        y = (u.T @ logq) / cluster[:, None]
        alpha = update_alpha(alpha, y, mm)
        log_norm = torch.lgamma(alpha.sum(-1)) - torch.lgamma(alpha).sum(-1)
        logits = log_norm[None, :] + logq @ (alpha - 1).T
        v = torch.log(cluster / n + eps) + 1
        logits += scale * (k / 5.0) * v[None, :]
        u = logits.softmax(-1)
    return u


def run(scores: torch.Tensor, *, batch_size: int, topm: int, temperature: float,
        scale: float, outer: int, mm: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base_top = scores.topk(20, dim=1).indices
    output = base_top[:, 0].clone()
    confidence = torch.zeros(len(scores))
    for start in range(0, len(scores), batch_size):
        end = min(start + batch_size, len(scores))
        pool = base_top[start:end, :topm].unique(sorted=True)
        prob = (scores[start:end, pool].to(device) / temperature).softmax(-1)
        u = em_batch(prob, scale, outer, mm)
        values, pred = u.topk(2, dim=1)
        output[start:end] = pool[pred[:, 0].cpu()]
        confidence[start:end] = (values[:, 0] - values[:, 1]).cpu()
    return output, confidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-features", type=Path, required=True)
    ap.add_argument("--text-features", type=Path, required=True)
    ap.add_argument("--candidate-classes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=75)
    ap.add_argument("--topm", type=int, default=10)
    ap.add_argument("--temperatures", default="0.02,0.033333,0.05")
    ap.add_argument("--scales", default="0.25,0.5,1.0")
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--mm", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image = normalize(image_payload["features"])
    ids = list(image_payload["image_ids"]); labels = list(image_payload.get("labels", []))
    classes = load_classes(args.candidate_classes)
    text = align_text(args.text_features, classes)
    scores = score(image, text, device)
    base = scores.argmax(1)
    truth = None
    class_set = set(classes)
    if labels and all(label in class_set for label in labels):
        pos = {name: i for i, name in enumerate(classes)}
        truth = torch.tensor([pos[label] for label in labels])
    rows, predictions = [], {}
    for temp in [float(x) for x in args.temperatures.split(",")]:
        for scale in [float(x) for x in args.scales.split(",")]:
            pred, conf = run(scores, batch_size=args.batch_size, topm=args.topm,
                             temperature=temp, scale=scale, outer=args.outer, mm=args.mm, device=device)
            key = f"t{temp:g}_s{scale:g}"
            row = {"config": key, "temperature": temp, "scale": scale,
                   "changed": int(pred.ne(base).sum())}
            if truth is not None:
                bc, pc = base.eq(truth), pred.eq(truth)
                row.update(top1=float(pc.float().mean()), base_top1=float(bc.float().mean()),
                           wins=int((pc & ~bc).sum()), losses=int((~pc & bc).sum()),
                           net=int(pc.sum() - bc.sum()))
            rows.append(row); predictions[key] = {"pred": pred, "confidence": conf}
    rows.sort(key=lambda x: (x.get("net", 0), -x["changed"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": ids, "labels": labels, "classes": classes, "base": base,
                "rows": rows, "predictions": predictions}, args.out)
    with args.out.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "rows": len(ids), "candidates": len(classes),
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
