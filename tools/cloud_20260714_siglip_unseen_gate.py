from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from evaluate_hf_zeroshot_topk import encode_text


SNAP = Path("/root/.cache/huggingface/hub/models--google--siglip2-base-patch16-naflex/snapshots/b53b807d3a2d5e2b3911292f2d69e5341cdc064c")
OUT = Path("runs/cloud_20260714/siglip2_unseen_gate")


def classes(path: Path) -> list[str]:
    x = json.loads(path.read_text(encoding="utf-8"))
    return list(x) if not isinstance(x, dict) else list(x.keys())


def main() -> None:
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(SNAP)
    model = AutoModel.from_pretrained(SNAP, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    all_classes = classes(Path("work/full_manifests/all_classes.json"))
    descriptions = json.loads(Path("dataset/descriptions.json").read_text(encoding="utf-8"))
    text_branches = {}
    for name, modes in {"taxon": ["taxon"], "fish": ["fish"], "fish_taxon": ["fish", "taxon"]}.items():
        text_branches[name] = encode_text(model=model, processor=processor, classes=all_classes,
            descriptions=descriptions, modes=modes, desc_words=45, batch_size=512, device=device, amp=True)
    all_pos = {name: i for i, name in enumerate(all_classes)}
    results = {}
    for kind in ("species", "genus"):
        image_payload = torch.load(OUT / f"pseudo_{kind}.pt", map_location="cpu", weights_only=False)
        image = F.normalize(image_payload["features"].float(), dim=1).to(device)
        labels = list(image_payload["labels"])
        for seed in (42, 43, 44):
            candidate_path = Path(f"work/pseudo_unseen/{kind}_1000_seed42/candidate_classes_11598_seed{seed}.json")
            candidates = classes(candidate_path); idx = torch.tensor([all_pos[x] for x in candidates])
            pos = {name: i for i, name in enumerate(candidates)}
            truth = torch.tensor([pos[x] for x in labels], device=device)
            for branch, all_text in text_branches.items():
                text = all_text[idx].to(device)
                pred_parts = []
                for start in range(0, len(image), 256): pred_parts.append((image[start:start+256] @ text.T).argmax(1).cpu())
                pred = torch.cat(pred_parts).to(device); correct = pred.eq(truth)
                results[f"{kind}{seed}_{branch}"] = {"top1": float(correct.float().mean()),
                    "correct": int(correct.sum()), "rows": len(labels)}
    torch.save({"classes": all_classes, "branches": text_branches}, OUT / "all_text_features.pt")
    (OUT / "zeroshot_gate.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
