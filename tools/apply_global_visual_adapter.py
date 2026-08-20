from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


class IdentityAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(3.5))
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize(self.proj(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    payload = torch.load(args.input_features, map_location="cpu", weights_only=False)
    features = normalize(payload["features"])
    ckpt = torch.load(args.adapter, map_location="cpu", weights_only=False)
    dim = int(features.shape[1])
    model = IdentityAdapter(dim)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], args.batch_size):
            end = min(start + args.batch_size, features.shape[0])
            chunks.append(model(features[start:end].to(device)).cpu())
    out_payload = dict(payload)
    out_payload["features"] = torch.cat(chunks, dim=0)
    out_payload["adapter"] = str(args.adapter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.out)
    summary = {
        "adapter": str(args.adapter),
        "input_features": str(args.input_features),
        "out": str(args.out),
        "rows": int(features.shape[0]),
        "dim": dim,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
