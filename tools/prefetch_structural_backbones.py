from __future__ import annotations

import argparse
import gc
import json

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--siglip2", default="google/siglip2-base-patch16-naflex")
    args = parser.parse_args()

    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {"device": str(device)}
    dino = timm.create_model(
        args.dinov3,
        pretrained=True,
        num_classes=0,
        img_size=512,
    ).to(device).eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        tokens = dino.forward_features(torch.zeros(1, 3, 512, 512, device=device))
        dino_features = torch.cat(
            [tokens[:, 0], tokens[:, dino.num_prefix_tokens :].mean(dim=1)], dim=1
        )
    report["dinov3"] = {
        "model": args.dinov3,
        "tokens": list(tokens.shape),
        "concat_features": list(dino_features.shape),
    }
    del dino, tokens, dino_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.siglip2)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    siglip = AutoModel.from_pretrained(
        args.siglip2,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    images = [Image.new("RGB", (960, 360), (128, 128, 128))]
    inputs = processor(images=images, max_num_patches=256, return_tensors="pt")
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.inference_mode():
        siglip_features = siglip.get_image_features(**inputs)
    report["siglip2"] = {
        "model": args.siglip2,
        "features": list(siglip_features.shape),
        "processor_tensors": {key: list(value.shape) for key, value in inputs.items()},
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
