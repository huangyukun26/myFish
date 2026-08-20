from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors_bytes


def load_state_dict_without_mmap(path: str | Path) -> dict[str, torch.Tensor]:
    data = Path(path).read_bytes()
    state = load_safetensors_bytes(data)
    del data
    return state


def patch_open_clip_safetensors_load_file() -> None:
    """Make open_clip load safetensors through bytes instead of mmap.

    Windows can fail to mmap multi-GB safetensors files when pagefile is small.
    This patch is intentionally opt-in because mmap is normally faster.
    """
    import safetensors.torch as safetensors_torch

    def load_file_without_mmap(filename: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        return load_state_dict_without_mmap(filename)

    safetensors_torch.load_file = load_file_without_mmap
