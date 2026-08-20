from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict


def gpu_snapshot() -> Dict[str, Any]:
    """Return a best-effort one-line GPU telemetry snapshot."""
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, util, mem_used, mem_total, temp = parts
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "util_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def torch_report() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    report: Dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        report["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return report


def environment_report() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_report(),
        "gpu": gpu_snapshot(),
    }


def print_environment_report() -> None:
    print(json.dumps(environment_report(), indent=2, ensure_ascii=False))

