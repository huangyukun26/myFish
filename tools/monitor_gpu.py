from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishnet.env import gpu_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/gpu_monitor.csv"))
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means run until interrupted.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "index", "name", "util_percent", "memory_used_mb", "memory_total_mb", "temperature_c"]
    write_header = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        count = 0
        while True:
            snap = gpu_snapshot()
            timestamp = datetime.now().isoformat(timespec="seconds")
            if snap.get("available"):
                for gpu in snap["gpus"]:
                    row = {"timestamp": timestamp, **gpu}
                    writer.writerow(row)
                    print(row, flush=True)
            else:
                row = {
                    "timestamp": timestamp,
                    "index": -1,
                    "name": snap.get("error", "no gpu"),
                    "util_percent": "",
                    "memory_used_mb": "",
                    "memory_total_mb": "",
                    "temperature_c": "",
                }
                writer.writerow(row)
                print(row, flush=True)
            fp.flush()
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
