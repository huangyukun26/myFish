from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--local-dir", type=Path, default=None)
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)
    files = [
        {"name": sibling.rfilename, "size": sibling.size}
        for sibling in info.siblings
        if args.include is None or any(Path(sibling.rfilename).match(pattern) for pattern in args.include)
    ]
    print(json.dumps({"repo_id": args.repo_id, "files": files}, indent=2), flush=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=args.include,
        local_dir=str(args.local_dir) if args.local_dir else None,
        local_dir_use_symlinks=False if args.local_dir else "auto",
        resume_download=True,
    )
    print(json.dumps({"repo_id": args.repo_id, "snapshot": path}, indent=2), flush=True)


if __name__ == "__main__":
    main()
