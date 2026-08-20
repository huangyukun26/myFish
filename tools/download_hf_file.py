from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("filename")
    args = parser.parse_args()

    info = HfApi().model_info(args.repo_id, files_metadata=True)
    match = next((s for s in info.siblings if s.rfilename == args.filename), None)
    print(
        json.dumps(
            {"repo_id": args.repo_id, "filename": args.filename, "size": getattr(match, "size", None)},
            indent=2,
        ),
        flush=True,
    )
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        resume_download=True,
    )
    print(json.dumps({"path": path}, indent=2), flush=True)


if __name__ == "__main__":
    main()
