#!/usr/bin/env python3
"""Upload completed SemanticVLA TraceX LeRobot components to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder


DATASETS = {
    "bridge": {
        "repo_id": "spikefly/SemanticVLA-Bridge-TraceX",
        "folder": Path(
            os.environ.get(
                "TRACEX_BRIDGE_FOLDER",
                "${WORK_ROOT}/tracex/bridge_train_1.0.0_lerobot",
            )
        ),
    },
    "fractal": {
        "repo_id": "spikefly/SemanticVLA-Fractal-TraceX",
        "folder": Path(
            os.environ.get(
                "TRACEX_FRACTAL_FOLDER",
                "${WORK_ROOT}/tracex/fractal_train_0.1.0_lerobot",
            )
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--print-report-every", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run login with HF_HOME pointing to a writable cache."
        )

    api = HfApi(token=token)
    who = api.whoami(token=token)
    print(f"[hf] authenticated as {who.get('name') or who.get('fullname')}", flush=True)

    for name in args.datasets:
        spec = DATASETS[name]
        repo_id = spec["repo_id"]
        folder = spec["folder"]
        if not folder.exists():
            raise FileNotFoundError(folder)
        readme = folder / "README.md"
        if not readme.exists():
            raise FileNotFoundError(readme)

        print(f"[hf] create/ensure dataset repo {repo_id}", flush=True)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=bool(args.private), exist_ok=True)

        print(f"[hf] upload {folder} -> datasets/{repo_id}", flush=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=folder,
            repo_type="dataset",
            private=bool(args.private),
            ignore_patterns=[".cache/**", "*.tmp", "*.tmp.*", "__pycache__/**"],
            num_workers=int(args.workers),
            print_report=True,
            print_report_every=int(args.print_report_every),
        )
        print(f"[hf] done {repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
