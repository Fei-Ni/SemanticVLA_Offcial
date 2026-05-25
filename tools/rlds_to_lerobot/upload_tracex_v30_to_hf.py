#!/usr/bin/env python3
"""Upload a SemanticVLA TraceX LeRobot v3.0 package to Hugging Face."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError


DEFAULT_REPO_ID = "spikefly/SemanticVLA-Bridge-TraceX-v3"
DEFAULT_FOLDER = Path(
    os.environ.get(
        "TRACEX_V30_FOLDER",
        "${WORK_ROOT}/tracex_v30/bridge_train_1.0.0_lerobot_v30",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("TRACEX_V30_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--print-report-every", type=int, default=30)
    parser.add_argument("--tag", default="v3.0")
    parser.add_argument("--retry-sleep", type=int, default=60)
    return parser.parse_args()


def _retry_hf(label: str, fn: Callable[[], Any], *, sleep_s: int, attempts: int = 6) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except HfHubHTTPError as exc:
            if "429" not in str(exc) or attempt == attempts:
                raise
            print(f"[hf] {label} hit rate limit; retry {attempt}/{attempts - 1} after {sleep_s}s", flush=True)
            time.sleep(max(1, int(sleep_s)))


def main() -> int:
    args = parse_args()
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Set HF_TOKEN or configure HF_HOME with a token.")
    folder = args.folder
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not (folder / "README.md").exists():
        raise FileNotFoundError(folder / "README.md")

    api = HfApi(token=token)
    try:
        who = api.whoami(token=token)
        print(f"[hf] authenticated as {who.get('name') or who.get('fullname')}", flush=True)
    except HfHubHTTPError as exc:
        if "429" not in str(exc):
            raise
        print("[hf] whoami hit rate limit; continuing with cached token", flush=True)
    print(f"[hf] create/ensure dataset repo {args.repo_id}", flush=True)
    _retry_hf(
        "create_repo",
        lambda: api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=bool(args.private), exist_ok=True),
        sleep_s=int(args.retry_sleep),
    )
    print(f"[hf] upload {folder} -> datasets/{args.repo_id}", flush=True)
    _retry_hf(
        "upload_large_folder",
        lambda: api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=folder,
            repo_type="dataset",
            private=bool(args.private),
            ignore_patterns=[".cache/**", "*.tmp", "*.tmp.*", "__pycache__/**"],
            num_workers=int(args.workers),
            print_report=True,
            print_report_every=int(args.print_report_every),
        ),
        sleep_s=int(args.retry_sleep),
    )
    if args.tag:
        try:
            api.delete_tag(args.repo_id, tag=args.tag, repo_type="dataset")
        except HfHubHTTPError:
            pass
        _retry_hf(
            "create_tag",
            lambda: api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset"),
            sleep_s=int(args.retry_sleep),
        )
        print(f"[hf] tagged {args.repo_id} as {args.tag}", flush=True)
    print(f"[hf] done {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
