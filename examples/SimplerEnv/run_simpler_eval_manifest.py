from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SIMPLER formal eval for a manifest shard.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes-per-task", type=int, default=24)
    parser.add_argument("--reuse-existing", action="store_true", default=False)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    repo_root = Path(__file__).resolve().parents[2]
    eval_script = repo_root / "examples" / "SimplerEnv" / "star_bridge_formal_eval_ms3.sh"

    records: list[dict[str, object]] = []
    for idx, item in enumerate(manifest["items"]):
        if idx % args.num_shards != args.shard_index:
            continue

        ckpt_path = Path(item["ckpt_path"]).expanduser().resolve()
        result: dict[str, object] = {
            "label": item["label"],
            "ckpt_path": str(ckpt_path),
            "source_group": item["source_group"],
            "job_id": item.get("job_id"),
            "run_id": item.get("run_id"),
            "ckpt_step": item["ckpt_step"],
            "status": "pending",
            "formal_summary_json": None,
        }

        existing_summary = item.get("existing_formal_summary")
        if args.reuse_existing and existing_summary:
            summary_path = Path(existing_summary).expanduser().resolve()
            if summary_path.is_file():
                result["status"] = "reused"
                result["formal_summary_json"] = str(summary_path)
                records.append(result)
                continue

        env = os.environ.copy()
        proc = subprocess.run(
            ["bash", str(eval_script), str(ckpt_path), str(args.episodes_per_task)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = proc.stdout
        stderr = proc.stderr
        result["returncode"] = proc.returncode
        result["stdout_tail"] = stdout[-4000:]
        result["stderr_tail"] = stderr[-4000:]

        formal_summary = None
        for line in stdout.splitlines():
            if line.startswith("FORMAL_SUMMARY_JSON="):
                formal_summary = line.split("=", 1)[1].strip()

        if proc.returncode == 0 and formal_summary:
            result["status"] = "completed"
            result["formal_summary_json"] = formal_summary
        else:
            result["status"] = "failed"

        records.append(result)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": str(manifest_path),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "episodes_per_task": args.episodes_per_task,
                "records": records,
            },
            f,
            indent=2,
        )

    print(f"SHARD_RESULT_JSON={output_path}")


if __name__ == "__main__":
    main()
