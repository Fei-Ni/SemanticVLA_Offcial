#!/usr/bin/env python3
"""Summarize SimplerEnv WidowX eval summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ORDER = [
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True, help="Root containing task/*/summary.json files.")
    parser.add_argument("--baseline-root", type=Path, default=None, help="Optional baseline eval root to compare against.")
    parser.add_argument("--label", default=None, help="Label for the eval being summarized.")
    parser.add_argument("--baseline-label", default="baseline", help="Label for the baseline.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _latest_summary_for_task(root: Path, task: str) -> Path | None:
    candidates = sorted((root / task).glob("*/summary.json"))
    if candidates:
        return candidates[-1]
    flat = root / task / "summary.json"
    return flat if flat.exists() else None


def _load_task_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    episodes = data.get("episodes") or []
    num_episodes = int(data.get("num_episodes") or len(episodes))
    if episodes:
        success_count = int(sum(float(ep.get("success", 0.0)) for ep in episodes))
        mean_success = success_count / max(num_episodes, 1)
    else:
        mean_success = float(data.get("mean_success", 0.0))
        success_count = int(round(mean_success * num_episodes))
    steps = [int(ep.get("steps", 0)) for ep in episodes if "steps" in ep]
    return {
        "task": str(data.get("task") or path.parents[1].name),
        "summary_path": str(path),
        "num_episodes": num_episodes,
        "success_count": success_count,
        "mean_success": mean_success,
        "avg_steps": (sum(steps) / len(steps)) if steps else None,
        "ckpt_path": data.get("ckpt_path"),
        "save_tag": data.get("save_tag"),
    }


def collect(root: Path) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    discovered = {p.parent.parent.name for p in root.glob("*/**/summary.json")}
    ordered = [task for task in TASK_ORDER if task in discovered or (root / task).exists()]
    ordered.extend(sorted(discovered - set(ordered)))
    for task in ordered:
        path = _latest_summary_for_task(root, task)
        if path is None:
            continue
        tasks[task] = _load_task_summary(path)
    macro = sum(item["mean_success"] for item in tasks.values()) / max(len(tasks), 1)
    return {
        "root": str(root),
        "num_tasks": len(tasks),
        "macro_success": macro,
        "tasks": tasks,
    }


def with_comparison(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return current
    out = dict(current)
    out["baseline_root"] = baseline["root"]
    out["baseline_macro_success"] = baseline["macro_success"]
    out["macro_delta"] = current["macro_success"] - baseline["macro_success"]
    compared = {}
    for task, item in current["tasks"].items():
        base = baseline["tasks"].get(task)
        if base is None:
            continue
        compared[task] = {
            "baseline_mean_success": base["mean_success"],
            "delta": item["mean_success"] - base["mean_success"],
        }
    out["comparison"] = compared
    return out


def format_md(summary: dict[str, Any], label: str, baseline_label: str) -> str:
    lines = [
        f"# SimplerEnv Eval Summary: {label}",
        "",
        f"- eval root: `{summary['root']}`",
        f"- tasks: `{summary['num_tasks']}`",
        f"- macro SR: `{summary['macro_success']:.4f}`",
    ]
    if "baseline_macro_success" in summary:
        lines.extend(
            [
                f"- {baseline_label} macro SR: `{summary['baseline_macro_success']:.4f}`",
                f"- macro delta: `{summary['macro_delta']:+.4f}`",
            ]
        )
    lines.extend(["", "| Task | SR | Successes | Avg steps | Delta |", "|---|---:|---:|---:|---:|"])
    for task in [*TASK_ORDER, *sorted(set(summary["tasks"]) - set(TASK_ORDER))]:
        if task not in summary["tasks"]:
            continue
        item = summary["tasks"][task]
        delta = ""
        if "comparison" in summary and task in summary["comparison"]:
            delta = f"{summary['comparison'][task]['delta']:+.4f}"
        avg_steps = "" if item["avg_steps"] is None else f"{item['avg_steps']:.1f}"
        lines.append(
            f"| `{task}` | {item['mean_success']:.4f} | "
            f"{item['success_count']}/{item['num_episodes']} | {avg_steps} | {delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    current = collect(args.eval_root)
    baseline = collect(args.baseline_root) if args.baseline_root else None
    summary = with_comparison(current, baseline)
    label = args.label or args.eval_root.name
    md = format_md(summary, label=label, baseline_label=args.baseline_label)

    print(md)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
