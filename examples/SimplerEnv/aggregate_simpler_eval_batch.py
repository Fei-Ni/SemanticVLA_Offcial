from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_ORDER = [
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SIMPLER formal eval shard outputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-results", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    by_ckpt: dict[str, dict[str, object]] = {}
    for shard_path_str in args.shard_results:
        shard_path = Path(shard_path_str).expanduser().resolve()
        with shard_path.open("r", encoding="utf-8") as f:
            shard = json.load(f)
        for record in shard["records"]:
            by_ckpt[record["ckpt_path"]] = record

    rows: list[dict[str, object]] = []
    for item in manifest["items"]:
        ckpt_path = item["ckpt_path"]
        record = by_ckpt.get(ckpt_path)
        if not record:
            raise KeyError(f"Missing shard result for {ckpt_path}")
        if record["status"] not in {"completed", "reused"}:
            raise RuntimeError(f"Eval did not complete for {ckpt_path}: {record['status']}")

        summary_path = Path(record["formal_summary_json"]).expanduser().resolve()
        summary = load_summary(summary_path)

        row = {
            "label": item["label"],
            "source_group": item["source_group"],
            "job_id": item.get("job_id"),
            "run_id": item.get("run_id"),
            "ckpt_step": item["ckpt_step"],
            "formal_summary_json": str(summary_path),
            "windowx_mean_success": summary["windowx_mean_success"],
        }
        for task in TASK_ORDER:
            row[task] = summary["task_success"][task]
        rows.append(row)

    rows.sort(key=lambda x: (-float(x["windowx_mean_success"]), x["label"]))

    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"manifest": str(manifest_path), "rows": rows}, f, indent=2)

    fieldnames = [
        "label",
        "source_group",
        "job_id",
        "run_id",
        "ckpt_step",
        "windowx_mean_success",
        *TASK_ORDER,
        "formal_summary_json",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| label | group | ckpt step | avg | eggplant | spoon | carrot | stack |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {group} | {step} | {avg:.4f} | {egg:.4f} | {spoon:.4f} | {carrot:.4f} | {stack:.4f} |".format(
                label=row["label"],
                group=row["source_group"],
                step=row["ckpt_step"],
                avg=float(row["windowx_mean_success"]),
                egg=float(row["widowx_put_eggplant_in_basket"]),
                spoon=float(row["widowx_spoon_on_towel"]),
                carrot=float(row["widowx_carrot_on_plate"]),
                stack=float(row["widowx_stack_cube"]),
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"AGG_JSON={output_json}")
    print(f"AGG_CSV={output_csv}")
    print(f"AGG_MD={output_md}")


if __name__ == "__main__":
    main()
