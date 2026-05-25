# Stage 2 — Dense Trace via CoTracker

This stage takes the sparse Molmo-72B keyframes from Stage 1 and propagates
them into a per-frame dense trace using
[CoTracker](https://github.com/facebookresearch/co-tracker).

## How it works

For each episode:

1. Load the full episode video.
2. For each Stage 1 keyframe `(frame_idx, x_kf, y_kf)`:
   - Run CoTracker **forward** from the keyframe to the end of the episode.
   - Run CoTracker **backward** from the keyframe to the start of the episode.
3. Concatenate the forward + backward halves to produce one candidate
   trajectory anchored at this keyframe.
4. **Filter** candidates that look broken (low visibility, large per-step jumps,
   degenerate span, near-zero motion).
5. **Fuse** the surviving candidates frame-by-frame using median / mean / a
   small consistency check, producing a single per-frame `(x, y)`.

The result is a JSON file with one row per `(episode, step)`, see
`row schema` in [`../README.md`](../README.md).

## Files

| File / dir | Purpose |
|---|---|
| `component/` | Shared CoTracker integration (model loader, tracking, fusion, scatter / GIF visualisation, animation). Identical across datasets. |
| `dataset_io/bcz_data.py` | BC-Z TFRecord loader + indexing. |
| `dataset_io/fractal_data.py` | Fractal TFRecord loader + indexing. |
| `run_bcz_dense_trace.py` | Per-worker driver for BC-Z. |
| `run_fractal_dense_trace.py` | Per-worker driver for Fractal. |
| `merge_results.py` | Merge per-worker JSON shards into a single output file. |

Bridge does **not** ship its own Stage 2 driver here — the released Bridge
dense trace was produced by the same algorithm on a different cluster; only
the visualisation helper [`../check/visualize_bridge.py`](../check/visualize_bridge.py)
ships in this repo for verifying the public artifact.

## Requirements

- `torch` (CUDA) with a CoTracker-compatible build
- `opencv-python` (for frame resizing + optional MP4 writing)
- `pillow`, `numpy`, `tensorflow` (for raw TFRecord IO)
- CoTracker is loaded via `torch.hub` from facebookresearch's repo on first run.

## Usage

```bash
# Per-worker (writes one JSON shard)
python run_bcz_dense_trace.py \
    --dataset_path  /path/to/bcz_rlds_root \
    --keyframe_json /path/to/stage1/keyframes.json \
    --output_dir    /path/to/stage2_out/worker_0 \
    --start_episode 0 \
    --end_episode   10000 \
    --device        cuda:0

# Merge all worker shards
python merge_results.py \
    --result_dir_pattern '/path/to/stage2_out/worker_*' \
    --output_json /path/to/stage2_out/dense_trace.json
```

For multi-node jobs use `--total_nodes` / `--current_node` to auto-partition
the episode range across workers.

## Fusion knobs

The `FusionConfig` dataclass in `component/fusion.py` exposes the candidate-
filter thresholds (visibility floor, max per-step jump, span minimum, etc.) and
the fusion mode (`median` / `mean` / `consistency`). Defaults match what
produced the released trace; override via CLI for experimentation.

## Output

Per-worker output:

```
<output_dir>/
├── worker_run.log
├── json_shards/
│   ├── shard_00000.json
│   └── shard_00001.json
├── shard_manifest.json
└── <output_json_name>.json     # merged-on-completion view (worker-local)
```

After running `merge_results.py` across all workers you get a single
`dense_trace.json` matching the schema in [`../README.md`](../README.md).
