# Trace Annotation Check / Verification

Tools to verify, visualise, and inspect a trace annotation bundle.

## What you can check

| Question | Script |
|---|---|
| "Does my annotation file overlay correctly on the raw video?" | `visualize_check_bcz.py` / `visualize_check_fractal.py` / `visualize_bridge.py` — render GIFs / PNGs |
| "Is the annotation file structurally well-formed (row count, coordinate range, episode contiguity)?" | `verify_trace_bundle.py` |
| "What are the high-level statistics of an annotation file?" | `print_stats.py` |
| "Can I round-trip a single episode from the annotation back to the raw frame?" | `load_annotation_bcz.py` / `load_annotation_fractal.py` |

## Files

| File | Dataset | Purpose |
|---|---|---|
| `load_annotation_bcz.py` | BC-Z | Match annotation rows to raw TFRecord episodes; round-trip a single episode. |
| `load_annotation_fractal.py` | Fractal | Same for Fractal. |
| `visualize_check_bcz.py` | BC-Z | Overlay coords on raw frames → GIF + PNG. |
| `visualize_check_fractal.py` | Fractal | Same for Fractal. |
| `visualize_bridge.py` | Bridge | Overlay coords on raw Bridge TFRecord frames. |
| `print_stats.py` | any | Summary stats on a dense-trace JSON. |
| `verify_trace_bundle.py` | any | Streaming structural verifier (row count, coord ∈ `[0, 100]`, episode index contiguity, keyframe / interpolated counts). |

## Coordinate convention

All scripts assume the coordinate convention used by the rest of the trace
pipeline: normalised image-space `[0, 100]`, `x` is the column axis, `y` is the
row axis. See [`../README.md`](../README.md).

## Quick start

```bash
# 1. Print summary stats on the annotation JSON.
python print_stats.py /path/to/bcz_stage2_dense_trace.json

# 2. Round-trip a single episode (annotation row -> raw TFRecord frame).
python load_annotation_bcz.py \
    /path/to/bcz_rlds_root \
    /path/to/bcz_stage2_dense_trace.json \
    0

# 3. Render reference GIFs + per-keyframe PNGs for a handful of episodes.
python visualize_check_bcz.py \
    --dataset_root /path/to/bcz_rlds_root \
    --annotation   /path/to/bcz_stage2_dense_trace.json \
    --output_dir   ./viz_out \
    --episodes     0,100,5000,30000

# 4. Bundle-level verification (streams the JSON; safe on multi-GB files).
python verify_trace_bundle.py \
    --trace_root /path/to/trace_annotations \
    --droid_dir  /path/to/droid_projection_trace
```

## Verifier output

`verify_trace_bundle.py` writes a JSON report covering:

- per-dataset row + episode counts
- coordinate `(x, y)` min / max across all rows
- keyframe / interpolated row counts
- per-episode contiguity (every episode covers `step_idx = 0..T-1`)
- a sample of bad rows (if any)

A bundle is `ok` iff `status == ok` for every dataset.

## Dependencies

- `tensorflow` (for BC-Z / Fractal / Bridge TFRecord readers)
- `pillow`, `numpy`, `matplotlib` (for visualisation + GIF writing)
- `verify_trace_bundle.py` is intentionally **dependency-free** so it can be
  run anywhere with stock Python.
