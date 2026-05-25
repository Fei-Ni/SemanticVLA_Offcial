# Trace Annotation Pipeline

This directory contains the code used to produce the dense end-effector trace
annotations shipped with the SemanticVLA TraceX 240K dataset collection
(Bridge / Fractal / BC-Z / DROID).

The trace files themselves live in the corresponding HuggingFace datasets,
embedded directly into every LeRobot frame row as the `trace.x` / `trace.y`
columns. This directory is for users who want to:

- understand how those annotations were produced,
- re-run the pipeline on a new dataset,
- or verify / visualise the released annotations against raw video.

## Pipeline overview

Two annotation pipelines are used. Bridge, Fractal, and BC-Z use a two-stage
vision-language pipeline; DROID uses a calibrated 3D projection.

### Two-stage pipeline (Bridge / Fractal / BC-Z)

| Stage | Code | What it does |
|---|---|---|
| **Stage 1** — sparse keyframes | [`stage1_keyframes_molmo/`](stage1_keyframes_molmo/) | Prompt Molmo-72B with *"point to the robot gripper"* on 10 fixed keyframes per episode. Output: sparse `(episode_idx, step_idx, x, y)` rows in `[0, 100]` normalised image coordinates. |
| **Stage 2** — dense propagation | [`stage2_dense_cotracker/`](stage2_dense_cotracker/) | Run CoTracker bidirectionally from each Stage 1 keyframe (forward + backward) and fuse the resulting candidate trajectories into a per-frame dense trace. |

### Projection pipeline (DROID)

| Stage | Code | What it does |
|---|---|---|
| **3D projection** | [`droid_projection/`](droid_projection/) | Use DROID's calibrated camera intrinsics + extrinsics + the `cartesian_position` end-effector pose to deterministically project the gripper centre into each image. No VLM, no CoTracker. |

### Verification / visualisation

| Code | What it does |
|---|---|
| [`check/`](check/) | Self-contained scripts to load an annotation file, render trajectory overlays on raw frames, and run integrity / coverage checks on a finished bundle. |

## Released annotations

The dense trace produced by these scripts ships in the HuggingFace TraceX 240K
collection. Per-dataset counts:

| Dataset | Method | Episodes | Frames | HF dataset |
|---|---|---:|---:|---|
| Bridge   | two-stage | 53,192 | 1,999,410 | [`SemanticVLA-TraceX-240K-Bridge`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-Bridge) |
| Fractal  | two-stage | 87,182 | 3,786,274 | [`SemanticVLA-TraceX-240K-Fractal`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-Fractal) |
| BC-Z     | two-stage | 39,350 | 5,471,693 | [`SemanticVLA-TraceX-240K-BC-Z`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-BC-Z) |
| DROID    | projection | 48,000 | 13,762,962 | [`SemanticVLA-TraceX-240K-DROID`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-DROID) |

## Coordinate convention

All scripts use the same coordinate convention for outputs:

- normalised image-space `(x, y)`
- range: `[0, 100]`
- `x` is the image *column* axis, `y` is the image *row* axis
- **not** pixel coordinates

To convert to pixels:

```python
px = round(x / 100.0 * (width - 1))
py = round(y / 100.0 * (height - 1))
```

## Row schema (Stage 2 / final output)

```json
{
  "episode_idx": 0,
  "step_idx": 4,
  "coordinate": [36.07, 21.61],
  "is_keyframe": true,
  "stage1_coordinate": [39.6, 18.2]
}
```

`stage1_coordinate` is only present on keyframe rows; it records the original
Molmo anchor before CoTracker interpolation.

## Quick start: verify a downloaded annotation

```bash
# 1. Download the annotation HF dataset of choice (Bridge / Fractal / BC-Z / DROID).
huggingface-cli download spikefly/SemanticVLA-TraceX-240K-Bridge \
    --repo-type dataset --local-dir ./bridge_trace

# 2. Render reference GIFs/PNGs overlaying the trace onto raw frames.
python check/visualize_check_bcz.py --help        # BC-Z
python check/visualize_check_fractal.py --help    # Fractal
python check/visualize_bridge.py --help           # Bridge

# 3. Bundle-level integrity check (row count, coordinate range, episode contiguity).
python check/verify_trace_bundle.py --help
```

## Reproducing the pipeline

```bash
# Stage 1 (Molmo-72B keyframe annotation, GPU-heavy)
python stage1_keyframes_molmo/bcz_keyframe_inference.py \
    --dataset_path <BCZ_RLDS_ROOT> \
    --output_dir <STAGE1_OUT>

# Stage 2 (CoTracker dense propagation)
python stage2_dense_cotracker/run_bcz_dense_trace.py \
    --dataset_path <BCZ_RLDS_ROOT> \
    --keyframe_json <STAGE1_OUT>/keyframes.json \
    --output_dir <STAGE2_OUT>

# Merge worker shards into a single JSON
python stage2_dense_cotracker/merge_results.py \
    --result_dir_pattern '<STAGE2_OUT>/worker_*' \
    --output_json <STAGE2_OUT>/dense_trace.json
```

DROID uses a different entry point:

```bash
python droid_projection/run_droid_projection_trace.py \
    --droid_rlds <DROID_RLDS_ROOT> \
    --output_dir <DROID_OUT>
```

See the README inside each subdirectory for detailed flag explanations.

## Acknowledgements

- **Molmo** ([allenai/Molmo](https://huggingface.co/allenai)) — the open VLM used
  for sparse keyframe pointing.
- **CoTracker** ([facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)) —
  the dense point-tracker used for Stage 2 propagation.
- **DROID** ([droid-dataset.github.io](https://droid-dataset.github.io/)) —
  provides the calibrated stereo + cartesian-position data that powers the
  DROID projection pipeline.
