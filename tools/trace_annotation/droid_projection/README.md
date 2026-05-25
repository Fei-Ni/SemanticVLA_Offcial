# DROID — Calibrated Projection Trace

DROID uses a different pipeline from Bridge / Fractal / BC-Z. Instead of
prompting a VLM and propagating with CoTracker, the gripper trace is computed
**deterministically** by projecting the calibrated end-effector pose into each
camera frame.

## Why projection (not Molmo + CoTracker)?

- DROID ships per-episode **calibrated** stereo extrinsics and intrinsics.
- DROID ships per-frame `cartesian_position` for the end-effector.
- Together these are enough to compute the 2D pixel position of the gripper
  exactly, with no model in the loop.

So Stage 1 + Stage 2 of the VLM pipeline collapse into a single deterministic
projection step.

## Files

| File | Purpose |
|---|---|
| `run_droid_projection_trace.py` | Project the end-effector pose into each camera view (`exterior_image_1_left`, `exterior_image_2_left`). |
| `fill_droid_null_coordinates.py` | Post-process: dense-fill the few frames where calibration is missing using clip-first interpolation. |
| `build_droid_48k_trace.py` | Select the 48k-episode subset, merge ext1 + ext2 outputs into the final bundle. |
| `inspect_droid_coverage.py` | Report which episodes have valid calibration for which camera view (audit tool). |

## Output schema

Same as the two-stage pipeline (see [`../README.md`](../README.md)) so the
downstream loader is identical:

```json
{
  "episode_idx": 0,
  "step_idx": 4,
  "coordinate": [36.07, 21.61],
  "is_keyframe": false,
  "is_interpolated": false
}
```

`is_keyframe` is always `false` for projection output (every frame is computed
directly). `is_interpolated` is `true` for frames filled by
`fill_droid_null_coordinates.py`.

## Requirements

- DROID RLDS root (with per-episode calibration JSONs)
- `numpy`, `tensorflow` (for TFRecord IO)
- No GPU required.

## Usage

```bash
# 1. Project the gripper into every episode of an external camera view.
python run_droid_projection_trace.py \
    --droid_rlds   /path/to/droid_rlds_root \
    --camera       exterior_image_1_left \
    --output_dir   /path/to/droid_out/ext1

python run_droid_projection_trace.py \
    --droid_rlds   /path/to/droid_rlds_root \
    --camera       exterior_image_2_left \
    --output_dir   /path/to/droid_out/ext2

# 2. Fill nulls (frames where calibration was missing).
python fill_droid_null_coordinates.py \
    --input_dir  /path/to/droid_out/ext1 \
    --output_dir /path/to/droid_out/ext1_dense

# 3. Build the final 48K bundle from ext1 + ext2.
python build_droid_48k_trace.py \
    --ext1_dir   /path/to/droid_out/ext1_dense \
    --ext2_dir   /path/to/droid_out/ext2_dense \
    --output_dir /path/to/droid_out/final_48k

# 4. Audit which episodes have valid coverage.
python inspect_droid_coverage.py \
    --bundle_dir /path/to/droid_out/final_48k
```

## Notes

- The DROID 48K release uses `exterior_image_1_left` + `exterior_image_2_left`
  as parallel camera views: each episode contributes up to 2 trace bundles.
- The DROID trace is included in
  [`SemanticVLA-TraceX-240K-DROID`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-DROID).
