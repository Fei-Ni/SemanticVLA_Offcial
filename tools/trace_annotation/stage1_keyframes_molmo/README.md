# Stage 1 — Sparse Keyframe Annotation with Molmo-72B

This stage produces sparse `(episode_idx, step_idx, x, y)` keyframe annotations
by prompting Molmo-72B with *"point to the robot gripper"* on a small set of
fixed keyframes per episode.

## Keyframe selection

For each episode, 10 keyframes are picked deterministically:

- first frame (frame `0`)
- last frame (frame `T - 1`)
- 8 evenly-spaced middle frames between them

Short episodes (< 12 frames) fall back to sampling every 2 frames.

## Output format

```json
[
  {
    "episode_idx": 0,
    "step_idx": 0,
    "coordinate": [39.6, 18.2],
    "is_keyframe": true
  },
  ...
]
```

Coordinates are in normalised image space, `[0, 100]`.

## Scripts

| Script | Dataset |
|---|---|
| `bcz_keyframe_inference.py` | BC-Z (xArm) RLDS |
| `fractal_keyframe_inference.py` | Fractal (RT-1, Google Robot) RLDS |

Both scripts share the same overall structure:

1. Read TFRecord shards for the target episodes.
2. Pick keyframes per the strategy above.
3. Batch-call Molmo-72B with the gripper-pointing prompt.
4. Parse the response into `(x, y)` and write a JSON output (with shard-based
   resumption support so multi-GPU / multi-node workers can co-exist).

## Requirements

- Molmo-72B weights ([allenai/Molmo-72B-0924](https://huggingface.co/allenai/Molmo-72B-0924))
  cached locally; needs a large GPU (we used a 4 x H100 node).
- `torch`, `transformers`, `tensorflow` (for TFRecord IO), `pillow`, `numpy`.

## Usage

```bash
python bcz_keyframe_inference.py \
    --dataset_path  /path/to/bcz_rlds_root \
    --output_dir    /path/to/stage1_out \
    --molmo_model   allenai/Molmo-72B-0924 \
    --max_keyframes 10 \
    --keyframe_batch_size 4 \
    --start_episode 0 \
    --end_episode   39350
```

For multi-node use the `--total_nodes` / `--current_node` flags to partition
episodes evenly across workers; each worker writes its own shard, which can be
merged with `stage2_dense_cotracker/merge_results.py` later.

## Notes

- Stage 1 *only* produces keyframe rows. Dense per-frame propagation is done in
  [`../stage2_dense_cotracker/`](../stage2_dense_cotracker/).
- Coordinate parsing tolerates several Molmo output formats (XML `<point>`,
  `coordinates: (x, y)`, `(x, y)`, `x: ..., y: ...`).
- All outputs are atomic-write (temp file + rename) so a killed worker does not
  leave a partially-written JSON.
