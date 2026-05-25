# Trace overlay visualisation

Scripts used to render the orange trace overlays shown in the main README and the per-dataset HF dataset cards.

## Files

| Script | Purpose |
|---|---|
| [`render_trace_gifs.py`](render_trace_gifs.py) | For each of the 4 TraceX-240K datasets (Bridge / Fractal / BC-Z / DROID), pick *N* evenly-spaced episodes, decode their MP4 frames, and overlay the per-frame `trace.x` / `trace.y` columns as an orange gradient trail with a bright current point. Outputs one GIF per episode. |
| [`make_composite_4x4.py`](make_composite_4x4.py) | Combine 16 hand-picked single-episode GIFs (4 per dataset) into a single 4×4 composite GIF for the README hero. |

## Style

The overlay style is intentionally minimal — one bright color, a short gradient trail, a compact top-left label box. It matches the look used by the reference handoff verification tooling so visualisations across our release stay consistent.

- **Orange `(255, 145, 28)`** progressive trail (line + dot, 14-frame window, alpha fades along the trail direction).
- **Current point**: filled orange disc with a 2-pixel black outline.
- **Label box**: top-left corner, semi-transparent black background, white text. The last line (the task instruction) is highlighted in yellow when the label has 3+ lines.

## DROID camera handling

DROID episodes have an extra metadata field — `source.camera_for_trace` — that records which of the two exterior cameras the trace was projected onto (since per-episode calibration quality determined the choice). `render_trace_gifs.py` reads that field for every DROID episode and overlays on the matching camera. If you forget this and always render onto `exterior_image_1_left`, roughly half of the DROID episodes will appear visibly misaligned.

## Usage

```bash
# Render the 4-per-dataset gallery (40 GIFs total) used in the dataset READMEs.
# Output: <OUT_BASE>/<dataset>/episode_<id>.gif
python tools/visualize/render_trace_gifs.py

# Or render one specific dataset only:
python tools/visualize/render_trace_gifs.py bridge

# Build the 4x4 composite hero shown in the main README.
# Input: the 16 GIFs hand-picked from the previous step.
# Output: <OUT_BASE>/composite_4x4.gif
python tools/visualize/make_composite_4x4.py
```

Both scripts have absolute paths hard-coded at the top (the source LeRobot v3 datasets live under our `/projects/...` data root). Edit `ROOT_BASE` and `OUT_BASE` before running on a different machine.

## Dependencies

- `pyarrow` (for reading the v3 LeRobot parquet meta + data shards)
- `opencv-python` (for MP4 decoding)
- `imageio` (for GIF encoding)
- `pillow` (for the overlay drawing)
- `numpy`
