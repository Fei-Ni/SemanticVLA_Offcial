# RLDS to LeRobot Tools

Utilities for building self-aligned OpenX LeRobot mirrors and trace indexes for
OXE LAM training and TraceX dataset release packaging.

The release path is LeRobot v3. The full-preserve v2-style component builds are
kept as local intermediates because they are easy to verify episode-by-episode;
public component repos are repacked into chunked LeRobot v3 with:

- raw source observation/action fields preserved in `data/**/*.parquet`;
- source camera streams preserved under `videos/{video_key}/chunk-*/file-*.mp4`;
- dense trace supervision stored as scalar per-frame columns `trace.x` and
  `trace.y`;
- no standalone JSON trace sidecar in the public data package.

The first implemented utility is `build_trace_index.py`, which converts the
large flat dense-trace JSON files into per-array `.npy` files:

```bash
python tools/rlds_to_lerobot/build_trace_index.py \
  --datasets bridge fractal bcz \
  --output-dir /home/u6gs/spikefly.u6gs/trace_annotations/_npy_index \
  --overwrite
```

The resulting `*_coords.npy` files can be opened with
`np.load(path, mmap_mode="r")` by the training dataloader.

Current v3 component repos:

- `spikefly/SemanticVLA-TraceX-240K-Bridge`
- `spikefly/SemanticVLA-TraceX-240K-Fractal`
- `spikefly/SemanticVLA-TraceX-240K-BC-Z`
- `spikefly/SemanticVLA-TraceX-240K-DROID`
