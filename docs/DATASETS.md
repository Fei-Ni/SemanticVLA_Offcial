# Datasets — TraceX 240K

We release four trace-annotated LeRobot v3 conversions, aggregated under the [TraceX 240K collection](https://hf.co/collections/spikefly/semanticvla-datasets):

| Dataset | Source | Robot | HF repo |
|---|---|---|---|
| Bridge   | BridgeData V2 (OXE `bridge_orig`)    | WidowX       | [`SemanticVLA-TraceX-240K-Bridge`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-Bridge) |
| Fractal  | RT-1 (OXE `fractal20220817_data`)    | Google Robot | [`SemanticVLA-TraceX-240K-Fractal`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-Fractal) |
| BC-Z     | BC-Z (OXE `bc_z`)                    | xArm         | [`SemanticVLA-TraceX-240K-BC-Z`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-BC-Z) |
| DROID    | DROID                                | Franka Panda | [`SemanticVLA-TraceX-240K-DROID`](https://huggingface.co/datasets/spikefly/SemanticVLA-TraceX-240K-DROID) |

All four follow the [LeRobot](https://github.com/huggingface/lerobot) v3 chunked Parquet + MP4 layout.

## Per-dataset layout

```
SemanticVLA-TraceX-240K-<name>/
├── meta/                       # info.json, episodes.jsonl, tasks.jsonl, stats.json, modality.json
├── data/                       # per-episode parquet shards
└── videos/                     # per-episode H264 mp4s
```

## Trace annotations

Dense end-effector / object traces are stored **directly in the frame rows** of each parquet shard:

| Column | Type | Meaning |
|---|---|---|
| `trace.x` | `float32` array | image-space x coordinates for the trace anchor points |
| `trace.y` | `float32` array | image-space y coordinates |
| `trace.present` | `bool` | whether a trace is present for this frame |

Anchor count is fixed per dataset (`12` for the released subsets). The same NPY trace index used in OXE LAM training is also derivable from these per-frame columns.

## How to load

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="spikefly/SemanticVLA-TraceX-240K-Bridge",
    repo_type="dataset",
)

# Then load with LeRobot's standard dataset class.
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(local)
```

## Per-dataset size

Exact `(episode, frame)` counts are recorded in the dataset card of each HF repo. Approximate scales:

| Dataset | Episodes | Frames |
|---|---:|---:|
| Bridge   | ~53,000 | ~2.0M |
| Fractal  | ~87,000 | ~3.8M |
| BC-Z     | ~39,000 | ~5.5M |
| DROID    | ~92,000 | ~28M  |

## Citation

If you use any of these datasets, please cite SemanticVLA **and** the corresponding upstream dataset (BridgeData V2, RT-1 / Fractal, BC-Z, DROID, Open X-Embodiment). BibTeX is in each dataset card.
