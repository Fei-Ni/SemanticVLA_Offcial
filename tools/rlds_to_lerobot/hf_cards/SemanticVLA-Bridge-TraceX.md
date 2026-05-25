---
pretty_name: SemanticVLA Bridge TraceX
tags:
- robotics
- lerobot
- bridge
- widowx
- semanticvla
- tracex
license: other
---

# SemanticVLA Bridge TraceX

Trace-augmented LeRobot conversion of the BridgeData V2 / WidowX train split used by SemanticVLA. This package preserves the original RLDS step fields that can be represented in LeRobot parquet/video form and adds one dense per-frame trace column:

`observation.trace.xy`: `float32[2]`, normalized image-space `(x, y)` coordinates on a 0-100 scale.

## Contents

- Episodes: 53,192
- Frames: 1,999,410
- Videos: 212,768
- FPS: 5
- Robot type: `widowx`
- Source dataset: BridgeData V2 `bridge_dataset` v1.0.0
- Format: LeRobot v2-style `data/`, `videos/`, `meta/`

## Modalities

Video streams:

- `observation.images.image_0`
- `observation.images.image_1`
- `observation.images.image_2`
- `observation.images.image_3`

Parquet columns include the original Bridge action, language, reward/discount/terminal fields, `observation.state`, standard LeRobot indices, and `observation.trace.xy`.

## Notes

The Hugging Face Bridge LeRobot release was used only as a package-shape reference. Episode order and trace alignment are derived from our local raw RLDS shard order and SemanticVLA trace annotations.

Use of this dataset is subject to the original BridgeData V2 terms, plus the terms of the added SemanticVLA TraceX annotations.
