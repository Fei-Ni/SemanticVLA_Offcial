---
pretty_name: SemanticVLA Fractal TraceX
tags:
- robotics
- lerobot
- fractal
- rt-1
- google-robot
- semanticvla
- tracex
license: other
---

# SemanticVLA Fractal TraceX

Trace-augmented LeRobot conversion of the Fractal / RT-1 Google Robot train split used by SemanticVLA. This package preserves the original RLDS step fields that can be represented in LeRobot parquet/video form and adds one dense per-frame trace column:

`observation.trace.xy`: `float32[2]`, normalized image-space `(x, y)` coordinates on a 0-100 scale.

## Contents

- Episodes: 87,182
- Frames: 3,786,274
- Videos: 87,182
- FPS: 5
- Robot type: `google_robot`
- Source dataset: Open-X-Embodiment `fractal20220817_data` v0.1.0
- Format: LeRobot v2-style `data/`, `videos/`, `meta/`

## Modalities

Video stream:

- `observation.images.image`

Parquet columns include original RT-1 action/observation fields such as `action.world_vector`, `action.rotation_delta`, `action.gripper_closedness_action`, `observation.workspace_bounds`, `observation.natural_language_instruction`, derived canonical `action` / `observation.state`, standard LeRobot indices, and `observation.trace.xy`.

## Coverage

The original Fractal train split has 87,212 episodes. The TraceX package contains the 87,182 episodes with valid dense trace annotations. The 30 episodes without dense trace are excluded; no synthetic fallback trace is inserted.

Use of this dataset is subject to the original Fractal / RT-1 dataset terms, plus the terms of the added SemanticVLA TraceX annotations.
