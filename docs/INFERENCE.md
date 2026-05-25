# Inference

This page covers loading the three released checkpoints and running them on the bundled evaluators (LIBERO and SimplerEnv WidowX). For real-robot deployment see [`examples/REAL/arx/`](../examples/REAL/arx/).

## 1. Install

```bash
pip install -e .
```

## 2. Download checkpoints

```bash
pip install -U huggingface_hub
huggingface-cli login   # optional, only if rate-limited

# 2a. Unified OXE LAM (only needed if you want to precompute LAM labels yourself).
huggingface-cli download spikefly/SemanticVLA-LAM --local-dir ./SemanticVLA-LAM

# 2b. LIBERO policy.
huggingface-cli download spikefly/SemanticVLA-LIBERO --local-dir ./SemanticVLA-LIBERO

# 2c. SimplerEnv WidowX policy.
huggingface-cli download spikefly/SemanticVLA-SimplerEnv --local-dir ./SemanticVLA-SimplerEnv
```

## 3. Load a VLA policy

Both VLA policies (`SemanticVLA-LIBERO` and `SemanticVLA-SimplerEnv`) share the same loader:

```python
from semanticvla.model.framework.base_framework import baseframework

policy = baseframework.from_pretrained("./SemanticVLA-LIBERO/pytorch_model.pt")
policy.eval()
```

The loader walks two directory levels up from the checkpoint file to locate `config.yaml` and `dataset_statistics.json`. The released layout follows this convention:

```
SemanticVLA-LIBERO/
├── config.yaml
├── dataset_statistics.json
└── final_model/
    └── pytorch_model.pt
```

## 4. Load the LAM

```python
import yaml, torch
from semanticvla.model.modules.latent_action_model import TraceLatentActionModel

cfg = yaml.safe_load(open("./SemanticVLA-LAM/config.yaml"))

lam = TraceLatentActionModel.from_config(cfg["model"], variant=cfg["variant"])
state = torch.load("./SemanticVLA-LAM/pytorch_model.pt", map_location="cpu")
lam.load_state_dict(state)
lam.eval()
```

DINOv2 weights are not bundled with the LAM repo — set `cfg["model"]["dino_repo_root"]` and `cfg["model"]["dino_weights"]` to point at your local DINOv2 ViT-B/14 installation before loading.

## 5. Run LIBERO

```bash
cd examples/LIBERO
bash eval_libero.sh
```

Expected mean SR (4 suites): **0.982**.

| Suite | SR |
|---|---:|
| LIBERO-Spatial | 0.988 |
| LIBERO-Object  | 0.996 |
| LIBERO-Goal    | 0.974 |
| LIBERO-10      | 0.970 |

## 6. Run SimplerEnv WidowX

```bash
cd examples/SimplerEnv
bash start_simpler_env.sh   # or start_simpler_env_ms3.sh for ManiSkill3
```

Expected mean SR (4 tasks): **0.802**.

| Task | SR |
|---|---:|
| Put Eggplant in Basket | 0.958 |
| Spoon on Towel         | 1.000 |
| Carrot on Plate        | 0.792 |
| Stack Cube             | 0.458 |

## 7. Serve over WebSocket

For policy-server style evaluation (used by both SimplerEnv and real-robot clients):

```bash
bash examples/SimplerEnv/run_policy_server_official.sh \
    --checkpoint ./SemanticVLA-LIBERO/pytorch_model.pt
```

The WebSocket server / client code is in [`deployment/model_server/`](../deployment/model_server/) and reused by [`examples/REAL/arx/`](../examples/REAL/arx/).
