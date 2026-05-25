# Training

Three reproduction recipes are shipped with this repo:

1. **LIBERO** — finetune SemanticVLA on the LIBERO benchmark.
2. **OXE** — full three-stage recipe: trace label → unified OXE LAM → 100K OXE VLA.
3. **Real robot** — Arx5 server example (no training, deployment only). See [`examples/REAL/arx/`](../examples/REAL/arx/).

Each example folder has a `README.md` and runnable shell entrypoints. Below is the high-level flow.

## 0. Environment

```bash
pip install -e .
# DINOv2 weights are needed for LAM training; download ViT-B/14:
#   https://github.com/facebookresearch/dinov2
```

## 1. LIBERO ([`examples/LIBERO/`](../examples/LIBERO/))

```bash
cd examples/LIBERO

# 1a. Convert LIBERO data to LeRobot v3 (one-time).
bash prepare_libero_lerobot_data.sh

# 1b. Generate modality JSON (one-time).
python generate_libero_modality_json.py

# 1c. Train SemanticVLA on LIBERO (uses the unified OXE LAM as the tokenizer).
bash run_libero_train.sh
# or, 4-GPU:
bash run_libero_train_4gpu.sh

# 1d. Evaluate.
bash eval_libero.sh
```

LAM labels for LIBERO are precomputed from the released `spikefly/SemanticVLA-LAM` checkpoint via `precompute_lam_labels.py`.

## 2. OXE ([`examples/OXE/`](../examples/OXE/))

Three stages. Each stage is independent and resumable.

### Stage A — Dense trace labels

Build the `(episode, step)` → `(x, y)` NPY index for each OXE subset (Bridge, Fractal, BC-Z). The trace labels are also included in the released `TraceX 240K` datasets, so most users can skip this stage and download directly.

### Stage B — Train the unified OXE LAM

```bash
cd examples/OXE
python train_lam_oxe.py --config configs/oxe_lam_paper_strict.yaml
```

The `paper_strict` config is what produces the released `SemanticVLA-LAM`. Training runs ~50K steps on 4 GPUs and consumes the trace NPY index from Stage A.

### Stage C — Train the VLA with LAM labels (100K)

```bash
# Precompute LAM labels with the trained LAM (one-time per dataset).
python precompute_lam_labels_oxe.py \
    --lam-checkpoint /path/to/SemanticVLA-LAM/pytorch_model.pt \
    --lam-config     /path/to/SemanticVLA-LAM/config.yaml \
    --dataset bridge

# Train the VLA for 100K steps.
bash train_oxe_vla.sh
```

The released `SemanticVLA-SimplerEnv` checkpoint uses BridgeData V2 in Stage C. For multi-dataset training, swap the dataset argument and adjust the mix in `train_oxe_vla.sh`.

## 3. Hyperparameter reference

The configs that ship with the repo match what produced the released checkpoints. Highlights:

| Item | LAM (`paper_strict`) | LIBERO VLA | OXE VLA |
|---|---|---|---|
| Image resolution | 224 × 224 | 224 × 224 | 224 × 224 |
| Trace window `W` | 12 | 12 | 12 |
| LAM latent tokens `K` | 4 | 4 (consumed) | 4 (consumed) |
| LAM vocab `V` | 32 | 32 (consumed) | 32 (consumed) |
| Action horizon `H` | 8 (training-time) | 8 | 16 |
| LM loss weight | — | 0.10 | 0.10 |
| Default steps | ~50K | ~30K | 100K |

## 4. Sanity tests

A small smoke test that loads each released checkpoint and runs a single forward pass is in [`examples/LIBERO/semanticvla/tests/`](../examples/LIBERO/semanticvla/tests/) (or run `pytest` from the repo root).
