# Method

SemanticVLA augments a Qwen-VL action backbone with two complementary semantic signals supervised in the language stream of the VLM:

1. **Trace prediction** — the VLM is asked to autoregressively emit short tokenized end-effector / object trace sequences for the next few seconds.
2. **Latent Action Model (LAM) tokens** — a discrete codebook is learned offline from short trace + future-action segments, and the VLM is asked to emit the corresponding codebook tokens.

Both heads are co-trained with the standard VLA action loss. The continuous action head (a DiT-B flow-matching expert) is unchanged; the semantic heads only modify the language-side training target.

## Notation

| Symbol | Meaning |
|---|---|
| `o_t` | image observation at time `t` |
| `l` | language instruction |
| `a_{t:t+H}` | continuous action chunk over horizon `H` |
| `τ_{t:t+W}` | end-effector trace over window `W` |
| `z_{t:t+W}` | LAM latent action tokens for the window |

## Components

### VLM action backbone

- **Backbone**: Qwen3-VL-4B-Instruct.
- **Action head**: DiT-B flow-matching expert producing continuous action chunks of length `H`.
- **Inputs**: current image(s) and the language instruction.
- **Outputs**: a continuous action chunk plus the auxiliary semantic tokens described below.

### Trace head

The VLM emits a tokenized representation of the next trace segment `τ_{t:t+W}` as plain language tokens, interleaved with the language instruction. The trace covers the projected end-effector (or object-of-interest) trajectory over a window of `W = 12` frames.

### Latent Action Model (LAM)

The LAM is trained **once**, offline, on OXE robot trajectories:

- A frozen DINOv2 ViT-B/14 encodes image patches.
- A trace encoder embeds `τ_{t:t+W}`.
- A small transformer fuses both streams and produces `K = 4` query tokens.
- A VQ codebook with `V = 32` entries quantizes each query.

At downstream VLA training time the LAM is **frozen**; we precompute its tokens for every training sample and ask the VLM to predict them as additional language-side targets.

### Co-training loss

Total loss:

```
L = L_action + λ_trace * L_trace_lm + λ_lam * L_lam_lm
```

The two language-modeling terms are cross-entropy losses over the trace and LAM token streams respectively. The released checkpoints use `λ_trace ≈ λ_lam`, with the combined LM loss weight balanced against the action loss at roughly `0.10`.

## Default training recipe

The released checkpoints supervise both auxiliary heads entirely in the VLM's **language stream**:

- Trace tokens are emitted as language tokens.
- Latent action tokens are emitted as language tokens.
- The action decoder is **unmodified** — it receives no extra semantic-embedding injection beyond the standard VLM hidden states.

This is the simplest possible co-training arrangement and is what we recommend as the default starting point for downstream work. Variants that additionally inject parsed trace embeddings into the action decoder were evaluated during development; on the unified OXE LAM, the language-stream-only recipe matches or exceeds them on the LIBERO mean, so we release it as the default.

## Released LAM: unified OXE

The released LAM checkpoint is the **unified OXE** variant — a single LAM jointly trained on three OXE datasets (BridgeData V2, Fractal/RT-1, BC-Z) under the `paper_strict` recipe. The same LAM is consumed by both the released LIBERO and SimplerEnv-Bridge policies; we do not release per-domain LAMs in the public release.

## Results summary

| Benchmark | Mean SR |
|---|---:|
| LIBERO (4 suites) | **0.982** |
| SimplerEnv WidowX (4 tasks) | **0.802** |

Per-suite / per-task numbers are in the model card READMEs and in the main project README.
