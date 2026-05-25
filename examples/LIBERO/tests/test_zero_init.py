"""Zero-init sanity for the SemanticVLA trace conditioning.

Two levels:
  1. TraceEncoder direct: output_proj last linear is zero-init, so for ANY
     input, forward returns all zeros (logical identity for downstream
     consumers).
  2. SemanticVLA_ActionHead integration: with `injection_mode='sa_embs'`,
     two forward passes with *different* random trace_coords but the same
     vl_embs / actions / state / Beta-sample seed must produce IDENTICAL
     loss. Same check for `injection_mode='adaln'` and `injection_mode='both'`.

These checks guarantee M3 introduces no behavior change relative to the
baseline at training step 0, which is the contractual property the rest of
the migration plan (M4 smoke train, M5 ablation) relies on.

Run:
    cd ${REPO_ROOT}
    PYTHONPATH=. python -m pytest examples/LIBERO/semanticvla/tests/test_zero_init.py -v
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from semanticvla.model.modules.action_model.SemanticVLA_ActionHeader import (
    SemanticVLA_ActionHead,
)
from semanticvla.model.modules.action_model.trace_encoder import TraceEncoder


# Use a small DiT-B head for fast CPU/GPU tests (still exercises the full
# DiT + trace_proj wiring on the AdaLN path).
def _make_config(injection_mode: str = "sa_embs"):
    # hidden_size must equal diffusion_model_cfg.output_dim so action_decoder
    # input dim matches the DiT projection output (mirrors the real yaml,
    # cf. cotrain_oxe.yaml: hidden_size=1024, output_dim=1024).
    HIDDEN = 768  # matches DiT-B input_embedding_dim for test
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": HIDDEN,
                    "action_dim": 7,
                    "state_dim": 8,
                    "future_action_window_size": 15,
                    "action_horizon": 16,
                    "past_action_window_size": 0,
                    "num_inference_timesteps": 4,
                    "num_timestep_buckets": 1000,
                    "num_target_vision_tokens": 32,
                    "max_seq_len": 1024,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "clamp_sample_time": False,
                    "add_pos_embed": True,
                    "diffusion_model_cfg": {
                        "num_layers": 4,  # smaller for test speed
                        "cross_attention_dim": 256,  # mock vl_embs hidden size
                        "output_dim": HIDDEN,  # must match hidden_size
                        "trace_dim": HIDDEN if injection_mode in {"adaln", "both"} else 0,
                    },
                    "trace": {
                        "injection_mode": injection_mode,
                        "hidden_dim": 128,
                        "num_layers": 2,
                        "num_heads": 4,
                        "window_size": 12,
                        "num_tokens": 4,
                        "dropout": 0.0,
                    },
                }
            }
        }
    )


# ---------------- Level 1: TraceEncoder zero-init ----------------


def test_trace_encoder_zero_init_all_zero_for_arbitrary_input():
    enc = TraceEncoder(
        input_dim=2,
        hidden_dim=64,
        output_dim=128,
        num_layers=2,
        num_heads=4,
        max_trace_len=12,
        output_num_tokens=4,
        dropout=0.0,
    )
    enc.eval()
    torch.manual_seed(0)
    coords = torch.randn(3, 12, 2)
    out = enc(coords)
    assert out.shape == (3, 4, 128)
    # last linear is zero-init → output exactly 0 regardless of input
    assert torch.all(out == 0), f"max abs={out.abs().max().item()}"


def test_trace_encoder_pooled_zero_init_all_zero():
    enc = TraceEncoder(output_dim=128, max_trace_len=12, output_num_tokens=4, dropout=0.0)
    enc.eval()
    out = enc.pooled(torch.randn(2, 12, 2))
    assert out.shape == (2, 128)
    assert torch.all(out == 0)


def test_trace_encoder_can_disable_zero_init_for_pretrained_use():
    # When loading a pre-trained TraceEncoder, the constructor should not
    # zero out the trained weights.
    enc = TraceEncoder(output_dim=128, dropout=0.0, zero_init_output=False)
    enc.eval()
    out = enc(torch.randn(2, 12, 2))
    # extremely unlikely to be exactly zero with random init
    assert out.abs().max().item() > 0.0


# ---------------- Level 2: ActionHead with vs. without trace ----------------


def _seeded_forward(head, vl_embs, actions, state, trace_coords, seed):
    torch.manual_seed(seed)
    return head(vl_embs, actions, state, trace_coords_window=trace_coords).item()


@torch.no_grad()
def _mk_inputs(batch=2, seq_len=16, vl_dim=256, action_dim=7, state_dim=8, horizon=16, win=12, device="cpu"):
    torch.manual_seed(123)
    vl = torch.randn(batch, seq_len, vl_dim, device=device)
    actions = torch.randn(batch, horizon, action_dim, device=device)
    state = torch.randn(batch, 1, state_dim, device=device)
    trace1 = torch.rand(batch, win, 2, device=device)
    trace2 = torch.rand(batch, win, 2, device=device) * 0.5 + 0.1  # very different
    return vl, actions, state, trace1, trace2


def test_semanticvla_head_sa_embs_zero_init_invariant_to_trace():
    cfg = _make_config(injection_mode="sa_embs")
    head = SemanticVLA_ActionHead(cfg).eval()
    vl, actions, state, t1, t2 = _mk_inputs()

    # With injection_mode='sa_embs' the trace_tokens (zero-init) get prepended
    # to sa_embs as all-zero rows. The action loss should be invariant to
    # the specific trace inputs at construction time.
    l_with_t1 = _seeded_forward(head, vl, actions, state, t1, seed=7)
    l_with_t2 = _seeded_forward(head, vl, actions, state, t2, seed=7)
    assert l_with_t1 == l_with_t2, f"loss diff: {abs(l_with_t1 - l_with_t2):.3e}"


def test_semanticvla_head_adaln_zero_init_invariant_to_trace():
    cfg = _make_config(injection_mode="adaln")
    head = SemanticVLA_ActionHead(cfg).eval()
    vl, actions, state, t1, t2 = _mk_inputs()
    l1 = _seeded_forward(head, vl, actions, state, t1, seed=11)
    l2 = _seeded_forward(head, vl, actions, state, t2, seed=11)
    assert l1 == l2, f"loss diff: {abs(l1 - l2):.3e}"


def test_semanticvla_head_both_zero_init_invariant_to_trace():
    cfg = _make_config(injection_mode="both")
    head = SemanticVLA_ActionHead(cfg).eval()
    vl, actions, state, t1, t2 = _mk_inputs()
    l1 = _seeded_forward(head, vl, actions, state, t1, seed=13)
    l2 = _seeded_forward(head, vl, actions, state, t2, seed=13)
    assert l1 == l2, f"loss diff: {abs(l1 - l2):.3e}"


def test_semanticvla_head_none_mode_matches_no_trace_arg():
    cfg = _make_config(injection_mode="none")
    head = SemanticVLA_ActionHead(cfg).eval()
    vl, actions, state, t1, _ = _mk_inputs()
    # injection_mode='none' → trace input ignored, output must equal calling
    # without trace.
    l_with = _seeded_forward(head, vl, actions, state, t1, seed=17)
    l_no = _seeded_forward(head, vl, actions, state, None, seed=17)
    assert l_with == l_no


# ---------------- Level 2b: predict_action invariance ----------------


@torch.no_grad()
def test_semanticvla_head_predict_action_invariant_to_trace_at_init():
    cfg = _make_config(injection_mode="both")
    head = SemanticVLA_ActionHead(cfg).eval()
    vl, _, state, t1, t2 = _mk_inputs()

    a1 = head.predict_action(
        vl, state, do_sample=False, sample_seed=0, trace_coords_window=t1
    )
    a2 = head.predict_action(
        vl, state, do_sample=False, sample_seed=0, trace_coords_window=t2
    )
    assert torch.allclose(a1, a2, atol=0.0, rtol=0.0), (
        f"action diff: {(a1 - a2).abs().max().item():.3e}"
    )
