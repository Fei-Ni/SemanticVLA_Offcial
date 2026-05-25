"""SemanticVLA action head: FlowmatchingActionHead + trace conditioning.

Wraps the baseline `FlowmatchingActionHead` with optional trace-guided
conditioning. Two injection modes (selectable per config):

  β  "sa_embs":   prepend `num_trace_tokens` trace tokens into the sa_embs
                  sequence next to state / future / action features.
  γ  "adaln":     pool the trace tokens to a single vector and inject into
                  DiT's AdaLN `temb` channel (via the optional `trace_proj`
                  added in cross_attention_dit.py).
  "both":         apply both paths.
  "none":         passthrough behavior — identical to FlowmatchingActionHead.

Zero-init contract: `TraceEncoder.output_proj` last linear + DiT's
`trace_proj` are both zero-initialized at construction. So at step 0 the
trace contribution is exactly 0 and forward()/predict_action() outputs match
the baseline bit-by-bit. Verified by `test_zero_init.py`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from semanticvla.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
)
from semanticvla.model.modules.action_model.trace_encoder import TraceEncoder


_VALID_MODES = {"none", "sa_embs", "adaln", "both"}


class SemanticVLA_ActionHead(FlowmatchingActionHead):
    """FlowmatchingActionHead + trace conditioning (paths β / γ)."""

    def __init__(self, full_config):
        super().__init__(full_config)

        trace_cfg = getattr(full_config.framework.action_model, "trace", None)
        if trace_cfg is None:
            # No trace config → behave exactly like the baseline.
            self.injection_mode: str = "none"
            self.trace_encoder: Optional[TraceEncoder] = None
            self.trace_dim: int = 0
            self.num_trace_tokens: int = 0
            return

        mode = str(trace_cfg.get("injection_mode", "sa_embs")).lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"framework.action_model.trace.injection_mode must be one of "
                f"{sorted(_VALID_MODES)}; got '{mode}'"
            )
        self.injection_mode = mode

        if mode == "none":
            self.trace_encoder = None
            self.trace_dim = 0
            self.num_trace_tokens = 0
            return

        hidden_dim = int(trace_cfg.get("hidden_dim", 256))
        num_layers = int(trace_cfg.get("num_layers", 3))
        num_heads = int(trace_cfg.get("num_heads", 8))
        max_trace_len = int(trace_cfg.get("window_size", 12))
        num_tokens = int(trace_cfg.get("num_tokens", 4))
        dropout = float(trace_cfg.get("dropout", 0.1))

        # Output dim must match the DiT's input embedding dim because trace
        # tokens are prepended to sa_embs which is fed into the DiT
        # transformer blocks at input_embedding_dim wide.
        self.trace_dim = int(self.input_embedding_dim)
        self.num_trace_tokens = num_tokens

        self.trace_encoder = TraceEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=self.trace_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_trace_len=max_trace_len,
            output_num_tokens=num_tokens,
            dropout=dropout,
            zero_init_output=True,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _encode_trace(
        self, trace_coords_window: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (trace_tokens, trace_pooled) or (None, None) if trace is off.

        - trace_tokens: (B, num_trace_tokens, input_embedding_dim) for β.
        - trace_pooled: (B, input_embedding_dim) for γ.
        """
        if (
            self.trace_encoder is None
            or self.injection_mode == "none"
            or trace_coords_window is None
        ):
            return None, None
        tokens = self.trace_encoder(trace_coords_window)
        need_β = self.injection_mode in {"sa_embs", "both"}
        need_γ = self.injection_mode in {"adaln", "both"}
        return (tokens if need_β else None), (tokens.mean(dim=1) if need_γ else None)

    def _build_sa_embs(
        self,
        state_features: Optional[torch.Tensor],
        future_tokens: torch.Tensor,
        action_features: torch.Tensor,
        trace_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mirror FlowmatchingActionHead's sa_embs construction, inserting
        trace_tokens right after state_features (β path) when present.

        Order (matches GR00T_ActionHeader:308 baseline plus β):
            [state, trace, future, action]
        """
        parts = []
        if state_features is not None:
            parts.append(state_features)
        if trace_tokens is not None:
            parts.append(trace_tokens.to(action_features.dtype))
        parts.append(future_tokens)
        parts.append(action_features)
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # forward / predict_action — override with trace conditioning
    # ------------------------------------------------------------------

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        trace_coords_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = vl_embs.device

        # Flow-matching noise sampling — identical to baseline.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

        trace_tokens, trace_pooled = self._encode_trace(trace_coords_window)
        sa_embs = self._build_sa_embs(state_features, future_tokens, action_features, trace_tokens)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,
            trace_emb=trace_pooled,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        do_sample: bool = True,
        sample_seed: Optional[int] = None,
        trace_coords_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        generator = None
        if sample_seed is not None or not do_sample:
            seed = 0 if sample_seed is None else int(sample_seed)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
            generator=generator,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Trace conditioning is constant across denoising steps — encode once.
        trace_tokens, trace_pooled = self._encode_trace(trace_coords_window)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = self._build_sa_embs(state_features, future_tokens, action_features, trace_tokens)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                trace_emb=trace_pooled,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        return actions


def get_semanticvla_action_model(config=None):
    """Factory parallel to GR00T_ActionHeader.get_action_model."""
    return SemanticVLA_ActionHead(full_config=config)
