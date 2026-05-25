"""TraceEncoder for SemanticVLA on SemanticVLA.

Ported from UniVLA's `latent_action_model.genie.modules.lam_with_trace_notrace_recon_v3`
TraceEncoder, simplified to remove the LAM-internal positional-encoding
dependency (a standalone sinusoidal PE is bundled here).

Input:  trace_coords  (B, max_trace_len, 2)   — 2D pixel coords, typically [0,1] normalized
Output: tokens        (B, output_num_tokens, output_dim)

Design notes
------------
- Final `output_proj` is **zero-initialized** so an untrained encoder produces
  all-zero token contributions. This guarantees that at step 0 (before any
  trace-related gradient flow) the SemanticVLA model's action prediction is
  bit-identical to the baseline (sa_embs prepend = baseline + zeros = baseline).
- For the AdaLN injection path, `pooled()` returns mean-pooled tokens.
- `output_dim` is parameterized so the same module fits DiT-B (768) or
  DiT-L (1536) without modification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Standalone sinusoidal positional encoding, identical formula to the
    UniVLA `latent_action_model.genie.modules.blocks.PositionalEncoding`.
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TraceEncoder(nn.Module):
    """Encode a (B, N, 2) pixel-coord trace window into a fixed-size token
    sequence via Transformer + learnable-query pooling.

    Same architectural skeleton as UniVLA SemanticVLA v3 TraceEncoder; the
    final output projection is zero-initialized so an untrained encoder is
    a no-op for downstream consumers.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 1536,
        num_layers: int = 3,
        num_heads: int = 8,
        max_trace_len: int = 12,
        output_num_tokens: int = 4,
        dropout: float = 0.1,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_trace_len = max_trace_len
        self.output_num_tokens = output_num_tokens

        # 1) Per-step coord embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2) Positional encoding over the time axis
        self.pos_enc = SinusoidalPE(hidden_dim, max_len=max_trace_len)

        # 3) Transformer encoder over the N=max_trace_len positions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Learnable queries + cross-attention pooling → fixed output_num_tokens
        self.trace_queries = nn.Parameter(
            torch.randn(1, output_num_tokens, hidden_dim) * 0.02
        )
        self.pooling_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 5) Output projection (zero-init last linear so the whole encoder
        #    contributes 0 to the downstream sequence/temb at init).
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if zero_init_output:
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        trace_coords: torch.Tensor,
        trace_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode trace coords into a fixed-length token sequence.

        Args:
            trace_coords: (B, N, 2). N must be <= max_trace_len.
            trace_mask:   (B, N) boolean. True == valid, False == padding.
                          If None, all positions are valid.

        Returns:
            (B, output_num_tokens, output_dim) float tensor.
        """
        if trace_coords.dim() != 3 or trace_coords.size(-1) != self.input_dim:
            raise ValueError(
                f"TraceEncoder expects (B, N, {self.input_dim}); got {tuple(trace_coords.shape)}"
            )
        B, N, _ = trace_coords.shape
        if N > self.max_trace_len:
            raise ValueError(
                f"trace length {N} exceeds max_trace_len {self.max_trace_len}"
            )

        # (B, N, hidden_dim)
        x = self.input_proj(trace_coords)
        x = self.pos_enc(x)

        # Transformer expects (B, N, D). For padding mask: True = ignore.
        if trace_mask is not None:
            # nn.Transformer uses src_key_padding_mask where True = padding
            src_pad_mask = ~trace_mask.bool()
        else:
            src_pad_mask = None
        x = self.transformer(x, src_key_padding_mask=src_pad_mask)

        # Pool to fixed number of tokens via cross-attention from learnable
        # queries to the trace sequence.
        queries = self.trace_queries.expand(B, -1, -1)  # (B, Q, hidden)
        pooled, _ = self.pooling_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=src_pad_mask,
            need_weights=False,
        )  # (B, Q, hidden)

        out = self.output_proj(pooled)  # (B, Q, output_dim)
        return out

    def pooled(
        self,
        trace_coords: torch.Tensor,
        trace_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single pooled vector per sample for the AdaLN injection path.

        Returns (B, output_dim) — mean over the output tokens.
        """
        tokens = self.forward(trace_coords, trace_mask)
        return tokens.mean(dim=1)
