from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINO_DIMS = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
}

_DINO_BUILDERS = {
    "vits14": "dinov2_vits14",
    "vitb14": "dinov2_vitb14",
    "vitl14": "dinov2_vitl14",
    "vitg14": "dinov2_vitg14",
}


def _normalize_dino_variant(variant: str) -> str:
    variant = str(variant).strip()
    if variant.startswith("dinov2_"):
        variant = variant[len("dinov2_") :]
    if variant not in _DINO_DIMS:
        raise ValueError(f"Unsupported DINOv2 variant: {variant}")
    return variant


def _load_frozen_dinov2(repo_root: str, weights_path: str, variant: str):
    repo_root = os.path.expanduser(repo_root)
    weights_path = os.path.expanduser(weights_path)
    variant = _normalize_dino_variant(variant)
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"DINOv2 repo_root not found: {repo_root}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from dinov2.hub import backbones as dinov2_backbones  # type: ignore

    model = getattr(dinov2_backbones, _DINO_BUILDERS[variant])(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TraceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 8,
        max_trace_len: int = 12,
        output_num_tokens: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = SinusoidalPositionEncoding(hidden_dim, max_len=max_trace_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(ff_dim or hidden_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.queries = nn.Parameter(torch.randn(1, output_num_tokens, hidden_dim) * 0.02)
        self.pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, trace_coords: torch.Tensor, trace_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(trace_coords)
        x = self.pos_enc(x)
        padding_mask = None
        if trace_mask is not None:
            padding_mask = ~trace_mask.bool()
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        queries = self.queries.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.out(pooled)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int):
        super().__init__()
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.codebook = nn.Embedding(self.num_latents, self.latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_latents, 1.0 / self.num_latents)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(x.float(), self.codebook.weight.float())
        indices = torch.argmin(distances, dim=-1)
        z = self.codebook(indices).to(dtype=x.dtype)
        z_q = x + (z - x).detach()
        return z_q, z, x, indices


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        ff_dim: int | None = None,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionEncoding(dim, max_len=max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(ff_dim or dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.pos_enc(x)))


@dataclass(frozen=True)
class LAMVariant:
    use_trace: bool
    use_trace_in_decoder: bool
    num_action_tokens: int
    num_latents: int
    two_stage: bool = False


def build_lam_variant_config(variant: str) -> LAMVariant:
    """Return the LAM configuration. Only the released `paper_strict` variant is exposed."""
    variant = str(variant)
    variants = {
        "paper_strict": LAMVariant(True, False, 4, 32, True),
    }
    if variant not in variants:
        valid = ", ".join(sorted(variants))
        raise ValueError(f"Unknown LAM variant {variant!r}; valid: {valid}")
    return variants[variant]


class TraceLatentActionModel(nn.Module):
    """Trace-guided VQ latent action tokenizer for SemanticVLA LIBERO experiments.

    The model follows the UniVLA trace-LAM idea, but uses a dependency-light
    Transformer stack and local DINOv2 weights. It tokenizes motion between
    a historical image pair, normally `(t-11, t)`.
    """

    def __init__(
        self,
        *,
        dino_repo_root: str,
        dino_weights: str,
        dino_variant: str = "vits14",
        model_dim: int | None = None,
        enc_layers: int = 4,
        dec_layers: int = 4,
        num_heads: int = 6,
        num_action_tokens: int = 4,
        num_latents: int = 16,
        latent_dim: int = 128,
        use_trace: bool = True,
        use_trace_in_decoder: bool = False,
        trace_hidden_dim: int = 256,
        trace_layers: int = 2,
        trace_num_heads: int = 8,
        trace_tokens: int = 4,
        max_trace_len: int = 12,
        vq_beta: float = 0.25,
        dropout: float = 0.0,
        mock_dino: bool = False,
    ):
        super().__init__()
        dino_variant = _normalize_dino_variant(dino_variant)
        self.dino_dim = _DINO_DIMS[dino_variant]
        self.model_dim = int(model_dim or self.dino_dim)
        self.num_action_tokens = int(num_action_tokens)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.use_trace = bool(use_trace)
        self.use_trace_in_decoder = bool(use_trace_in_decoder)
        self.trace_tokens = int(trace_tokens)
        self.vq_beta = float(vq_beta)
        self.mock_dino = bool(mock_dino)

        if self.mock_dino:
            self.dino_encoder = None
        else:
            self.dino_encoder = _load_frozen_dinov2(dino_repo_root, dino_weights, dino_variant)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self.patch_proj = nn.Identity() if self.dino_dim == self.model_dim else nn.Linear(self.dino_dim, self.model_dim)
        self.patch_out = nn.Identity() if self.dino_dim == self.model_dim else nn.Linear(self.model_dim, self.dino_dim)

        if self.use_trace or self.use_trace_in_decoder:
            self.trace_encoder = TraceEncoder(
                hidden_dim=trace_hidden_dim,
                output_dim=self.model_dim,
                num_layers=trace_layers,
                output_num_tokens=self.trace_tokens,
                max_trace_len=max_trace_len,
                dropout=dropout,
            )
        else:
            self.trace_encoder = None

        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_action_tokens, self.model_dim))
        nn.init.uniform_(self.action_latent, a=-1.0, b=1.0)

        self.encoder = SequenceTransformer(self.model_dim, enc_layers, num_heads, dropout=dropout)
        self.to_codebook = nn.Linear(self.model_dim, self.latent_dim)
        self.vq = VectorQuantizer(self.num_latents, self.latent_dim)
        self.action_up = nn.Linear(self.latent_dim, self.model_dim)
        self.decoder = SequenceTransformer(self.model_dim, dec_layers, num_heads, dropout=dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_dino_patches(self, videos: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        if self.mock_dino:
            return torch.randn(B, T, 256, self.dino_dim, device=videos.device, dtype=videos.dtype)
        x = (videos - self.image_mean.to(videos)) / self.image_std.to(videos)
        x = x.reshape(B * T, C, H, W)
        with torch.no_grad():
            features = self.dino_encoder.forward_features(x)["x_norm_patchtokens"]
        features = features.detach()
        return features.reshape(B, T, features.shape[1], features.shape[2]).to(dtype=videos.dtype)

    def _trace_tokens(self, batch: dict[str, torch.Tensor], T: int) -> torch.Tensor | None:
        if self.trace_encoder is None:
            return None
        trace_mask = batch.get("trace_mask")
        trace_tokens = self.trace_encoder(batch["traces"], trace_mask)
        return trace_tokens.unsqueeze(1).expand(-1, T, -1, -1)

    def vq_encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        videos = batch["videos"]
        B, T = videos.shape[:2]
        dino_patches = self.encode_dino_patches(videos)
        patches = self.patch_proj(dino_patches)
        action_tokens = self.action_latent.expand(B, T, -1, -1)
        tokens = [action_tokens, patches]
        trace_tokens = self._trace_tokens(batch, T)
        if self.use_trace and trace_tokens is not None:
            tokens.append(trace_tokens)
        per_timestep = torch.cat(tokens, dim=2)
        S = per_timestep.shape[2]
        encoded = self.encoder(per_timestep.reshape(B, T * S, self.model_dim)).reshape(B, T, S, self.model_dim)
        z_e = self.to_codebook(encoded[:, 1:, : self.num_action_tokens])
        flat = z_e.reshape(B * (T - 1), self.num_action_tokens, self.latent_dim)
        z_q, z, emb, indices = self.vq(flat)
        return {
            "patches": dino_patches,
            "patches_model": patches,
            "trace_tokens": trace_tokens,
            "z_q": z_q.reshape(B, T - 1, self.num_action_tokens, self.latent_dim),
            "z": z,
            "emb": emb,
            "indices": indices.reshape(B, T - 1, self.num_action_tokens),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.vq_encode(batch)
        patches_model = outputs["patches_model"]
        B, T = patches_model.shape[:2]
        past_patches = patches_model[:, :-1]
        action_patches = self.action_up(outputs["z_q"])
        dec_tokens = [action_patches, past_patches]
        if self.use_trace_in_decoder and outputs["trace_tokens"] is not None:
            dec_tokens.append(outputs["trace_tokens"][:, 1:])
        per_timestep = torch.cat(dec_tokens, dim=2)
        S = per_timestep.shape[2]
        decoded = self.decoder(per_timestep.reshape(B, (T - 1) * S, self.model_dim))
        decoded = decoded.reshape(B, T - 1, S, self.model_dim)
        recon_model = decoded[:, :, self.num_action_tokens : self.num_action_tokens + past_patches.shape[2]]
        recon = self.patch_out(recon_model)
        outputs.update({"recon": recon, "target": outputs["patches"][:, 1:]})
        return outputs

    def compute_loss(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon_loss = F.mse_loss(outputs["recon"].float(), outputs["target"].float())
        codebook_loss = ((outputs["emb"].detach().float() - outputs["z"].float()) ** 2).mean()
        commit_loss = ((outputs["emb"].float() - outputs["z"].detach().float()) ** 2).mean()
        loss = recon_loss + codebook_loss + self.vq_beta * commit_loss

        indices = outputs["indices"].reshape(-1)
        counts = torch.bincount(indices, minlength=self.num_latents).float()
        used = counts > 0
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs[used] * torch.log(probs[used].clamp_min(1e-12))).sum()
        perplexity = torch.exp(entropy)
        metrics = {
            "loss": loss.detach(),
            "recon_loss": recon_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commit_loss": commit_loss.detach(),
            "code_usage": used.float().mean().detach(),
            "code_entropy": entropy.detach(),
            "code_perplexity": perplexity.detach(),
        }
        return loss, metrics

    @classmethod
    def from_config(cls, cfg: dict[str, Any], variant: str):
        variant_cfg = build_lam_variant_config(variant)
        model_cfg = dict(cfg)
        if variant_cfg.two_stage:
            model_cfg.update(
                num_action_tokens=variant_cfg.num_action_tokens,
                num_latents=variant_cfg.num_latents,
            )
            return TraceTwoStageLatentActionModel(**model_cfg)
        model_cfg.update(
            use_trace=variant_cfg.use_trace,
            use_trace_in_decoder=variant_cfg.use_trace_in_decoder,
            num_action_tokens=variant_cfg.num_action_tokens,
            num_latents=variant_cfg.num_latents,
        )
        return cls(**model_cfg)


class TraceSequenceDecoder(nn.Module):
    """Decode a small set of latent tokens back to a fixed trace sequence."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        num_tokens: int = 4,
        max_trace_len: int = 12,
        num_layers: int = 0,
        num_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.upsample = nn.Linear(num_tokens, max_trace_len)
        self.pos = nn.Parameter(torch.randn(1, max_trace_len, hidden_dim) * 0.02)
        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(ff_dim or hidden_dim * 4),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.decoder = None
            self.norm = nn.Identity()
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 2),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens)
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos.to(dtype=x.dtype, device=x.device)
        if self.decoder is not None:
            x = self.norm(self.decoder(x))
        return self.out(x)


class TraceTwoStageLatentActionModel(nn.Module):
    """Two-stage trace-guided latent action tokenizer.

    This is a SemanticVLA-native approximation of the SemanticVLA tokenizer:

    Stage 1 learns a pure geometric trace codebook from normalized 2D traces.
    Stage 2 uses quantized trace codebook entries as geometric queries over
    frozen DINOv2 features, then quantizes the fused result into action codes.
    Both decoders are training-only; downstream VLM training uses `indices`.
    """

    def __init__(
        self,
        *,
        dino_repo_root: str,
        dino_weights: str,
        dino_variant: str = "vits14",
        model_dim: int | None = None,
        enc_layers: int = 4,
        dec_layers: int = 4,
        num_heads: int = 6,
        num_action_tokens: int = 4,
        num_latents: int = 32,
        latent_dim: int = 128,
        trace_hidden_dim: int = 256,
        trace_layers: int = 2,
        trace_num_heads: int = 8,
        trace_ff_dim: int | None = None,
        trace_tokens: int = 4,
        max_trace_len: int = 12,
        trace_num_latents: int = 32,
        trace_decoder_hidden_dim: int = 256,
        trace_decoder_layers: int = 0,
        trace_decoder_heads: int = 8,
        trace_decoder_ff_dim: int | None = None,
        fusion_layers: int = 4,
        fusion_heads: int | None = None,
        fusion_ff_dim: int | None = None,
        decoder_ff_dim: int | None = None,
        action_trace_decoder_layers: int = 0,
        action_trace_decoder_heads: int = 8,
        action_trace_decoder_ff_dim: int | None = None,
        stage1_warmup_steps: int = 5000,
        freeze_trace_stage_after_warmup: bool = True,
        stage1_loss_weight: float = 1.0,
        trace_recon_loss_weight: float = 1.0,
        visual_recon_loss_weight: float = 1.0,
        action_trace_recon_loss_weight: float | None = None,
        vq_beta: float = 0.25,
        dropout: float = 0.0,
        mock_dino: bool = False,
        **_: Any,
    ):
        super().__init__()
        dino_variant = _normalize_dino_variant(dino_variant)
        self.dino_dim = _DINO_DIMS[dino_variant]
        self.model_dim = int(model_dim or self.dino_dim)
        self.num_action_tokens = int(num_action_tokens)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.trace_tokens = int(trace_tokens)
        self.trace_num_latents = int(trace_num_latents)
        self.max_trace_len = int(max_trace_len)
        self.vq_beta = float(vq_beta)
        self.stage1_warmup_steps = int(stage1_warmup_steps)
        self.stage1_loss_weight = float(stage1_loss_weight)
        self.trace_recon_loss_weight = float(trace_recon_loss_weight)
        self.visual_recon_loss_weight = float(visual_recon_loss_weight)
        if action_trace_recon_loss_weight is None:
            action_trace_recon_loss_weight = trace_recon_loss_weight
        self.action_trace_recon_loss_weight = float(action_trace_recon_loss_weight)
        self.freeze_trace_stage_after_warmup = bool(freeze_trace_stage_after_warmup)
        self.current_step = 0
        self.mock_dino = bool(mock_dino)

        if self.mock_dino:
            self.dino_encoder = None
        else:
            self.dino_encoder = _load_frozen_dinov2(dino_repo_root, dino_weights, dino_variant)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self.patch_proj = nn.Identity() if self.dino_dim == self.model_dim else nn.Linear(self.dino_dim, self.model_dim)
        self.patch_out = nn.Identity() if self.dino_dim == self.model_dim else nn.Linear(self.model_dim, self.dino_dim)

        self.trace_encoder = TraceEncoder(
            hidden_dim=trace_hidden_dim,
            output_dim=self.model_dim,
            num_layers=trace_layers,
            num_heads=trace_num_heads,
            ff_dim=trace_ff_dim,
            output_num_tokens=self.trace_tokens,
            max_trace_len=self.max_trace_len,
            dropout=dropout,
        )
        self.trace_vq = VectorQuantizer(self.trace_num_latents, self.model_dim)
        self.trace_decoder = TraceSequenceDecoder(
            input_dim=self.model_dim,
            hidden_dim=trace_decoder_hidden_dim,
            num_tokens=self.trace_tokens,
            max_trace_len=self.max_trace_len,
            num_layers=trace_decoder_layers,
            num_heads=trace_decoder_heads,
            ff_dim=trace_decoder_ff_dim,
            dropout=dropout,
        )

        fusion_heads = int(fusion_heads or num_heads)
        self.visual_norm = nn.LayerNorm(self.model_dim)
        self.trace_norm = nn.LayerNorm(self.model_dim)
        self.trace_visual_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=fusion_heads,
            dropout=dropout,
            batch_first=True,
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=fusion_heads,
            dim_feedforward=int(fusion_ff_dim or self.model_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.to_action_codebook = nn.Linear(self.model_dim, self.latent_dim)
        self.action_vq = VectorQuantizer(self.num_latents, self.latent_dim)

        self.action_up = nn.Linear(self.latent_dim, self.model_dim)
        self.decoder = SequenceTransformer(self.model_dim, dec_layers, num_heads, dropout=dropout, ff_dim=decoder_ff_dim)
        self.action_trace_decoder = TraceSequenceDecoder(
            input_dim=self.latent_dim,
            hidden_dim=trace_decoder_hidden_dim,
            num_tokens=self.num_action_tokens,
            max_trace_len=self.max_trace_len,
            num_layers=action_trace_decoder_layers,
            num_heads=action_trace_decoder_heads,
            ff_dim=action_trace_decoder_ff_dim,
            dropout=dropout,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_train_step(self, step: int) -> None:
        self.current_step = int(step)

    def _warmup_active(self) -> bool:
        return 0 < self.current_step <= self.stage1_warmup_steps

    def _freeze_trace_stage(self) -> bool:
        return self.freeze_trace_stage_after_warmup and not self._warmup_active()

    def encode_dino_patches(self, videos: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        if self.mock_dino:
            return torch.randn(B, T, 256, self.dino_dim, device=videos.device, dtype=videos.dtype)
        x = (videos - self.image_mean.to(videos)) / self.image_std.to(videos)
        x = x.reshape(B * T, C, H, W)
        with torch.no_grad():
            features = self.dino_encoder.forward_features(x)["x_norm_patchtokens"]
        return features.detach().reshape(B, T, features.shape[1], features.shape[2]).to(dtype=videos.dtype)

    def vq_encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        videos = batch["videos"]
        B, T = videos.shape[:2]

        trace_mask = batch.get("trace_mask")
        freeze_trace = self._freeze_trace_stage()
        with torch.set_grad_enabled(not freeze_trace):
            trace_z_e = self.trace_encoder(batch["traces"], trace_mask)
            trace_z_q, trace_z, trace_emb, trace_indices = self.trace_vq(trace_z_e)
            trace_recon = self.trace_decoder(trace_z_q)

        dino_patches = self.encode_dino_patches(videos)
        patches = self.patch_proj(dino_patches)
        visual_context = patches.reshape(B, T * patches.shape[2], self.model_dim)
        visual_context = self.visual_norm(visual_context)

        trace_queries = self.trace_norm(trace_z_q)
        attended, _ = self.trace_visual_attn(
            query=trace_queries,
            key=visual_context,
            value=visual_context,
            need_weights=False,
        )
        fused = self.fusion_encoder(trace_z_q + attended)
        action_z_e = self.to_action_codebook(fused[:, : self.num_action_tokens])
        action_z_q, action_z, action_emb, action_indices = self.action_vq(action_z_e)

        return {
            "patches": dino_patches,
            "patches_model": patches,
            "trace_z_q": trace_z_q,
            "trace_z": trace_z,
            "trace_emb": trace_emb,
            "trace_indices": trace_indices,
            "trace_recon": trace_recon,
            "action_z_q": action_z_q,
            "z_q": action_z_q.unsqueeze(1),
            "z": action_z,
            "emb": action_emb,
            "indices": action_indices.unsqueeze(1),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.vq_encode(batch)
        patches_model = outputs["patches_model"]
        past_patches = patches_model[:, :-1]
        action_patches = self.action_up(outputs["action_z_q"]).unsqueeze(1)
        per_timestep = torch.cat([action_patches, past_patches], dim=2)
        B, Tm1, S = per_timestep.shape[:3]
        decoded = self.decoder(per_timestep.reshape(B, Tm1 * S, self.model_dim))
        decoded = decoded.reshape(B, Tm1, S, self.model_dim)
        recon_model = decoded[:, :, self.num_action_tokens : self.num_action_tokens + past_patches.shape[2]]
        recon = self.patch_out(recon_model)
        outputs.update(
            {
                "recon": recon,
                "target": outputs["patches"][:, 1:],
                "trace_target": batch["traces"],
                "action_trace_recon": self.action_trace_decoder(outputs["action_z_q"]),
            }
        )
        return outputs

    def compute_loss(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stage1_trace_recon = F.mse_loss(outputs["trace_recon"].float(), outputs["trace_target"].float())
        trace_codebook = ((outputs["trace_emb"].detach().float() - outputs["trace_z"].float()) ** 2).mean()
        trace_commit = ((outputs["trace_emb"].float() - outputs["trace_z"].detach().float()) ** 2).mean()
        stage1_loss = stage1_trace_recon + trace_codebook + self.vq_beta * trace_commit

        visual_recon = F.mse_loss(outputs["recon"].float(), outputs["target"].float())
        action_trace_recon = F.mse_loss(outputs["action_trace_recon"].float(), outputs["trace_target"].float())
        action_codebook = ((outputs["emb"].detach().float() - outputs["z"].float()) ** 2).mean()
        action_commit = ((outputs["emb"].float() - outputs["z"].detach().float()) ** 2).mean()
        action_vq = action_codebook + self.vq_beta * action_commit
        stage2_loss = (
            self.visual_recon_loss_weight * visual_recon
            + self.action_trace_recon_loss_weight * action_trace_recon
            + action_vq
        )

        warmup_active = self._warmup_active()
        if warmup_active:
            loss = stage1_loss
        elif self.freeze_trace_stage_after_warmup:
            loss = stage2_loss
        else:
            loss = stage2_loss + self.stage1_loss_weight * stage1_loss

        indices = outputs["indices"].reshape(-1)
        counts = torch.bincount(indices, minlength=self.num_latents).float()
        used = counts > 0
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs[used] * torch.log(probs[used].clamp_min(1e-12))).sum()
        perplexity = torch.exp(entropy)

        trace_counts = torch.bincount(outputs["trace_indices"].reshape(-1), minlength=self.trace_num_latents).float()
        trace_used = trace_counts > 0
        trace_probs = trace_counts / trace_counts.sum().clamp_min(1.0)
        trace_entropy = -(trace_probs[trace_used] * torch.log(trace_probs[trace_used].clamp_min(1e-12))).sum()

        metrics = {
            "loss": loss.detach(),
            "recon_loss": visual_recon.detach(),
            "codebook_loss": action_codebook.detach(),
            "commit_loss": action_commit.detach(),
            "code_usage": used.float().mean().detach(),
            "code_entropy": entropy.detach(),
            "code_perplexity": perplexity.detach(),
            "stage1_loss": stage1_loss.detach(),
            "stage1_trace_recon_loss": stage1_trace_recon.detach(),
            "trace_codebook_loss": trace_codebook.detach(),
            "trace_commit_loss": trace_commit.detach(),
            "trace_code_usage": trace_used.float().mean().detach(),
            "trace_code_entropy": trace_entropy.detach(),
            "trace_code_perplexity": torch.exp(trace_entropy).detach(),
            "action_trace_recon_loss": action_trace_recon.detach(),
            "stage2_loss": stage2_loss.detach(),
            "warmup_active": torch.tensor(float(warmup_active), device=loss.device),
            "trace_stage_frozen": torch.tensor(float(self._freeze_trace_stage()), device=loss.device),
        }
        return loss, metrics
