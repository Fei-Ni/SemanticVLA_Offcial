"""SemanticVLA framework — trace-guided VLA built on Qwen-VL + Flow-Matching head.

Inherits directly from `baseframework`
to keep the SemanticVLA forward / predict_action paths fully independent of
the other framework subclasses (in particular, this never imports from or
depends on other framework variants).

Inputs flowing in from the dataloader (one dict per batch item):
    image                 List[PIL.Image]   — multi-view RGB observations
    lang                  str               — task instruction
    action                np.ndarray [T, D] — future action target
    state                 np.ndarray [1, D] — optional proprioceptive state
    trace_coords_window   np.ndarray [W, 2] — optional, attached by
                                              TraceAugmentedDataset / LiberoTraceManager

When `trace_coords_window` is absent (no trace data, mixture sibling that is
not LIBERO, or trace explicitly disabled), the framework runs identically to
the baseline Qwen_GR00T.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from semanticvla.model.framework.base_framework import baseframework
from semanticvla.model.modules.action_model.GR00T_ActionHeader import DiTConfig
from semanticvla.model.modules.action_model.SemanticVLA_ActionHeader import (
    SemanticVLA_ActionHead,
)
from semanticvla.model.modules.action_model.semantic_text_codec import (
    format_semantic_batch,
    lam_tokens,
    parse_trace_from_semantic_text,
    semantic_prompt_template,
)
from semanticvla.model.modules.action_model.trace_text_codec import (
    format_trace_batch,
    parse_trace_batch,
    prompt_template,
)
from semanticvla.model.modules.vlm import get_vlm_model
from semanticvla.model.tools import FRAMEWORK_REGISTRY
from semanticvla.training.trainer_utils import initialize_overwatch
from semanticvla.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("SemanticVLA")
class SemanticVLA(baseframework):
    """Qwen-VL + Flow-Matching head with optional trace-guided conditioning.

    Two injection paths gated by `framework.action_model.trace.injection_mode`:
        - "sa_embs":  trace tokens are prepended into the DiT sa_embs sequence
        - "adaln":    pooled trace vector is injected into DiT's AdaLN temb
        - "both":     both of the above
        - "none":     baseline behavior (no trace use)
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # 1) VLM backbone (Qwen-VL).
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        # 2) If γ / both is requested, tell the DiT how wide the trace_proj
        #    should be (matches input_embedding_dim of the DiT variant).
        trace_cfg = getattr(self.config.framework.action_model, "trace", None)
        injection_mode = "none"
        if trace_cfg is not None:
            injection_mode = str(trace_cfg.get("injection_mode", "sa_embs")).lower()
        if injection_mode in {"adaln", "both"}:
            action_model_type = self.config.framework.action_model.action_model_type
            input_emb_dim = DiTConfig[action_model_type]["input_embedding_dim"]
            self.config.framework.action_model.diffusion_model_cfg.trace_dim = (
                input_emb_dim
            )

        # 3) Action head with trace conditioning baked in.
        self.action_model: SemanticVLA_ActionHead = SemanticVLA_ActionHead(
            full_config=self.config
        )

        # 4) Cache useful constants for forward.
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = (
            self.past_action_window_size + 1 + self.future_action_window_size
        )
        self.injection_mode = injection_mode

        # 5) the LM-trace path settings (mode A / B).
        #    Gated by `framework.action_model.trace.lm_aux_loss`; when False
        #    the framework behaves like v0 (uses GT trace as decoder input).
        if trace_cfg is not None:
            self.lm_aux_loss = bool(trace_cfg.get("lm_aux_loss", False))
            self.aux_loss_weight = float(trace_cfg.get("aux_loss_weight", 0.1))
            self.prompt_style = str(trace_cfg.get("prompt_style", "plain"))
            self.num_anchor_points = int(trace_cfg.get("num_anchor_points", 0))  # 0 → not anchor-mode (v0)
            self.coord_range = int(trace_cfg.get("coord_range", 1000))
        else:
            self.lm_aux_loss = False
            self.aux_loss_weight = 0.0
            self.prompt_style = "plain"
            self.num_anchor_points = 0
            self.coord_range = 1000

        # semantic-output path. This generalizes the LM-trace
        # to support latent-action special tokens and trace+latent targets.
        semantic_cfg = getattr(self.config.framework.action_model, "semantic_output", None)
        self.semantic_output_enabled = (
            semantic_cfg is not None and bool(semantic_cfg.get("enabled", False))
        )
        self.semantic_output_mode = "trace_only"
        self.semantic_output_order = "trace_latent"
        self.semantic_lm_loss_weight = self.aux_loss_weight
        self.semantic_prompt_style = self.prompt_style
        self.semantic_trace_anchor_points = max(self.num_anchor_points, 1)
        self.semantic_latent_vocab_size = 0
        self.semantic_latent_num_tokens = 0
        self.semantic_latent_prefix = "LAM"
        self.semantic_parse_trace_for_decoder = False
        self.semantic_trainable_token_rows = False
        self.semantic_preload_tokens_before_checkpoint = False
        self.semantic_lam_token_ids: list[int] = []
        self._semantic_tokens_initialized = False
        if self.semantic_output_enabled:
            self.semantic_output_mode = str(semantic_cfg.get("mode", "trace_latent")).lower()
            self.semantic_output_order = str(semantic_cfg.get("order", self.semantic_output_mode)).lower()
            self.semantic_lm_loss_weight = float(semantic_cfg.get("lm_loss_weight", 0.1))
            self.semantic_prompt_style = str(semantic_cfg.get("prompt_style", self.prompt_style))
            self.semantic_trace_anchor_points = int(
                semantic_cfg.get("trace_anchor_points", max(self.num_anchor_points, 1))
            )
            self.semantic_latent_vocab_size = int(semantic_cfg.get("latent_vocab_size", 32))
            self.semantic_latent_num_tokens = int(semantic_cfg.get("latent_num_tokens", 2))
            self.semantic_latent_prefix = str(semantic_cfg.get("latent_token_prefix", "LAM")).strip("<>_")
            self.semantic_parse_trace_for_decoder = bool(semantic_cfg.get("parse_trace_for_decoder", False))
            self.semantic_trainable_token_rows = bool(semantic_cfg.get("trainable_token_rows", False))
            self.semantic_preload_tokens_before_checkpoint = bool(
                semantic_cfg.get("preload_tokens_before_checkpoint", False)
            )
            trainer_cfg = getattr(self.config, "trainer", None)
            pretrained_ckpt = getattr(trainer_cfg, "pretrained_checkpoint", None)
            resume_ckpt = getattr(trainer_cfg, "resume_from_checkpoint", None)
            # When continuing from an earlier, add new token rows after the old checkpoint
            # is loaded so existing embedding/lm-head rows are preserved.
            # When loading from an the LAM checkpoint that already contains LAM token
            # rows, callers can request preloading so embedding/lm-head shapes
            # match and the checkpoint rows are preserved.
            # Model-only resume is expected to restore an already-expanded (or
            # tokenizer-length-shrunk) the LAM checkpoint, so prepare the same token
            # rows before strict state_dict loading.
            if resume_ckpt or (not pretrained_ckpt) or self.semantic_preload_tokens_before_checkpoint:
                self.ensure_semantic_output_tokens()

        # Derive the LM-trace for logging:
        #   v0 — no LM aux; trace (if any) goes only through trace_encoder
        #   A  — LM aux on, injection_mode=none (pred trace is supervised only, not fed into decoder)
        #   B  — LM aux on, injection_mode in {sa_embs, adaln, both} (pred trace re-fed into decoder)
        if not self.lm_aux_loss:
            self.semantic_mode = "v0"
        elif self.injection_mode == "none":
            self.semantic_mode = "A"
        else:
            self.semantic_mode = "B"

        logger.info(
            f"[SemanticVLA] initialized mode='{self.semantic_mode}' "
            f"injection_mode='{self.injection_mode}' "
            f"lm_aux={self.lm_aux_loss} aux_weight={self.aux_loss_weight} "
            f"prompt_style='{self.prompt_style}' num_anchor_points={self.num_anchor_points} "
            f"semantic_output={self.semantic_output_enabled} semantic_mode='{self.semantic_output_mode}'"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_trace(examples, device, dtype) -> Optional[torch.Tensor]:
        """Collate per-example `trace_coords_window` into a (B, W, 2) tensor.

        Returns None if any example is missing the field (heterogeneous batch
        of LIBERO + non-LIBERO datasets, or trace disabled at the loader).
        """
        per_ex = [ex.get("trace_coords_window") for ex in examples]
        if any(t is None for t in per_ex):
            return None
        arr = np.stack([np.asarray(t, dtype=np.float32) for t in per_ex], axis=0)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def ensure_semantic_output_tokens(self) -> None:
        if not self.semantic_output_enabled or self._semantic_tokens_initialized:
            return
        if self.semantic_latent_vocab_size <= 0:
            self._semantic_tokens_initialized = True
            return
        tokens = lam_tokens(self.semantic_latent_vocab_size)
        if self.semantic_latent_prefix != "LAM":
            tokens = [tok.replace("<LAM_", f"<{self.semantic_latent_prefix}_") for tok in tokens]
        self.semantic_lam_token_ids = self.qwen_vl_interface.ensure_special_tokens(tokens)
        if self.semantic_trainable_token_rows:
            self.qwen_vl_interface.set_trainable_token_rows(self.semantic_lam_token_ids)
        self._semantic_tokens_initialized = True

    def _semantic_has_trace(self) -> bool:
        return self.semantic_output_mode in {"trace_only", "trace_latent", "latent_trace"}

    def _semantic_has_latent(self) -> bool:
        return self.semantic_output_mode in {"latent_only", "trace_latent", "latent_trace"}

    def _semantic_prompt_suffix(self) -> str:
        return semantic_prompt_template(
            mode=self.semantic_output_mode,
            prompt_style=self.semantic_prompt_style,
            num_anchors=self.semantic_trace_anchor_points,
            latent_num_tokens=self.semantic_latent_num_tokens,
            order=self.semantic_output_order,
        )

    def _semantic_max_new_tokens(self) -> int:
        trace_tokens = 0
        if self._semantic_has_trace():
            trace_tokens = 24 * self.semantic_trace_anchor_points + 16 if self.semantic_prompt_style == "qwen_point_2d" else 8 * self.semantic_trace_anchor_points + 8
        latent_tokens = self.semantic_latent_num_tokens + 8 if self._semantic_has_latent() else 0
        return int(max(8, trace_tokens + latent_tokens))

    # ------------------------------------------------------------------
    # forward / predict_action
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # Collect GT trace coords once (used by both the LM target text and the
        # action-decoder injection paths). Shape: (B, N, 2) in [0, 1] or None.
        device_marker = self.qwen_vl_interface.model.device  # used before we have last_hidden
        gt_trace_np = None
        if self.semantic_output_enabled and self._semantic_has_trace():
            per_ex = [ex.get("trace_coords_window") for ex in examples]
            if all(t is not None for t in per_ex):
                gt_trace_np = np.stack([np.asarray(t, dtype=np.float32) for t in per_ex], axis=0)
            else:
                raise KeyError("semantic_output mode requires trace_coords_window in every sample")
        elif self.lm_aux_loss:
            per_ex = [ex.get("trace_coords_window") for ex in examples]
            if all(t is not None for t in per_ex):
                gt_trace_np = np.stack([np.asarray(t, dtype=np.float32) for t in per_ex], axis=0)

        latent_np = None
        if self.semantic_output_enabled and self._semantic_has_latent():
            per_ex_latent = [ex.get("latent_action_idx") for ex in examples]
            if all(t is not None for t in per_ex_latent):
                latent_np = [np.asarray(t, dtype=np.int64).tolist() for t in per_ex_latent]
            else:
                raise KeyError("semantic_output mode requires latent_action_idx in every sample")

        # build semantic target text + prompt suffix for LM aux loss.
        solutions = None
        prompt_suffix = None
        if self.semantic_output_enabled:
            solutions = format_semantic_batch(
                mode=self.semantic_output_mode,
                batch_trace_coords=gt_trace_np if self._semantic_has_trace() else None,
                batch_latent_indices=latent_np if self._semantic_has_latent() else None,
                prompt_style=self.semantic_prompt_style,
                coord_range=self.coord_range,
                order=self.semantic_output_order,
                latent_prefix=self.semantic_latent_prefix,
            )
            prompt_suffix = self._semantic_prompt_suffix()
        elif self.lm_aux_loss and gt_trace_np is not None:
            solutions = format_trace_batch(
                gt_trace_np, style=self.prompt_style, coord_range=self.coord_range
            )
            prompt_suffix = prompt_template(self.prompt_style, gt_trace_np.shape[1])

        # VLM encode (with trace text as LM target when in mode A/B)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=solutions,
            mask_strategy="after_assistant" if solutions is not None else "action_token",
            prompt_suffix=prompt_suffix,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
            lm_aux_loss_bf16 = (
                qwenvl_outputs.loss
                if (solutions is not None and getattr(qwenvl_outputs, "loss", None) is not None)
                else torch.zeros((), device=last_hidden.device, dtype=torch.bfloat16)
            )

        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions_t[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer
                else 4
            )
            actions_target_rep = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_rep = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_rep = state_t.repeat(repeated_diffusion_steps, 1, 1)

            # Decoder-side trace: only fed when injection_mode != "none" (mode B / legacy v0).
            # In mode A (LM aux + injection_mode=none) the action_model.trace_encoder
            # is set to None at construction so trace_rep is ignored.
            if self.injection_mode != "none":
                trace_t = self._stack_trace(
                    examples, device=last_hidden.device, dtype=last_hidden.dtype
                )
                trace_rep = (
                    trace_t.repeat(repeated_diffusion_steps, 1, 1) if trace_t is not None else None
                )
            else:
                trace_rep = None

            pure_action_loss = self.action_model(
                last_hidden_rep,
                actions_target_rep,
                state_rep,
                trace_coords_window=trace_rep,
            )

            lm_aux_loss = lm_aux_loss_bf16.float()
            lm_weight = self.semantic_lm_loss_weight if self.semantic_output_enabled else self.aux_loss_weight
            total_loss = pure_action_loss + lm_weight * lm_aux_loss

        # NOTE: trainer reads `action_loss` for backprop. We return `action_loss=total_loss`
        # so train.py stays untouched. Sub-losses also returned for logging by
        # train_semanticvla.py fork.
        out = {
            "action_loss": total_loss,
            "pure_action_loss": pure_action_loss.detach(),
            "lm_trace_loss": lm_aux_loss.detach(),
        }
        if self.semantic_output_enabled:
            out["lm_semantic_loss"] = lm_aux_loss.detach()
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        trace_coords_window: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> dict:
        """Action inference.

        Behavior depends on `self.semantic_mode`:
          - v0:  vanilla VLM forward; caller may pass `trace_coords_window` to
                 inject (matches the training-time v0 contract).
          - A:   VLM autoregressive-generates trace text, then re-forwards the
                 full (prompt + generated) sequence to get a hidden state
                 aligned with the training distribution. Parsed trace is
                 discarded — `injection_mode == 'none'`.
          - B:   same generate + re-forward, but additionally parse the
                 generated text into coords and feed through the action head's
                 TraceEncoder for sa_embs / adaln injection.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        if self.semantic_output_enabled:
            prompt_suffix = self._semantic_prompt_suffix()
            prompt_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_suffix=prompt_suffix,
            )

            tokenizer = self.qwen_vl_interface.processor.tokenizer
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_out = self.qwen_vl_interface.model.generate(
                    **prompt_inputs,
                    max_new_tokens=self._semantic_max_new_tokens(),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                )

            prompt_len = prompt_inputs["input_ids"].shape[1]
            generated_ids = gen_out[:, prompt_len:]
            gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

            attn = prompt_inputs["attention_mask"]
            new_attn = torch.cat(
                [attn, torch.ones(
                    (attn.size(0), generated_ids.size(1)),
                    device=attn.device,
                    dtype=attn.dtype,
                )],
                dim=1,
            )
            full_inputs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask", "labels")}
            full_inputs["input_ids"] = gen_out
            full_inputs["attention_mask"] = new_attn
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full_outputs = self.qwen_vl_interface(
                    **full_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = full_outputs.hidden_states[-1]

            trace_t: Optional[torch.Tensor] = None
            if self.semantic_parse_trace_for_decoder and self._semantic_has_trace():
                pred_trace_np = np.stack(
                    [
                        parse_trace_from_semantic_text(
                            text,
                            num_anchors=self.semantic_trace_anchor_points,
                            style=self.semantic_prompt_style,
                            coord_range=self.coord_range,
                        )
                        for text in gen_texts
                    ],
                    axis=0,
                )
                trace_t = torch.from_numpy(pred_trace_np).to(
                    last_hidden.device, dtype=last_hidden.dtype
                )
        elif self.lm_aux_loss:
            # ---- the LM-trace (mode A or B) ----
            n_anchors = max(self.num_anchor_points, 1)
            prompt_suffix = prompt_template(self.prompt_style, n_anchors)
            prompt_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_suffix=prompt_suffix,
            )

            # Roughly size `max_new_tokens` to the expected target length.
            # plain / cot_bbox: "[[xxx, yyy], ...]" ~ 6 token/anchor + 2 brackets
            # qwen_point_2d JSON: dict with point_2d + label ~ 18 token/anchor
            if self.prompt_style == "qwen_point_2d":
                max_new = 24 * n_anchors + 16
            else:
                max_new = 8 * n_anchors + 8

            tokenizer = self.qwen_vl_interface.processor.tokenizer
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_out = self.qwen_vl_interface.model.generate(
                    **prompt_inputs,
                    max_new_tokens=int(max_new),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=False,
                )

            prompt_len = prompt_inputs["input_ids"].shape[1]
            generated_ids = gen_out[:, prompt_len:]
            gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Re-forward the full (prompt + generated) sequence to obtain
            # last_hidden aligned with training distribution.
            attn = prompt_inputs["attention_mask"]
            new_attn = torch.cat(
                [attn, torch.ones(
                    (attn.size(0), generated_ids.size(1)),
                    device=attn.device,
                    dtype=attn.dtype,
                )],
                dim=1,
            )
            full_inputs = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask", "labels")}
            full_inputs["input_ids"] = gen_out
            full_inputs["attention_mask"] = new_attn
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full_outputs = self.qwen_vl_interface(
                    **full_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = full_outputs.hidden_states[-1]

            # Mode B: parse generated trace text → (B, N, 2) coords for TraceEncoder.
            trace_t: Optional[torch.Tensor] = None
            if self.semantic_mode == "B":
                pred_trace_np = parse_trace_batch(
                    gen_texts,
                    num_anchors=n_anchors,
                    style=self.prompt_style,
                    coord_range=self.coord_range,
                )
                trace_t = torch.from_numpy(pred_trace_np).to(
                    last_hidden.device, dtype=last_hidden.dtype
                )
        else:
            # ---- v0 path (no LM aux; caller-supplied trace_coords_window) ----
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]
            trace_t = (
                torch.from_numpy(np.asarray(trace_coords_window, dtype=np.float32))
                .to(last_hidden.device, dtype=last_hidden.dtype)
                if trace_coords_window is not None
                else None
            )

        state_t = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        do_sample = kwargs.get("do_sample", True)
        if isinstance(do_sample, str):
            do_sample = do_sample.lower() in {"1", "true", "yes"}
        sample_seed = kwargs.get("sample_seed")

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state_t,
                do_sample=do_sample,
                sample_seed=sample_seed,
                trace_coords_window=trace_t,
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
