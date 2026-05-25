# Copyright 2025 semanticvla community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# [151936, 153984]
_ACTION_TOKEN_MIN = 151936 # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
_ACTION_TOKEN_MAX = 153984 # here only for fast_tokenizer, see semanticvla/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")
        device_map = qwenvl_config.get("device_map", "cuda")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self._embed_tokens_hook_handle = None
        self._embed_tokens_hook_mask = None

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def _flatten_image_batch(self, images) -> Tuple[List, List[int]]:
        """
        Normalize image inputs into a flat list plus per-sample image counts.

        Supported inputs:
          - batch of multi-view samples: [[img0, img1], [img2], ...]
          - flat list of images: [img0, img1, ...]
          - single image
        """
        if images is None:
            raise ValueError("images must not be None")

        if isinstance(images, list):
            if len(images) == 0:
                return [], []
            if isinstance(images[0], list):
                sample_counts = [len(sample) for sample in images]
                flat_images = [img for sample in images for img in sample]
                return flat_images, sample_counts
            return list(images), [1] * len(images)

        return [images], [1]

    def build_qwenvl_vision_inputs(self, images):
        """
        Build vision-only inputs for the Qwen3-VL image tower.

        Returns:
            BatchFeature with `pixel_values`, `image_grid_thw`, and
            `sample_image_counts` describing how flat images map back to samples.
        """
        flat_images, sample_image_counts = self._flatten_image_batch(images)
        if len(flat_images) == 0:
            raise ValueError("No images provided for vision inputs")

        image_inputs = self.processor.image_processor(
            images=flat_images,
            return_tensors="pt",
        )
        image_inputs = image_inputs.to(self.model.device)
        image_inputs["sample_image_counts"] = sample_image_counts
        return image_inputs

    def _extract_vit_patch_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Run the Qwen3-VL vision tower up to the final vision block, but stop
        before the patch merger so we keep per-patch ViT features.
        """
        visual = self.model.visual

        hidden_states = visual.patch_embed(pixel_values)
        pos_embeds = visual.fast_pos_embed_interpolate(image_grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in visual.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        split_sizes = image_grid_thw.prod(-1).tolist()
        return list(torch.split(hidden_states, split_sizes, dim=0))

    def extract_vit_pooled_features(
        self,
        images,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Extract ViT-only pooled image features:
          patch tokens -> spatial average pool -> optional L2 normalisation

        Returns:
            Tensor [batch_size, d_vit]
        """
        vision_inputs = self.build_qwenvl_vision_inputs(images)
        pixel_values = vision_inputs["pixel_values"]
        image_grid_thw = vision_inputs["image_grid_thw"]
        sample_image_counts = vision_inputs["sample_image_counts"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            per_image_tokens = self._extract_vit_patch_tokens(pixel_values, image_grid_thw)
            per_image_pooled = torch.stack(
                [tokens.mean(dim=0) for tokens in per_image_tokens],
                dim=0,
            )

        pooled_per_sample = []
        start = 0
        for count in sample_image_counts:
            sample_feat = per_image_pooled[start : start + count].mean(dim=0)
            pooled_per_sample.append(sample_feat)
            start += count

        features = torch.stack(pooled_per_sample, dim=0).float()
        if normalize:
            features = F.normalize(features, dim=-1, eps=eps)
        return features

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        mask_strategy: str = "action_token",
        prompt_suffix: str | None = None,
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        Args:
            images: list[list[PIL.Image]], one inner list per sample (multi-view).
            instructions: list[str], one task instruction per sample.
            solutions: list[str] or None. If provided, each solution is appended
                as an assistant turn and the returned dict carries `labels` with
                non-solution tokens masked to IGNORE_INDEX. If None, no labels.
            mask_strategy: how to mask labels when solutions are given.
              - "action_token" (DEFAULT, legacy): mask everything up to the
                first action-token id (>= _ACTION_TOKEN_MIN). Used by OFT-style
                action-token training.
              - "after_assistant" (the LM-trace path): mask the entire
                user-prompt segment, keep only the assistant-turn tokens
                (i.e. the solution text) for CE. Works regardless of the token
                vocabulary — derives prompt length by re-templating the same
                messages with the assistant turn stripped.
            prompt_suffix: if not None, appended after the (resolved) user
                prompt with a leading space. Used by SemanticVLA to inject the
                trace-prediction instruction without editing
                `datasets.vla_data.CoT_prompt`.
        """
        if mask_strategy not in ("action_token", "after_assistant"):
            raise ValueError(
                f"mask_strategy must be 'action_token' or 'after_assistant'; got {mask_strategy!r}"
            )

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            if prompt_suffix:
                prompt = f"{prompt} {prompt_suffix}"

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            labels = batch_inputs['input_ids'].clone()

            if mask_strategy == "action_token":
                action_token_min = _ACTION_TOKEN_MIN  # see add_qwen_special_tokens/README.md
                action_token_max = _ACTION_TOKEN_MAX
                # For each sequence, find the first action-token and mask everything before it.
                for i in range(labels.size(0)):
                    seq = labels[i]
                    mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                    nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                    if nonzero_indices.numel() > 0:
                        first_action_index = nonzero_indices[0].item()
                        seq[:first_action_index] = IGNORE_INDEX
                    else:
                        seq[:] = IGNORE_INDEX
                        RuntimeWarning(
                            f"action token are on in yout tokenizer, plz see "
                            f"semanticvla/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                        )

            elif mask_strategy == "after_assistant":
                # Re-template the same messages WITHOUT the assistant turn, taking
                # advantage of `add_generation_prompt=True` so the prompt-only
                # encoding ends exactly at `<|im_start|>assistant\n` — the first
                # token the model is expected to produce. Masking the labels up
                # to that length keeps only the solution tokens in CE.
                prompt_only_msgs = [m[:1] for m in messages]  # strip assistant turn
                prompt_only_inputs = self.processor.apply_chat_template(
                    prompt_only_msgs,
                    tokenize=True,
                    padding=False,  # per-sample lengths needed; no padding
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors=None,  # python lists, easier to len()
                )
                prompt_lens = [len(ids) for ids in prompt_only_inputs["input_ids"]]
                for i, plen in enumerate(prompt_lens):
                    # Note: the full-batch is padded on the right; the prompt
                    # portion occupies the first `plen` positions (Qwen padding
                    # is right-side by default for vision-language models).
                    labels[i, :plen] = IGNORE_INDEX

            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)

    def ensure_special_tokens(
        self,
        tokens: List[str],
        *,
        init_std: float = 0.01,
        init_token_id: Optional[int] = None,
    ) -> List[int]:
        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        missing_tokens = [token for token in tokens if token not in vocab]
        if missing_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            self.model.resize_token_embeddings(len(tokenizer))

            embed = self.model.get_input_embeddings().weight
            if init_token_id is None:
                init_token_id = tokenizer.eos_token_id
                if init_token_id is None:
                    init_token_id = tokenizer.pad_token_id
                if init_token_id is None:
                    init_token_id = 0

            with torch.no_grad():
                ref = embed[int(init_token_id)].detach().clone()
                for token in missing_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    embed[token_id].copy_(ref + init_std * torch.randn_like(ref))

        return [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]

    def set_trainable_token_rows(self, token_ids: List[int]):
        embed_tokens = self.model.get_input_embeddings()
        embed_tokens.weight.requires_grad = True

        if self._embed_tokens_hook_handle is not None:
            self._embed_tokens_hook_handle.remove()
            self._embed_tokens_hook_handle = None

        mask = torch.zeros(
            embed_tokens.weight.shape[0],
            device=embed_tokens.weight.device,
            dtype=embed_tokens.weight.dtype,
        )
        mask[token_ids] = 1.0
        self._embed_tokens_hook_mask = mask.view(-1, 1)

        def grad_hook(grad):
            if (
                self._embed_tokens_hook_mask.device != grad.device
                or self._embed_tokens_hook_mask.dtype != grad.dtype
            ):
                self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.to(
                    device=grad.device,
                    dtype=grad.dtype,
                )
            return grad * self._embed_tokens_hook_mask

        self._embed_tokens_hook_handle = embed_tokens.weight.register_hook(grad_hook)

    def append_special_token_ids(self, batch_inputs: BatchFeature, token_ids: List[int]):
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        token_tensor = torch.tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
        token_tensor = token_tensor.unsqueeze(0).expand(batch_size, -1)
        attn_tail = torch.ones_like(token_tensor, dtype=attention_mask.dtype)

        base_seq_len = int(input_ids.shape[1])
        out = BatchFeature(dict(batch_inputs))
        out["input_ids"] = torch.cat([input_ids, token_tensor], dim=1)
        out["attention_mask"] = torch.cat([attention_mask, attn_tail], dim=1)
        if "labels" in out:
            ignore_tail = torch.full(
                (batch_size, len(token_ids)),
                fill_value=IGNORE_INDEX,
                device=out["labels"].device,
                dtype=out["labels"].dtype,
            )
            out["labels"] = torch.cat([out["labels"], ignore_tail], dim=1)
        return out, base_seq_len




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./semanticvla/config/training/cotrain_oxe.yamll", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
