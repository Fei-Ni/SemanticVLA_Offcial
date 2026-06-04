"""Export SemanticVLA-LIBERO assets for pi.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.masking_utils import create_causal_mask

from semanticvla.model.framework.base_framework import baseframework

MODEL_NAME = "qwenvla"
PACKAGE_NAME = "qwenvla_libero"
ONNX_OPSET = 20
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PROMPT_CACHE_DIR = "prompt_cache/libero"
PROMPT_DUMMY_IMAGE_SIZE = 224
QWEN_IMAGE_SIZE = 256
QWEN_PIXEL_SHAPE = (512, 1536)
QWEN_IMAGE_GRID_THW = ((1, 16, 16), (1, 16, 16))
BACKBONE_OUTPUT_SCALE = 128.0
QWENVLA_INFERENCE_STEPS = 10
EXPORT_DTYPE = torch.float16


class QwenVlaActionStep(torch.nn.Module):
    def __init__(self, action_model: torch.nn.Module) -> None:
        super().__init__()
        self.action_model = action_model

    def forward(
        self,
        last_hidden: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        action_model = self.action_model
        last_hidden = last_hidden.float() * BACKBONE_OUTPUT_SCALE
        actions = actions.float()
        action_features = action_model.action_encoder(actions, timestep)
        if action_model.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            action_features = action_features + action_model.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = action_model.future_tokens.weight.unsqueeze(0).expand(last_hidden.shape[0], -1, -1)
        sa_embs = torch.cat((future_tokens, action_features), dim=1)
        model_output = action_model.model(
            hidden_states=sa_embs,
            encoder_hidden_states=last_hidden,
            timestep=timestep,
        )
        pred = action_model.action_decoder(model_output)
        pred_velocity = pred[:, -action_model.action_horizon :]
        return actions + (1.0 / action_model.num_inference_timesteps) * pred_velocity


class QwenVlaBackbone(torch.nn.Module):
    def __init__(self, qwen_vl_interface: torch.nn.Module) -> None:
        super().__init__()
        self.qwen_vl_interface = qwen_vl_interface
        self.register_buffer(
            "image_grid_thw",
            torch.tensor(QWEN_IMAGE_GRID_THW, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_select: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        model = self.qwen_vl_interface.model.model
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, self.image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * (1.0 - visual_select.sum(dim=-1, keepdim=True))
        inputs_embeds = inputs_embeds + torch.matmul(visual_select, image_embeds)

        hidden_states = inputs_embeds
        position_embeddings = model.language_model.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(model.language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            if layer_idx in range(len(deepstack_image_embeds)):
                visual_delta = torch.matmul(
                    visual_select,
                    deepstack_image_embeds[layer_idx].to(hidden_states.device, hidden_states.dtype),
                )
                hidden_states = hidden_states + visual_delta

        return (hidden_states / BACKBONE_OUTPUT_SCALE).float()


def _copy_qwen_assets(base_vlm: Path, output_dir: Path) -> None:
    qwen_dir = output_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    for path in base_vlm.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt"}:
            shutil.copy2(path, qwen_dir / path.name)


def _task_language(path: Path) -> str:
    name = path.name
    if name[0].isupper():
        offset = 8 if "SCENE10" in name else 7
        name = name[name.find("SCENE") + offset :]
    return name.removesuffix(".bddl").replace("_", " ")


def _libero_tasks(bddl_root: Path) -> list[tuple[str, int, str]]:
    tasks = []
    for suite in LIBERO_SUITES:
        for task_id, path in enumerate(sorted((bddl_root / suite).glob("*.bddl"))):
            tasks.append((suite, task_id, _task_language(path)))
    return tasks


def _build_prompt_input(model: torch.nn.Module, task: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(
        np.full((PROMPT_DUMMY_IMAGE_SIZE, PROMPT_DUMMY_IMAGE_SIZE, 3), 127, dtype=np.uint8),
        mode="RGB",
    )
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[[image, image]],
        instructions=[task],
        prompt_suffix=model._semantic_prompt_suffix(),
    )
    return (
        inputs["input_ids"][0].detach().cpu().numpy().astype(np.int64),
        inputs["attention_mask"][0].detach().cpu().numpy().astype(np.int64),
    )


def _prompt_position_and_mask(
    model: torch.nn.Module,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids_t = torch.from_numpy(input_ids[None]).to(device="cuda", dtype=torch.long)
    attention_mask_t = torch.from_numpy(attention_mask[None]).to(device="cuda", dtype=torch.long)
    image_grid_thw = torch.tensor(QWEN_IMAGE_GRID_THW, dtype=torch.long, device="cuda")
    qwen_model = model.qwen_vl_interface.model.model

    position_ids, _ = qwen_model.get_rope_index(
        input_ids_t,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask_t,
    )
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids_t)
    cache_position = torch.arange(input_ids_t.shape[1], device=input_ids_t.device)
    mask = create_causal_mask(
        config=qwen_model.language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask_t,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids[0],
    )
    if mask is None:
        dtype = inputs_embeds.dtype
        mask = torch.full(
            (1, 1, input_ids_t.shape[1], input_ids_t.shape[1]),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=input_ids_t.device,
        )
        mask = torch.triu(mask, diagonal=1)
    return (
        position_ids.detach().cpu().numpy().astype(np.int64),
        mask.detach().cpu().numpy().astype(np.float16),
    )


def _visual_select(input_ids: np.ndarray) -> np.ndarray:
    positions = np.where(input_ids == 151655)[0]
    select = np.zeros((1, input_ids.shape[0], positions.shape[0]), dtype=np.float16)
    select[0, positions, np.arange(positions.shape[0])] = 1.0
    return select


def _write_prompt_cache(model: torch.nn.Module, bddl_root: Path, output_dir: Path) -> int:
    tasks = _libero_tasks(bddl_root)
    prompt_inputs = [(suite, task_id, task, *_build_prompt_input(model, task)) for suite, task_id, task in tasks]
    prompt_len = max(input_ids.shape[0] for _, _, _, input_ids, _ in prompt_inputs)
    pad_id = int(model.qwen_vl_interface.processor.tokenizer.pad_token_id)

    cache_dir = output_dir / PROMPT_CACHE_DIR
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for suite, task_id, task, input_ids, attention_mask in prompt_inputs:
        padded_ids = np.full((prompt_len,), pad_id, dtype=np.int64)
        padded_mask = np.zeros((prompt_len,), dtype=np.int64)
        padded_ids[-input_ids.shape[0] :] = input_ids
        padded_mask[-attention_mask.shape[0] :] = attention_mask
        position_ids, attention_mask_4d = _prompt_position_and_mask(model, padded_ids, padded_mask)
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        path = cache_dir / f"{digest}.npz"
        np.savez(
            path,
            input_ids=padded_ids[None],
            attention_mask=padded_mask[None],
            attention_mask_4d=attention_mask_4d,
            position_ids=position_ids,
            visual_select=_visual_select(padded_ids),
        )
        index.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task": task,
                "sha256": digest,
                "path": f"{PROMPT_CACHE_DIR}/{path.name}",
                "token_len": int(input_ids.shape[0]),
            }
        )

    (output_dir / "prompt_cache_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return prompt_len


def _dummy_qwen_image_inputs(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.fromarray(
        np.full((PROMPT_DUMMY_IMAGE_SIZE, PROMPT_DUMMY_IMAGE_SIZE, 3), 127, dtype=np.uint8),
        mode="RGB",
    )
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[[image, image]],
        instructions=["pick up the black bowl and place it on the plate"],
        prompt_suffix=model._semantic_prompt_suffix(),
    )
    return inputs["pixel_values"], inputs["image_grid_thw"]


def _write_manifest(
    model: torch.nn.Module,
    checkpoint: Path,
    base_vlm: Path,
    output_dir: Path,
    prompt_len: int,
) -> None:
    action_model = model.action_model
    manifest = {
        "model": MODEL_NAME,
        "package": PACKAGE_NAME,
        "checkpoint": str(checkpoint),
        "base_vlm": str(base_vlm),
        "task_suites": list(LIBERO_SUITES),
        "inputs": {
            "num_cameras": 2,
            "image_size": [224, 224],
            "state_dim": 0,
        },
        "qwen": {
            "prompt_len": int(prompt_len),
            "mode": "prompt_forward",
            "hidden_dim": int(model.qwen_vl_interface.model.config.hidden_size),
            "pixel_values_shape": list(QWEN_PIXEL_SHAPE),
            "image_grid_thw": [list(values) for values in QWEN_IMAGE_GRID_THW],
            "position_ids_shape": [3, 1, int(prompt_len)],
            "attention_mask_shape": [1, 1, int(prompt_len), int(prompt_len)],
            "visual_select_shape": [1, int(prompt_len), 128],
            "prompt_cache": PROMPT_CACHE_DIR,
            "image_resize": [PROMPT_DUMMY_IMAGE_SIZE, PROMPT_DUMMY_IMAGE_SIZE],
            "processor_resize": [QWEN_IMAGE_SIZE, QWEN_IMAGE_SIZE],
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "output_scale": BACKBONE_OUTPUT_SCALE,
        },
        "action": {
            "horizon": int(action_model.action_horizon),
            "dim": int(action_model.action_dim),
            "num_inference_timesteps": int(action_model.num_inference_timesteps),
            "num_timestep_buckets": int(action_model.num_timestep_buckets),
            "do_sample": False,
            "sample_seed": 0,
            "initial_actions": "initial_actions.npz",
        },
        "onnx": {
            "backbone": "onnx/backbone.onnx",
            "action_step": "onnx/action_step.onnx",
        },
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _write_initial_actions(model: torch.nn.Module, output_dir: Path, dtype: torch.dtype) -> None:
    action_model = model.action_model
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    actions = torch.randn(
        (
            1,
            int(action_model.action_horizon),
            int(action_model.action_dim),
        ),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    np.savez(output_dir / "initial_actions.npz", actions=actions.float().detach().cpu().numpy())


def _export_backbone(model: torch.nn.Module, output_dir: Path, prompt_len: int) -> None:
    qwen = QwenVlaBackbone(model.qwen_vl_interface).eval()
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    cache_path = sorted((output_dir / PROMPT_CACHE_DIR).glob("*.npz"))[0]
    cache = np.load(cache_path)
    input_ids = torch.from_numpy(cache["input_ids"]).to(device="cuda", dtype=torch.long)
    attention_mask = torch.from_numpy(cache["attention_mask_4d"]).to(
        device="cuda",
        dtype=next(qwen.parameters()).dtype,
    )
    position_ids = torch.from_numpy(cache["position_ids"]).to(device="cuda", dtype=torch.long)
    visual_select = torch.from_numpy(cache["visual_select"]).to(
        device="cuda",
        dtype=next(qwen.parameters()).dtype,
    )
    pixel_values, _ = _dummy_qwen_image_inputs(model)
    pixel_values = pixel_values.to(device="cuda", dtype=next(qwen.parameters()).dtype)

    torch.onnx.export(
        qwen,
        (input_ids, attention_mask, position_ids, visual_select, pixel_values),
        str(onnx_dir / "backbone.onnx"),
        input_names=["input_ids", "attention_mask", "position_ids", "visual_select", "pixel_values"],
        output_names=["last_hidden"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
    )


def _export_action_step(model: torch.nn.Module, output_dir: Path, prompt_len: int) -> None:
    action_model = model.action_model.eval()
    wrapper = QwenVlaActionStep(action_model).eval()
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim = int(model.qwen_vl_interface.model.config.hidden_size)
    action_horizon = int(action_model.action_horizon)
    action_dim = int(action_model.action_dim)
    seq_len = prompt_len
    last_hidden = torch.zeros((1, seq_len, hidden_dim), dtype=torch.float32, device="cuda")
    actions = torch.zeros((1, action_horizon, action_dim), dtype=torch.float32, device="cuda")
    timestep = torch.zeros((1,), dtype=torch.long, device="cuda")

    torch.onnx.export(
        wrapper,
        (last_hidden, actions, timestep),
        str(onnx_dir / "action_step.onnx"),
        input_names=["last_hidden", "actions", "timestep"],
        output_names=["actions_next"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
    )


def export_qwenvla(*, checkpoint: Path, base_vlm: Path, bddl_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = baseframework.from_pretrained(str(checkpoint)).to("cuda").eval()
    model.action_model.num_inference_timesteps = QWENVLA_INFERENCE_STEPS
    model.qwen_vl_interface.model.config._attn_implementation = "eager"
    model.qwen_vl_interface.model.config.text_config._attn_implementation = "eager"
    model.qwen_vl_interface.model.to(EXPORT_DTYPE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    shutil.copy2(checkpoint.parents[1] / "config.yaml", output_dir / "config.yaml")
    shutil.copy2(checkpoint.parents[1] / "dataset_statistics.json", output_dir / "dataset_statistics.json")
    _copy_qwen_assets(base_vlm, output_dir)
    prompt_len = _write_prompt_cache(model, bddl_root, output_dir)
    _write_initial_actions(model, output_dir, EXPORT_DTYPE)
    _write_manifest(model, checkpoint, base_vlm, output_dir, prompt_len)
    _export_backbone(model, output_dir, prompt_len)
    _export_action_step(model, output_dir, prompt_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--libero-bddl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_qwenvla(
        checkpoint=args.checkpoint,
        base_vlm=args.base_vlm,
        bddl_root=args.libero_bddl_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
