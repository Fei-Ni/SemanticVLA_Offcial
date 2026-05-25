#!/usr/bin/env python3
"""
Molmo-72B inference (V2) — keyframe strategy
Reads frames from the raw dataset and runs inference on a keyframe set (first, last, plus 8 evenly-spaced middle frames).
Speeds up annotation significantly while preserving reasonable accuracy.
"""

import os
import glob
import bisect
# environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid tokenizer parallelism warnings

import torch
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import logging
import re
import argparse
import json
import io
from typing import Dict, Optional, Tuple, List, Sequence, Set
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import multiprocessing as mp

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global model cache
_model_cache = {
    'model': None,
    'processor': None,
    'model_path': None
}

DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_STOP_STRINGS = ("<|endoftext|>", "\n\n")

def disable_tensorflow_gpu() -> None:
    """Use TF only to read TFRecord; keep inference GPU free."""
    try:
        tf.config.set_visible_devices([], "GPU")
        logger.info("Disabled TF GPU devices to avoid competing with PyTorch for GPU memory.")
    except Exception as e:
        logger.warning(f"Failed to disable TF GPU; continuing: {e}")

def parse_molmo_coordinates(response_text, quiet: bool = False):
    """
    Parse coordinates from a Molmo response.
    
    Args:
        response_text: Molmo model response text
        
    Returns:
        (x, y) tuple in the [0, 100] range, or None
    """
    # clean response text
    response_text = response_text.strip()
    
    # multiple regex patterns for various output formats
    patterns = [
        # XML format: <point x="43.0" y="58.0" alt="robot gripper">robot gripper</point>
        r'<point\s+x="([0-9]+(?:\.[0-9]+)?)"\s+y="([0-9]+(?:\.[0-9]+)?)"',
        # format: "Gripper coordinates: (x, y)"
        r'gripper\s+coordinates?\s*[:\s]*\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?',
        # format: "coordinates: (x, y)"
        r'coordinates?\s*[:\s]*\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?',
        # format: "(x, y)"
        r'\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?',
        # format: "x: 123, y: 456"
        r'x\s*[:\s]*(\d+(?:\.\d+)?).*?y\s*[:\s]*(\d+(?:\.\d+)?)',
        # format: "123, 456"
        r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text.lower())
        if matches:
            try:
                x, y = float(matches[0][0]), float(matches[0][1])
                # validate coordinate range
                if 0 <= x <= 100 and 0 <= y <= 100:
                    return (x, y)
                else:
                    if not quiet:
                        logger.warning(f"coordinate out of [0, 100] range: ({x}, {y})")
                    continue
            except (ValueError, IndexError) as e:
                if not quiet:
                    logger.warning(f"coordinate parse failed: {e}")
                continue
    
    if not quiet:
        logger.warning(f"could not parse coordinates from response: {response_text}")
    return None

def select_key_frames(episode_length: int) -> List[int]:
    """
    Pick keyframes: first + last + 8 evenly-spaced middle frames
    
    Args:
        episode_length: total episode length
        
    Returns:
        list of keyframe indices
    """
    if episode_length <= 2:
        return list(range(episode_length))
    elif episode_length <= 10:
        # short episode: sample every 2 frames
        return list(range(0, episode_length, 2))
    else:
        # long episode: first + last + 8 evenly-spaced middle frames
        key_frames = [0, episode_length - 1]  # first and last
        
        if episode_length > 10:
            # 8 evenly-spaced middle samples
            middle_frames = []
            step = (episode_length - 2) // 9  # 9 intervals, 8 middle frames
            for i in range(1, 9):
                frame_idx = i * step
                if 0 < frame_idx < episode_length - 1:
                    middle_frames.append(frame_idx)
            
            key_frames.extend(middle_frames)
        
        return sorted(list(set(key_frames)))

def load_model(model_path: str, use_cpu: bool = False):
    """
    Load the model with caching.
    
    Args:
        model_path: model path
        use_cpu: force CPU
        
    Returns:
        (model, processor) tuple
    """
    global _model_cache
    
    # return cached model if already loaded with the same path
    if (_model_cache['model'] is not None and 
        _model_cache['processor'] is not None and 
        _model_cache['model_path'] == model_path):
        logger.info("using cached model")
        return _model_cache['model'], _model_cache['processor']
    
    logger.info(f"loading model: {model_path}")
    start_time = time.time()
    
    # check CUDA availability
    cuda_available = torch.cuda.is_available() and not use_cpu
    torch_dtype = torch.bfloat16 if cuda_available else torch.float32
    
    logger.info(f"device: {'CUDA' if cuda_available else 'CPU'}")
    
    # load the processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # load the model
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch_dtype,
        'low_cpu_mem_usage': True
    }
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        model_kwargs.update({
            'device_map': 'auto',
            'max_memory': {gpu_id: "95GiB" for gpu_id in range(gpu_count)}
        })
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # if CPU was requested, move model to CPU explicitly
    if not cuda_available:
        model = model.to('cpu')
    
    # cache the model
    _model_cache['model'] = model
    _model_cache['processor'] = processor
    _model_cache['model_path'] = model_path
    
    load_time = time.time() - start_time
    logger.info(f"model loaded; {load_time:.2f}s")
    
    return model, processor


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def process_single_input(image, prompt: str, processor) -> Dict[str, torch.Tensor]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return processor.process(images=[image], text=prompt)


def collate_inputs(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_seq_len = max(sample["input_ids"].shape[0] for sample in samples)

    input_ids = samples[0]["input_ids"].new_full((batch_size, max_seq_len), -1)
    for row, sample in enumerate(samples):
        seq_len = sample["input_ids"].shape[0]
        input_ids[row, :seq_len] = sample["input_ids"]

    batch = {"input_ids": input_ids}

    if "images" not in samples[0]:
        return batch

    max_crops = max(sample["images"].shape[0] for sample in samples)
    num_patches = samples[0]["images"].shape[1]
    patch_dim = samples[0]["images"].shape[2]
    patch_tokens = samples[0]["image_input_idx"].shape[1]

    batch["images"] = samples[0]["images"].new_full((batch_size, max_crops, num_patches, patch_dim), -1)
    batch["image_input_idx"] = samples[0]["image_input_idx"].new_full((batch_size, max_crops, patch_tokens), -1)

    if "image_masks" in samples[0]:
        mask_tokens = samples[0]["image_masks"].shape[1]
        batch["image_masks"] = samples[0]["image_masks"].new_zeros((batch_size, max_crops, mask_tokens))

    for row, sample in enumerate(samples):
        num_crops = sample["images"].shape[0]
        batch["images"][row, :num_crops] = sample["images"]
        batch["image_input_idx"][row, :num_crops] = sample["image_input_idx"]
        if "image_masks" in batch:
            batch["image_masks"][row, :num_crops] = sample["image_masks"]

    return batch


def build_batch_inputs(images: Sequence[Image.Image], prompt: str, processor, device: torch.device) -> Dict[str, torch.Tensor]:
    processed = [process_single_input(image, prompt, processor) for image in images]
    collated = collate_inputs(processed)
    return {key: value.to(device) for key, value in collated.items()}


def decode_generated_texts(output: torch.Tensor, inputs: Dict[str, torch.Tensor], processor) -> List[str]:
    if output.dim() == 1:
        output = output.unsqueeze(0)

    prompt_len = inputs["input_ids"].size(1)
    generated_tokens = output[:, prompt_len:]
    decode_batch = getattr(processor, "batch_decode", None)
    if decode_batch is None:
        decode_batch = processor.tokenizer.batch_decode
    return decode_batch(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def inference_single_image_legacy(image, prompt, model, processor):
    """Legacy per-frame decoding path used only as a fallback."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor.process(
        images=[image],
        text=prompt
    )

    device = get_model_device(model)
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_enabled = torch.cuda.is_available()
    with torch.autocast(device_type=device_type, enabled=autocast_enabled, dtype=torch.bfloat16):
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                generated_tokens = [next_token.item()]
                input_ids = inputs['input_ids'].clone()

                for _ in range(200):
                    if next_token.item() == processor.tokenizer.eos_token_id:
                        break

                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_inputs = {k: v for k, v in inputs.items()}
                    new_inputs['input_ids'] = input_ids

                    outputs = model(**new_inputs)
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    generated_tokens.append(next_token.item())

                output = torch.tensor(generated_tokens).unsqueeze(0)
        except Exception as e:
            logger.error(f"legacy generation failed: {e}")
            output = torch.tensor([[processor.tokenizer.eos_token_id]])

    if output.dim() == 2 and output.size(0) == 1:
        return processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return processor.tokenizer.decode(output, skip_special_tokens=True)


def inference_batch_images(
    images: Sequence[Image.Image],
    prompt: str,
    model,
    processor,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
) -> List[str]:
    """Use generate_from_batch for true batched inference; fall back to legacy per-frame on failure."""
    if not images:
        return []

    device = get_model_device(model)
    inputs = build_batch_inputs(images, prompt, processor, device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stop_strings=list(stop_strings),
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_enabled = torch.cuda.is_available()

    try:
        sync_cuda()
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=autocast_enabled, dtype=torch.bfloat16):
                output = model.generate_from_batch(inputs, generation_config, tokenizer=processor.tokenizer)
        sync_cuda()
        return decode_generated_texts(output, inputs, processor)
    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"batched inference OOM; retrying with smaller batch（batch={len(images)}）: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(images) == 1:
            logger.error("per-frame inference still OOM; returning empty response to avoid repeated OOM")
            return [""]

        retry_results: List[str] = []
        for image in images:
            retry_results.extend(
                inference_batch_images(
                    [image],
                    prompt,
                    model,
                    processor,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                )
            )
        return retry_results
    except Exception as e:
        logger.warning(f"batched inference failed; falling back to legacy per-frame decoding: {e}")
        return [inference_single_image_legacy(image, prompt, model, processor) for image in images]

def inference_single_image(image, prompt, model, processor):
    """
    Run inference on a single image.
    
    Args:
        image: PIL.Image object
        prompt: inference prompt
        model: loaded model
        processor: loaded processor
        
    Returns:
        generated text
    """
    return inference_batch_images([image], prompt, model, processor)[0]

def linear_interpolation(key_positions: List[Tuple[int, float, float]], 
                        total_frames: int) -> List[Tuple[float, float]]:
    """
    Linearly interpolate to fill intermediate frame positions.
    
    Args:
        key_positions: keyframe positions [(frame_idx, x, y), ...]
        total_frames: total frame count
    
    Returns:
        All-frame positions: [(x, y), ...]
    """
    if len(key_positions) < 2:
        if key_positions:
            x, y = key_positions[0][1], key_positions[0][2]
            return [(x, y)] * total_frames
        else:
            return [(50.0, 50.0)] * total_frames
    
    key_positions.sort(key=lambda x: x[0])
    
    all_positions = []
    key_idx = 0
    
    for frame_idx in range(total_frames):
        while (key_idx < len(key_positions) - 1 and 
               key_positions[key_idx + 1][0] <= frame_idx):
            key_idx += 1
        
        if key_idx >= len(key_positions) - 1:
            x, y = key_positions[-1][1], key_positions[-1][2]
        elif key_positions[key_idx][0] == frame_idx:
            x, y = key_positions[key_idx][1], key_positions[key_idx][2]
        else:
            frame1, x1, y1 = key_positions[key_idx]
            frame2, x2, y2 = key_positions[key_idx + 1]
            
            if frame2 > frame1:
                t = (frame_idx - frame1) / (frame2 - frame1)
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
            else:
                x, y = x1, y1
        
        all_positions.append((x, y))
    
    return all_positions

def process_episode_keyframes(episode_images: List[Image.Image], 
                             episode_idx: int, 
                             language_instruction: str,
                             model, processor, 
                             prompt: str = "point to the robot gripper",
                             keyframe_batch_size: int = 4,
                             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                             log_frame_details: bool = False,
                             output_dir: str = None) -> List[Dict]:
    """
    Process a single episode using the keyframe strategy.
    
    Args:
        episode_images: all images of the episode
        episode_idx: episode index
        language_instruction: natural-language instruction
        model: loaded model
        processor: loaded processor
        prompt: inference prompt
        
    Returns:
        list of processed results
    """
    episode_length = len(episode_images)
    logger.info(f"Episode {episode_idx}: starting keyframe processing; total frames: {episode_length}")
    
    key_frames = select_key_frames(episode_length)
    logger.debug(f"Episode {episode_idx}: selected keyframes: {key_frames}")
    
    key_positions = []
    batch_size = max(1, keyframe_batch_size)
    for start_idx in range(0, len(key_frames), batch_size):
        batch_frames = [frame_idx for frame_idx in key_frames[start_idx:start_idx + batch_size] if frame_idx < episode_length]
        if not batch_frames:
            continue

        batch_images = [episode_images[frame_idx] for frame_idx in batch_frames]

        try:
            batch_start = time.time()
            generated_texts = inference_batch_images(
                batch_images,
                prompt,
                model,
                processor,
                max_new_tokens=max_new_tokens,
            )
            batch_time = time.time() - batch_start

            for frame_idx, generated_text in zip(batch_frames, generated_texts):
                coords = parse_molmo_coordinates(generated_text, quiet=not log_frame_details)

                if coords:
                    key_positions.append((frame_idx, coords[0], coords[1]))
                    if log_frame_details:
                        logger.info(f"Episode {episode_idx}, frame {frame_idx}: parsed coords {coords}")
                else:
                    if log_frame_details:
                        logger.warning(f"Episode {episode_idx}, frame {frame_idx}: could not parse a valid coordinate")

            logger.debug(
                f"Episode {episode_idx}: keyframe batch {batch_frames} inference took {batch_time:.2f}s "
                f"({batch_time / len(batch_frames):.2f}s/frame)"
            )
        except Exception as e:
            logger.error(f"Episode {episode_idx}: batch keyframe {batch_frames} inference failed: {e}")
    
    logger.info(f"Episode {episode_idx}: keyframe inference done; {len(key_positions)} positions")
    
    results = []
    for frame_idx, x, y in key_positions:
        result = {
            'episode_idx': episode_idx,
            'step_idx': frame_idx,
            'generated_text': f"Keyframe inference: ({x:.1f}, {y:.1f})",
            'molmo_coords': (x, y),
            'is_keyframe': True
        }
        results.append(result)
    
    logger.info(f"Episode {episode_idx}: keyframe processing done; {len(results)} results")
    
    if output_dir and key_positions:
        try:
            create_episode_animations(episode_images, key_positions, episode_idx, output_dir)
        except Exception as e:
            logger.warning(f"Episode {episode_idx}: animation creation failed: {e}")
    
    return results

def parse_tfrecord_example(example):
    """Parse a tfrecord example."""
    parsed = tf.io.parse_single_example(example, {
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'steps/observation/natural_language_instruction': tf.io.VarLenFeature(tf.string),
        'steps/observation/wrist_image': tf.io.VarLenFeature(tf.string),
        'steps/observation/image': tf.io.VarLenFeature(tf.string),
    })
    return parsed


def _is_split_tfrecord(name: str, split_name: str) -> bool:
    """True if filename looks like a BC-Z TFRecord shard for split_name."""
    if not (name.endswith(".tfrecord") or ".tfrecord-" in name):
        return False
    return (
        f"-{split_name}." in name
        or f"-{split_name}-" in name
        or f"_{split_name}." in name
        or f"_{split_name}-" in name
        or f"{split_name}.tfrecord" in name
    )


def _list_split_tfrecord_files(dataset_path: str, split_name: str = "train") -> List[str]:
    return sorted(name for name in os.listdir(dataset_path) if _is_split_tfrecord(name, split_name))


def get_total_episodes(dataset_path: str, split_name: str = "train") -> int:
    """Return the total episode count for the dataset."""
    tfrecord_files = [
        os.path.join(dataset_path, name) for name in _list_split_tfrecord_files(dataset_path, split_name)
    ]
    if not tfrecord_files:
        raise ValueError(f"no tfrecord files found under {dataset_path}")
    
    total_episodes = 0
    logger.info(f"scanning {len(tfrecord_files)} {split_name} tfrecord files for total episode count...")
    
    for file_idx, file in enumerate(tfrecord_files):
        try:
            dataset = tf.data.TFRecordDataset(file)
            file_episodes = 0
            for _ in dataset:
                file_episodes += 1
            total_episodes += file_episodes
            logger.info(f"file {file_idx+1}/{len(tfrecord_files)}: {file} contains {file_episodes} episodes")
        except Exception as e:
            logger.error(f"error while reading file {file}: {e}")
            continue
    
    if total_episodes == 0:
        raise ValueError(f"could not read episode data from any tfrecord file")
    
    logger.info(f"found {total_episodes} episodes in total")
    return total_episodes

def load_shard_lengths_from_dataset_info(
    dataset_path: str,
    expected_num_shards: int,
    split_name: str = "train",
) -> Optional[List[int]]:
    """Read per-tfrecord episode counts from dataset_info.json."""
    info_path = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception as e:
        logger.warning(f"failed to read dataset_info.json: {e}")
        return None

    splits = info.get("splits", [])
    if not isinstance(splits, list):
        return None

    selected_split = None
    for split in splits:
        if split.get("name") == split_name:
            selected_split = split
            break
    if selected_split is None and len(splits) == 1:
        selected_split = splits[0]
    if selected_split is None:
        return None

    raw_lengths = selected_split.get("shardLengths")
    if not isinstance(raw_lengths, list):
        return None
    if len(raw_lengths) != expected_num_shards:
        logger.warning(
            f"dataset_info shardLengths length ({len(raw_lengths)}) does not match tfrecord count ({expected_num_shards}); falling back to approximation"
        )
        return None

    try:
        return [int(x) for x in raw_lengths]
    except Exception as e:
        logger.warning(f"failed to parse shardLengths: {e}")
        return None

def load_episode_list_file(path: str) -> List[int]:
    """Load a target episode list from a text file. Comment lines are allowed; first column of each line is taken as an int."""
    episodes: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            try:
                episodes.append(int(token))
            except ValueError:
                logger.warning(f"episode list line {lineno} could not be parsed as int; skipping: {raw_line.rstrip()}")
    deduped = sorted(set(episodes))
    if not deduped:
        raise ValueError(f"episode list is empty: {path}")
    return deduped

def build_target_tfrecord_index_set(
    target_episode_set: Set[int],
    shard_prefix: Optional[List[int]],
    total_tfrecords: int,
    episodes_per_tfrecord: int,
    episode_index_offset: int = 0,
) -> Optional[Set[int]]:
    """From the target episode set, compute which tfrecord indices need to be read."""
    if not target_episode_set:
        return None

    target_tfrecord_idxs: Set[int] = set()
    sorted_targets = sorted(episode_idx - episode_index_offset for episode_idx in target_episode_set)

    if shard_prefix is not None:
        for episode_idx in sorted_targets:
            tfrecord_idx = bisect.bisect_right(shard_prefix, episode_idx) - 1
            if 0 <= tfrecord_idx < total_tfrecords and shard_prefix[tfrecord_idx] <= episode_idx < shard_prefix[tfrecord_idx + 1]:
                target_tfrecord_idxs.add(tfrecord_idx)
            else:
                logger.warning(f"episode {episode_idx} is out of dataset_info shard range; skipping")
        return target_tfrecord_idxs

    if episodes_per_tfrecord > 0:
        for episode_idx in sorted_targets:
            tfrecord_idx = episode_idx // episodes_per_tfrecord
            if 0 <= tfrecord_idx < total_tfrecords:
                target_tfrecord_idxs.add(tfrecord_idx)
        return target_tfrecord_idxs

    return None

def iter_episode_data(
    dataset_path: str,
    start_episode: int,
    end_episode: Optional[int],
    start_tfrecord_idx: int = 0,
    end_tfrecord_idx: Optional[int] = None,
    episodes_per_tfrecord: int = 80,
    target_episode_set: Optional[Set[int]] = None,
    split_name: str = "train",
    episode_index_offset: int = 0,
):
    """Stream-read episodes in the requested range; tfrecord-sharded."""
    tfrecord_files = _list_split_tfrecord_files(dataset_path, split_name)
    total_tfrecords = len(tfrecord_files)

    shard_lengths = load_shard_lengths_from_dataset_info(dataset_path, total_tfrecords, split_name)
    shard_prefix = None
    if shard_lengths is not None:
        shard_prefix = [0]
        for length in shard_lengths:
            shard_prefix.append(shard_prefix[-1] + length)
        logger.info("using dataset_info.shardLengths to compute the global episode_idx")
    else:
        logger.warning("no usable shardLengths; falling back to approximate episodes_per_tfrecord")

    tfrecord_start = max(0, start_tfrecord_idx)
    tfrecord_end = total_tfrecords if end_tfrecord_idx is None else min(end_tfrecord_idx, total_tfrecords)

    if tfrecord_start >= tfrecord_end:
        logger.error(
            f"invalid tfrecord range: start={tfrecord_start}, end={tfrecord_end}, total={total_tfrecords}"
        )
        return

    logger.info(
        f"processing tfrecord range: [{tfrecord_start}, {tfrecord_end}) / {total_tfrecords}, "
        f"episodes_per_tfrecord={episodes_per_tfrecord}"
    )

    selected_file_indices = list(range(tfrecord_start, tfrecord_end))
    if target_episode_set is not None:
        target_tfrecord_idxs = build_target_tfrecord_index_set(
            target_episode_set,
            shard_prefix,
            total_tfrecords,
            episodes_per_tfrecord,
            episode_index_offset,
        )
        if target_tfrecord_idxs is not None:
            selected_file_indices = [
                idx for idx in sorted(target_tfrecord_idxs)
                if tfrecord_start <= idx < tfrecord_end
            ]
            if not selected_file_indices:
                logger.warning("target episode list and current tfrecord filter range have no intersection")
                return
            logger.info(
                "episode-list mode: %d target episodes; matched %d tfrecords",
                len(target_episode_set),
                len(selected_file_indices),
            )
    if end_episode is not None:
        logger.info(f"global episode filter range: [{start_episode}, {end_episode})")
    else:
        logger.info(f"global episode filter range: [{start_episode}, end)")

    yielded_episodes = 0
    fallback_episode_idx = 0

    for file_idx in selected_file_indices:
        file_name = tfrecord_files[file_idx]
        file_path = os.path.join(dataset_path, file_name)
        logger.info(f"processing file {file_idx+1}/{total_tfrecords}: {file_name}")

        try:
            raw_dataset = tf.data.TFRecordDataset(file_path)
            parsed_dataset = raw_dataset.map(parse_tfrecord_example)

            local_episode_idx = 0
            for episode in parsed_dataset:
                if shard_prefix is not None:
                    split_episode_idx = shard_prefix[file_idx] + local_episode_idx
                elif episodes_per_tfrecord > 0:
                    split_episode_idx = file_idx * episodes_per_tfrecord + local_episode_idx
                else:
                    split_episode_idx = fallback_episode_idx
                    fallback_episode_idx += 1
                global_episode_idx = split_episode_idx + episode_index_offset

                if end_episode is not None and global_episode_idx >= end_episode:
                    logger.info(f"reached end episode {end_episode}; stopping")
                    logger.info(f"streaming load finished; yielded {yielded_episodes} episodes")
                    return

                if global_episode_idx < start_episode:
                    local_episode_idx += 1
                    continue

                if target_episode_set is not None and global_episode_idx not in target_episode_set:
                    local_episode_idx += 1
                    continue

                instructions = episode["steps/language_instruction"].values.numpy()
                if len(instructions) == 0:
                    instructions = episode["steps/observation/natural_language_instruction"].values.numpy()
                language_instruction = instructions[0].decode("utf-8") if len(instructions) > 0 else "N/A"

                main_images = episode["steps/observation/image"].values.numpy()
                if len(main_images) > 0:
                    images = []
                    for img_data in main_images:
                        try:
                            image = Image.open(io.BytesIO(img_data))
                            if not image.mode == "RGB":
                                image = image.convert("RGB")
                            images.append(image)
                        except Exception as e:
                            logger.warning(f"Episode {global_episode_idx} image decoding failed: {e}")

                    if images:
                        yielded_episodes += 1
                        if yielded_episodes <= 3 or yielded_episodes % 50 == 0:
                            logger.info(
                                f"loaded episode {global_episode_idx}: {len(images)} images "
                                f"(tfrecord={file_idx}, local_episode={local_episode_idx})"
                            )
                        yield (
                            images,
                            global_episode_idx,
                            language_instruction,
                            file_idx,
                            local_episode_idx,
                            file_name,
                        )

                local_episode_idx += 1

        except Exception as e:
            logger.error(f"error while processing file {file_name}: {e}")
            continue

    logger.info(f"streaming load done; yielded {yielded_episodes} episodes")

def load_episode_data(
    dataset_path: str,
    start_episode: int,
    end_episode: Optional[int],
    start_tfrecord_idx: int = 0,
    end_tfrecord_idx: Optional[int] = None,
    episodes_per_tfrecord: int = 80,
    target_episode_set: Optional[Set[int]] = None,
    split_name: str = "train",
    episode_index_offset: int = 0,
):
    """Compat shim: returns a list."""
    episodes = []
    for item in iter_episode_data(
        dataset_path,
        start_episode,
        end_episode,
        start_tfrecord_idx=start_tfrecord_idx,
        end_tfrecord_idx=end_tfrecord_idx,
        episodes_per_tfrecord=episodes_per_tfrecord,
        target_episode_set=target_episode_set,
        split_name=split_name,
        episode_index_offset=episode_index_offset,
    ):
        images, episode_idx, language_instruction, *_ = item
        episodes.append((images, episode_idx, language_instruction))
    return episodes

def infer_run_tag(explicit_tag: Optional[str]) -> str:
    """Derive a run identifier used to name result files and shards."""
    if explicit_tag:
        return explicit_tag

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"job_{slurm_job_id}"

    return f"run_{int(time.time())}"

def flush_results_shard(
    output_dir: str,
    run_tag: str,
    shard_index: int,
    shard_start_episode: Optional[int],
    shard_end_episode: Optional[int],
    shard_results: List[Dict],
    manifest_file: str,
) -> Optional[str]:
    """Flush the in-memory cache as an independent JSON shard and append a manifest entry."""
    if not shard_results:
        return None
    if shard_start_episode is None or shard_end_episode is None:
        return None

    shard_dir = os.path.join(output_dir, "json_shards")
    os.makedirs(shard_dir, exist_ok=True)

    shard_file = os.path.join(
        shard_dir,
        f"{run_tag}_shard_{shard_index:05d}_ep{shard_start_episode:06d}_{shard_end_episode:06d}.json"
    )
    with open(shard_file, "w", encoding="utf-8") as f:
        json.dump(shard_results, f, indent=2, ensure_ascii=False)

    manifest_record = {
        "shard_index": shard_index,
        "start_episode": shard_start_episode,
        "end_episode": shard_end_episode,
        "result_count": len(shard_results),
        "file": shard_file,
    }
    with open(manifest_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    logger.info(f"result shard saved: {shard_file} ({len(shard_results)} rows)")
    return shard_file

def main():
    parser = argparse.ArgumentParser(description="Molmo-72B keyframe inference script (V2)")
    parser.add_argument('--dataset', default='/home/s5c/spikefly.s5c/datasets/libero_10_no_noops/1.0.0',
                       help='Raw dataset path')
    parser.add_argument('--output_dir', default='/home/s5c/spikefly.s5c/molmoact/output_dir',
                       help='Output directory')
    parser.add_argument('--model_path', default='/home/s5c/spikefly.s5c/ckpts/molmo',
                       help='model path')
    parser.add_argument('--prompt', default='point to the robot gripper',
                       help='inference prompt')
    parser.add_argument('--use_cpu', action='store_true',
                       help='Force CPU mode')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Episodes between intermediate result saves')
    parser.add_argument('--start_episode', type=int, default=0,
                       help='First episode index to process')
    parser.add_argument('--end_episode', type=int, default=None,
                       help='Last episode index to process (None = run to end)')
    parser.add_argument('--start_tfrecord_idx', type=int, default=0,
                       help='First tfrecord index (inclusive)')
    parser.add_argument('--end_tfrecord_idx', type=int, default=None,
                       help='Last tfrecord index (exclusive; None = run to end)')
    parser.add_argument('--episode_list_file', type=str, default=None,
                       help='Path to an episode-list file (supports comments; first int column per line)')
    parser.add_argument('--episodes_per_tfrecord', type=int, default=80,
                       help='Episodes per tfrecord (used to compute global episode_idx; default 80)')
    parser.add_argument('--split_name', type=str, default='train', choices=['train', 'val'],
                       help="BC-Z split name; 'val' is used to append validation traces")
    parser.add_argument('--episode_index_offset', type=int, default=0,
                       help='Output episode_idx offset (recommend 39350 for BC-Z val so it follows train)')
    parser.add_argument('--total_nodes', type=int, default=1,
                       help='Total node count (used for automatic node-to-episode assignment)')
    parser.add_argument('--current_node', type=int, default=0,
                       help='Current node ID (0 to total_nodes-1)')
    parser.add_argument('--keyframe_batch_size', type=int, default=4,
                       help='Keyframe batch inference size')
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
                       help='Maximum tokens per generation (smaller saves memory)')
    parser.add_argument('--save_animations', action='store_true',
                       help='Whether to save per-episode GIF / MP4 visualisations (off by default)')
    parser.add_argument('--run_tag', type=str, default=None,
                       help='Run identifier used to name result files and shards')
    parser.add_argument('--disable_shard_save', action='store_true',
                       help='Disable shard saving (enabled by default)')
    parser.add_argument('--shard_save_interval', type=int, default=None,
                       help='Shard save interval in episodes (defaults to save_interval)')
    parser.add_argument('--log_frame_details', action='store_true',
                       help='Log every keyframe coordinate (off by default to avoid log bloat)')
    parser.add_argument('--log_every_n_episodes', type=int, default=10,
                       help='Episodes between progress summary prints (default 10)')
    parser.add_argument('--enable_tqdm', action='store_true',
                       help='Enable live tqdm progress bar (off by default to reduce log I/O)')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # TF is only used to read TFRecords; hide its GPU devices to avoid PyTorch OOM.
    disable_tensorflow_gpu()
    
    target_episode_list: Optional[List[int]] = None
    target_episode_set: Optional[Set[int]] = None
    if args.episode_list_file:
        target_episode_list = load_episode_list_file(args.episode_list_file)
        target_episode_set = set(target_episode_list)
        logger.info(
            "episode-list mode: %d target episodes; range [%d, %d]",
            len(target_episode_list),
            target_episode_list[0],
            target_episode_list[-1],
        )

    if args.total_nodes > 1 and args.end_episode is None:
        try:
            total_episodes = get_total_episodes(args.dataset, args.split_name)
            logger.info(f"detected total episode count: {total_episodes}")
            
            if total_episodes < args.total_nodes:
                logger.warning(f"total episode count ({total_episodes}) is less than node count ({args.total_nodes})")
                if args.current_node >= total_episodes:
                    logger.info(f"node {args.current_node} has no data to process; exiting")
                    return
                else:
                    args.start_episode = args.episode_index_offset + args.current_node
                    args.end_episode = args.episode_index_offset + args.current_node + 1
            else:
                episodes_per_node = total_episodes // args.total_nodes
                remainder = total_episodes % args.total_nodes
                
                if args.current_node < remainder:
                    start_episode = args.current_node * (episodes_per_node + 1)
                    end_episode = start_episode + episodes_per_node + 1
                else:
                    start_episode = args.current_node * episodes_per_node + remainder
                    end_episode = start_episode + episodes_per_node
                
                args.start_episode = args.episode_index_offset + start_episode
                args.end_episode = args.episode_index_offset + end_episode
            
            logger.info(f"node {args.current_node}/{args.total_nodes-1} processes episode range: {args.start_episode} to {args.end_episode-1}")
            
        except Exception as e:
            logger.error(f"failed to get total episode count: {e}")
            logger.error("falling back to single-node mode")
            args.total_nodes = 1
            args.current_node = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load the model
    logger.info("loading model...")
    model, processor = load_model(args.model_path, use_cpu=args.use_cpu)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if args.end_episode is not None and args.end_episode <= args.start_episode:
        logger.error(f"invalid episode range: start_episode={args.start_episode}, end_episode={args.end_episode}")
        return

    expected_episodes = None
    if target_episode_set is not None:
        expected_episodes = len(target_episode_set)
    elif args.end_episode is not None:
        expected_episodes = args.end_episode - args.start_episode
    elif args.end_tfrecord_idx is not None and args.episodes_per_tfrecord > 0:
        tfrecord_span = max(0, args.end_tfrecord_idx - args.start_tfrecord_idx)
        expected_episodes = tfrecord_span * args.episodes_per_tfrecord

    logger.info(
        f"streaming episode processing started; range: {args.start_episode} to "
        f"{(args.end_episode - 1) if args.end_episode is not None else 'end'}"
    )
    episode_iter = iter_episode_data(
        args.dataset,
        args.start_episode,
        args.end_episode,
        start_tfrecord_idx=args.start_tfrecord_idx,
        end_tfrecord_idx=args.end_tfrecord_idx,
        episodes_per_tfrecord=args.episodes_per_tfrecord,
        target_episode_set=target_episode_set,
        split_name=args.split_name,
        episode_index_offset=args.episode_index_offset,
    )
    total_start_time = time.time()

    all_results = []
    episode_count = 0
    processed_episode_count = 0
    seen_episode = False

    pbar = tqdm(
        total=expected_episodes,
        desc="processing episodes",
        unit="episode",
        disable=not args.enable_tqdm,
    )
    
    run_tag = infer_run_tag(args.run_tag)
    range_end_tag = "end" if args.end_episode is None else f"{args.end_episode - 1:06d}"
    tfrecord_end_tag = "end" if args.end_tfrecord_idx is None else f"{args.end_tfrecord_idx - 1:04d}"
    results_file = os.path.join(
        args.output_dir,
        f"keyframe_results_{args.split_name}_{run_tag}_tf{args.start_tfrecord_idx:04d}_{tfrecord_end_tag}_ep{args.start_episode:06d}_{range_end_tag}.json",
    )
    progress_file = os.path.join(args.output_dir, f"{run_tag}_progress.json")
    shard_manifest_file = os.path.join(args.output_dir, f"{run_tag}_shards_manifest.jsonl")
    enable_shard_save = not args.disable_shard_save
    shard_save_interval = args.shard_save_interval if (args.shard_save_interval and args.shard_save_interval > 0) else args.save_interval

    shard_results_buffer: List[Dict] = []
    shard_start_episode: Optional[int] = None
    shard_end_episode: Optional[int] = None
    shard_index = 0
    animation_output_dir = args.output_dir if args.save_animations else None

    if enable_shard_save:
        logger.info(
            f"shard saving enabled: interval={shard_save_interval}, manifest={shard_manifest_file}"
        )
    
    for episode_data in episode_iter:
        seen_episode = True
        processed_episode_count += 1
        pbar.update(1)
        (
            episode_images,
            episode_idx,
            language_instruction,
            tfrecord_idx,
            tfrecord_episode_idx,
            tfrecord_name,
        ) = episode_data
        episode_start_time = time.time()
        
        try:
            episode_results = process_episode_keyframes(
                episode_images, episode_idx, language_instruction, 
                model,
                processor,
                args.prompt,
                args.keyframe_batch_size,
                args.max_new_tokens,
                args.log_frame_details,
                animation_output_dir,
            )

            for result in episode_results:
                result["tfrecord_idx"] = tfrecord_idx
                result["tfrecord_episode_idx"] = tfrecord_episode_idx
                result["tfrecord_name"] = tfrecord_name

            all_results.extend(episode_results)
            if enable_shard_save and episode_results:
                if shard_start_episode is None:
                    shard_start_episode = episode_idx
                shard_end_episode = episode_idx
                shard_results_buffer.extend(episode_results)
            
            episode_count += 1
            episode_time = time.time() - episode_start_time
            
            if args.enable_tqdm:
                pbar.set_postfix({
                    'Episode': f"{episode_idx}",
                    'Frames': f"{len(episode_images)}",
                    'Time': f"{episode_time:.1f}s"
                })

            if args.log_every_n_episodes > 0 and processed_episode_count % args.log_every_n_episodes == 0:
                avg_processed_episode_time = (time.time() - total_start_time) / max(1, processed_episode_count)
                logger.info(
                    "progress: processed=%d successful=%d current_episode=%d tfrecord=%d local_episode=%d avg_episode_time=%.2fs",
                    processed_episode_count,
                    episode_count,
                    episode_idx,
                    tfrecord_idx,
                    tfrecord_episode_idx,
                    avg_processed_episode_time,
                )
            
            if processed_episode_count % shard_save_interval == 0:
                if enable_shard_save:
                    shard_file = flush_results_shard(
                        output_dir=args.output_dir,
                        run_tag=run_tag,
                        shard_index=shard_index,
                        shard_start_episode=shard_start_episode,
                        shard_end_episode=shard_end_episode,
                        shard_results=shard_results_buffer,
                        manifest_file=shard_manifest_file,
                    )
                    if shard_file:
                        shard_index += 1
                        shard_results_buffer = []
                        shard_start_episode = None
                        shard_end_episode = None

                progress = {
                    "run_tag": run_tag,
                    "processed_episodes": processed_episode_count,
                    "successful_episodes": episode_count,
                    "last_episode_idx": episode_idx,
                    "last_tfrecord_idx": tfrecord_idx,
                    "last_tfrecord_name": tfrecord_name,
                    "last_tfrecord_episode_idx": tfrecord_episode_idx,
                    "total_results_so_far": len(all_results),
                    "output_file": results_file,
                    "shard_manifest_file": shard_manifest_file if enable_shard_save else None,
                    "updated_at": int(time.time()),
                }
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)
                logger.info(f"progress updated: {progress_file}")
        
        except Exception as e:
            logger.error(f"error while processing episode {episode_idx}: {e}")
            continue
    
    pbar.close()
    if not seen_episode:
        logger.error("no episode data was found")
        return

    if enable_shard_save:
        shard_file = flush_results_shard(
            output_dir=args.output_dir,
            run_tag=run_tag,
            shard_index=shard_index,
            shard_start_episode=shard_start_episode,
            shard_end_episode=shard_end_episode,
            shard_results=shard_results_buffer,
            manifest_file=shard_manifest_file,
        )
        if shard_file:
            shard_index += 1
    
    total_time = time.time() - total_start_time
    total_successful = sum(1 for r in all_results if r['molmo_coords'] is not None)
    total_frames = len(all_results)
    avg_episode_time = total_time / episode_count if episode_count > 0 else 0
    avg_frame_time = total_time / total_frames if total_frames > 0 else 0
    throughput = total_frames / total_time if total_time > 0 else 0
    
    print(f"\n" + "="*60)
    print("Keyframe processing summary")
    print("="*60)
    print(f"processed episodes: {processed_episode_count} (successful {episode_count})")
    print(f"total frames: {total_frames}")
    print(f"successful: {total_successful}")
    print(f"total elapsed: {total_time:.2f}s")
    print(f"avg per episode: {avg_episode_time:.2f}s")
    print(f"avg per frame: {avg_frame_time:.2f}s")
    print(f"throughput: {throughput:.2f} frames/s")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"final result saved: {results_file}")
    if enable_shard_save:
        print(f"shard manifest: {shard_manifest_file}")
        print(f"shard directory: {os.path.join(args.output_dir, 'json_shards')}")
    
    keyframe_results = [r for r in all_results if r.get('is_keyframe', False)]
    keyframe_ratio = (len(keyframe_results) / total_frames * 100) if total_frames > 0 else 0
    print(f"\nKeyframe summary:")
    print(f"keyframes: {len(keyframe_results)}")
    print(f"keyframe ratio: {keyframe_ratio:.1f}%")
    
    logger.info("all processing done")

def create_episode_animations(episode_images: List[Image.Image], 
                             key_positions: List[Tuple[int, float, float]], 
                             episode_idx: int, 
                             output_dir: str):
    """Build an animation for a single episode."""
    try:
        episode_output_dir = os.path.join(output_dir, f"episode_{episode_idx}")
        os.makedirs(episode_output_dir, exist_ok=True)
        
        sorted_positions = sorted(key_positions, key=lambda x: x[0])
        
        create_keyframe_animation(episode_images, sorted_positions, episode_output_dir, episode_idx)
        
        create_trajectory_animation(episode_images[0], sorted_positions, episode_output_dir, episode_idx)
        
        logger.info(f"Episode {episode_idx}: animation created")
        
    except Exception as e:
        logger.error(f"Episode {episode_idx}: animation creation failed: {e}")

def create_keyframe_animation(episode_images: List[Image.Image], 
                             key_positions: List[Tuple[int, float, float]], 
                             output_dir: str, 
                             episode_id: int):
    """Create keyframe animations (GIF and video)."""
    try:
        animation_frames = []
        
        for frame_idx, molmo_x, molmo_y in key_positions:
            if frame_idx < len(episode_images):
                frame = episode_images[frame_idx].copy()
                
                if frame.size != (224, 224):
                    frame = frame.resize((224, 224), Image.Resampling.LANCZOS)
                
                draw = ImageDraw.Draw(frame)
                pixel_x, pixel_y = molmo_to_pixel_coords(molmo_x, molmo_y, 224, 224)
                
                point_size = 12
                draw.ellipse([pixel_x - point_size//2, pixel_y - point_size//2, 
                             pixel_x + point_size//2, pixel_y + point_size//2], 
                             fill='red', outline='white', width=3)
                
                cross_size = 8
                draw.line([pixel_x - cross_size, pixel_y, pixel_x + cross_size, pixel_y], 
                         fill='red', width=3)
                draw.line([pixel_x, pixel_y - cross_size, pixel_x, pixel_y + cross_size], 
                         fill='red', width=3)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                frame_text = f"Frame {frame_idx}"
                coord_text = f"({molmo_x:.1f}, {molmo_y:.1f})"
                
                text_bbox = draw.textbbox((0, 0), frame_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle([10, 10, 10 + text_width + 10, 10 + text_height * 2 + 20], 
                              fill=(0, 0, 0, 180))
                
                draw.text((15, 15), frame_text, fill='white', font=font)
                draw.text((15, 15 + text_height + 5), coord_text, fill='yellow', font=font)
                
                animation_frames.append(frame)
        
        if not animation_frames:
            logger.warning(f"Episode {episode_id}: no animation frames available")
            return
        
        gif_path = os.path.join(output_dir, f"episode_{episode_id}_keyframes_animation.gif")
        animation_frames[0].save(
            gif_path,
            save_all=True,
            append_images=animation_frames[1:],
            duration=1000,
            loop=0,
            optimize=True
        )
        logger.info(f"Episode {episode_id}: GIF animation saved to {gif_path}")
        
        try:
            import cv2
            video_path = os.path.join(output_dir, f"episode_{episode_id}_keyframes_animation.mp4")
            
            height, width = animation_frames[0].size[1], animation_frames[0].size[0]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 1.0, (width, height))  # 1 FPS
            
            for frame in animation_frames:
                frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(frame_cv)
            
            out.release()
            logger.info(f"Episode {episode_id}: MP4 video saved to {video_path}")
            
        except ImportError:
            logger.warning("OpenCV not installed; skipping MP4 video creation")
        except Exception as e:
            logger.error(f"MP4 video creation failed: {e}")
            
    except Exception as e:
        logger.error(f"keyframe animation creation failed: {e}")

def create_trajectory_animation(first_frame: Image.Image, 
                               key_positions: List[Tuple[int, float, float]], 
                               output_dir: str, 
                               episode_id: int):
    """Build a step-by-step trajectory animation."""
    try:
        if first_frame.size != (224, 224):
            first_frame = first_frame.resize((224, 224), Image.Resampling.LANCZOS)
        
        sorted_positions = sorted(key_positions, key=lambda x: x[0])
        
        trajectory_frames = []
        
        for i in range(len(sorted_positions) + 1):
            frame = first_frame.copy()
            draw = ImageDraw.Draw(frame)
            
            if i > 0:
                for j in range(i):
                    if j < len(sorted_positions) - 1:
                        start_frame_idx, start_x, start_y = sorted_positions[j]
                        end_frame_idx, end_x, end_y = sorted_positions[j + 1]
                        
                        start_pixel = molmo_to_pixel_coords(start_x, start_y, 224, 224)
                        end_pixel = molmo_to_pixel_coords(end_x, end_y, 224, 224)
                        
                        draw.line([start_pixel, end_pixel], fill='blue', width=4)
            
            for j in range(i):
                if j < len(sorted_positions):
                    frame_idx, molmo_x, molmo_y = sorted_positions[j]
                    pixel_x, pixel_y = molmo_to_pixel_coords(molmo_x, molmo_y, 224, 224)
                    
                    if j == 0:
                        color = 'green'
                    elif j == len(sorted_positions) - 1:
                        color = 'red'
                    else:
                        color = 'yellow'
                    
                    point_size = 8
                    draw.ellipse([pixel_x - point_size//2, pixel_y - point_size//2, 
                                 pixel_x + point_size//2, pixel_y + point_size//2], 
                                 fill=color, outline='white', width=2)
                    
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                    except:
                        font = ImageFont.load_default()
                    
                    text = str(j + 1)
                    text_x = pixel_x + point_size//2 + 3
                    text_y = pixel_y - point_size//2 - 3
                    draw.text((text_x, text_y), text, fill='white', font=font)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            progress_text = f"Progress: {i}/{len(sorted_positions)}"
            draw.rectangle([10, 224 - 30, 200, 224 - 10], fill=(0, 0, 0, 180))
            draw.text((15, 224 - 25), progress_text, fill='white', font=font)
            
            trajectory_frames.append(frame)
        
        trajectory_gif_path = os.path.join(output_dir, f"episode_{episode_id}_trajectory_animation.gif")
        trajectory_frames[0].save(
            trajectory_gif_path,
            save_all=True,
            append_images=trajectory_frames[1:],
            duration=800,
            loop=0,
            optimize=True
        )
        logger.info(f"Episode {episode_id}: trajectory GIF saved to {trajectory_gif_path}")
        
    except Exception as e:
        logger.error(f"trajectory animation creation failed: {e}")

def molmo_to_pixel_coords(molmo_x: float, molmo_y: float, img_width: int = 224, img_height: int = 224) -> Tuple[int, int]:
    """Convert Molmo coordinates to pixel coordinates."""
    x_255 = molmo_x * 2.55
    y_255 = molmo_y * 2.55
    
    x_pixel = int(round(x_255 / 255.0 * (img_width - 1)))
    y_pixel = int(round(y_255 / 255.0 * (img_height - 1)))
    
    x_pixel = max(0, min(x_pixel, img_width - 1))
    y_pixel = max(0, min(y_pixel, img_height - 1))
    
    return (x_pixel, y_pixel)

if __name__ == "__main__":
    main()
