#!/usr/bin/env python3
"""
Minimal SemanticVLA offline inference example (no websocket, no control loop).

Usage:
  python deployment/model_server/predict_action_offline.py \
      --run-dir <CHECKPOINT_DIR> \
      --image  path/to/frame.png \
      --task   "pick up the red block"

Output:
  - A numpy array of shape (H, A): action chunk of H steps, each A-dim
    (joints + gripper for the released SemanticVLA policies).
  - Already un-normalised to the training joint units (not [-1, 1]).

What this script does NOT do:
  - It does not connect to a real robot or send control commands.
  - It does not perform per-robot joint remapping; remap downstream if
    your target embodiment differs from the training one.
  - It does not maintain episode state; every call is treated as the
    first step of a new episode.

Dependencies: the SemanticVLA repo is cloned and its conda env is set up
(torch 2.6.0+cu126, transformers, omegaconf, PIL, numpy). Set the BASE_VLM
environment variable to a local Qwen3-VL-4B-Instruct checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

# Allow `import semanticvla.*` from this script (repo root is two levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semanticvla.model.framework import build_framework  # noqa: E402


def load_model(run_dir: Path, ckpt_path: Path | None, device: str = "cuda"):
    cfg_path = run_dir / "config.yaml"
    if ckpt_path is None:
        # Prefer final_model/; otherwise pick the highest-step file under checkpoints/.
        if (run_dir / "final_model" / "pytorch_model.pt").is_file():
            ckpt_path = run_dir / "final_model" / "pytorch_model.pt"
        else:
            cands = sorted((run_dir / "checkpoints").glob("steps_*_pytorch_model.pt"))
            if not cands:
                raise FileNotFoundError(f"No checkpoint under {run_dir}/checkpoints/")
            ckpt_path = cands[-1]
    cfg = OmegaConf.load(cfg_path)
    cfg.trainer.pretrained_checkpoint = None
    cfg.trainer.is_resume = False

    model = build_framework(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ckpt_path


def load_action_minmax(run_dir: Path):
    """Read action.joints min / max from dataset_statistics.json."""
    stats_path = run_dir / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text())
    # Top level is unnorm_key, then action / state, then per-modality key ("joints").
    if len(stats) != 1:
        raise ValueError(
            f"Expected exactly one unnorm_key in {stats_path}, got: {list(stats.keys())}"
        )
    [(_unnorm_key, body)] = stats.items()
    a = body["action"]["joints"]
    return np.asarray(a["min"], dtype=np.float32), np.asarray(a["max"], dtype=np.float32)


def unnorm_min_max(normalized: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Un-normalise from [-1, 1] to [lo, hi]."""
    return (normalized + 1.0) * 0.5 * (hi - lo) + lo


def predict_one_step(
    model,
    image: Image.Image,
    task: str,
    *,
    step: int = 0,
    sample_seed: int = 0,
) -> np.ndarray:
    """Single-step inference; returns normalized_actions of shape [H, A] in [-1, 1]."""
    img_224 = image.convert("RGB").resize((224, 224))
    with torch.inference_mode():
        out = model.predict_action(
            [[img_224]],            # batch_images: [batch=1][n_views=1] PIL
            [task],                 # instructions: list of length 1
            state=None,             # if config has include_state=false
            do_sample=False,        # deterministic deployment
            sample_seed=int(sample_seed),
            step=int(step),
        )
    arr = np.asarray(out["normalized_actions"], dtype=np.float32)
    if arr.shape[0] != 1:
        raise RuntimeError(f"expected batch=1, got shape {arr.shape}")
    return arr[0]   # [H, A]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Checkpoint directory containing config.yaml + dataset_statistics.json + checkpoints/")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Explicit .pt path (default: pytorch_model.pt or highest-step file)")
    parser.add_argument("--image", type=Path, required=True,
                        help="An RGB image (any size; resized to 224x224)")
    parser.add_argument("--task", type=str, required=True,
                        help="Language instruction matching the training distribution.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    model, cfg, ckpt_path = load_model(args.run_dir, args.ckpt, device=args.device)
    print(f"[OK] loaded {ckpt_path}", flush=True)
    print(f"     framework={cfg.framework.name}", flush=True)
    print(f"     action_horizon={cfg.framework.action_model.action_horizon}", flush=True)

    img = Image.open(args.image)
    normalized = predict_one_step(model, img, args.task, step=0)
    print(f"     normalized_actions shape={normalized.shape} range=[{normalized.min():.3f}, {normalized.max():.3f}]")

    lo, hi = load_action_minmax(args.run_dir)
    raw = unnorm_min_max(normalized, lo, hi)
    print(f"     raw_actions       shape={raw.shape}")
    print(f"     action_min        ={lo.tolist()}")
    print(f"     action_max        ={hi.tolist()}")
    print()
    print("=== Action chunk (raw joint units) ===")
    header_cols = [f"a_{i}" for i in range(raw.shape[1])]
    print("step  " + "  ".join(f"{c:>8s}" for c in header_cols))
    for k, row in enumerate(raw):
        vals = "  ".join(f"{v:8.4f}" for v in row)
        print(f"{k:4d}  {vals}")


if __name__ == "__main__":
    main()
