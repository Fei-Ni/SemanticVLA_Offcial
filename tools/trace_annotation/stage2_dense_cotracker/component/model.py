"""Model loading utilities for the refactored CoTracker pipeline."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def load_cotracker_model(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Load the CoTracker3 offline model via torch.hub.

    Parameters
    ----------
    device:
        Optional device to move the model to. If omitted the caller should manage
        device placement manually.

    Returns
    ----------
    torch.nn.Modeule or None
        The loaded Cotracker model if successfully loaded
    """
    try:
        model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline")
        # First call downloads the repo from GitHub and caches the model locally; subsequent loads use the local cache.
        if device is not None:
            model = model.to(device)  # move model to the specified device
        logger.info("CoTracker model loaded")
        return model
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.error("CoTracker load failed: %s", exc)
        return None
