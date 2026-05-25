from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from semanticvla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from semanticvla.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from semanticvla.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from semanticvla.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    video_backend = data_cfg.get("video_backend", "torchvision_av") if data_cfg else "torchvision_av"
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        lerobot_version=lerobot_version,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    if hasattr(data_cfg, "get") and data_cfg.get("delete_pause_frame", None) is not None:
        delete_pause_frame = bool(data_cfg.get("delete_pause_frame"))
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for entry in mixture_spec:
        d_name, d_weight, robot_type = entry[0], entry[1], entry[2]
        d_version = entry[3] if len(entry) > 3 else None
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, d_version))

    dataset_mixture = []
    for d_name, d_weight, robot_type, d_version in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            lerobot_version=d_version,
        ), d_weight))

    dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        max_retries=int(getattr(data_cfg, "sample_max_retries", 10)),
        strict_sample_errors=bool(getattr(data_cfg, "strict_sample_errors", False)),
        sequential_step_sampling=bool(getattr(data_cfg, "sequential_step_sampling", False)),
        action_stats_mask_mode=str(getattr(data_cfg, "action_stats_mask_mode", "gripper_false")),
        **kwargs,
    )

    # SemanticVLA: optionally wrap the mixture with TraceAugmentedDataset so
    # each sample carries `trace_coords_window`. Config-gated, fully no-op
    # when datasets.vla_data.trace is absent or disabled.
    trace_cfg = getattr(data_cfg, "trace", None)
    if trace_cfg is not None and bool(getattr(trace_cfg, "enabled", False)):
        from semanticvla.dataloader.gr00t_lerobot.trace_loader import (
            LiberoTraceManager,
            OxeTraceIndexManager,
            TraceAugmentedDataset,
        )
        from semanticvla.model.modules.action_model.trace_text_codec import (
            get_default_anchor_indices,
        )
        dataset_names = [name for name, _, _, _ in filtered_mixture_spec]
        # optionally sub-sample the W=12 window down to N anchor points.
        # `num_anchor_points` (int) is the the LM-trace; `anchor_indices`
        # (list[int], optional) can override the defaults from trace_text_codec.
        num_anchor_points = int(getattr(trace_cfg, "num_anchor_points", 0)) or None
        explicit_anchors = getattr(trace_cfg, "anchor_indices", None)
        if explicit_anchors is not None:
            anchor_indices = tuple(int(i) for i in explicit_anchors)
        elif num_anchor_points is not None:
            anchor_indices = get_default_anchor_indices(num_anchor_points)
        else:
            anchor_indices = None  # v0 / backward-compat: full 12-frame window
        trace_manager_type = str(getattr(trace_cfg, "manager", getattr(trace_cfg, "format", "libero")))
        if trace_manager_type in {"oxe", "oxe_npy", "npy", "npy_index"}:
            mgr = OxeTraceIndexManager(
                trace_root=trace_cfg.root,
                dataset_names=dataset_names,
                window_size=int(getattr(trace_cfg, "window_size", 12)),
                normalize=bool(getattr(trace_cfg, "normalize", True)),
                anchor_indices=anchor_indices,
            )
        else:
            mgr = LiberoTraceManager(
                trace_root=trace_cfg.root,
                dataset_names=dataset_names,
                window_size=int(getattr(trace_cfg, "window_size", 12)),
                normalize=bool(getattr(trace_cfg, "normalize", True)),
                anchor_indices=anchor_indices,
            )
        print(
            f"[lerobot_datasets] wrapping mixture with TraceAugmentedDataset: {mgr}, "
            f"anchor_indices={anchor_indices}"
        )
        dataset = TraceAugmentedDataset(dataset, mgr)

    # optionally attach precomputed LAM indices as LM-target labels.
    latent_cfg = getattr(data_cfg, "latent_action_labels", None)
    if latent_cfg is not None and bool(getattr(latent_cfg, "enabled", False)):
        from semanticvla.dataloader.gr00t_lerobot.trace_loader import (
            LatentActionAugmentedDataset,
            LatentActionLabelManager,
        )

        dataset_names = [name for name, _, _, _ in filtered_mixture_spec]
        label_root = getattr(latent_cfg, "root")
        variant = getattr(latent_cfg, "variant", None)
        out_key = str(getattr(latent_cfg, "out_key", "latent_action_idx"))
        strict = bool(getattr(latent_cfg, "strict", True))
        missing_policy = str(getattr(latent_cfg, "missing_policy", "error"))
        label_mgr = LatentActionLabelManager(
            label_root=label_root,
            dataset_names=dataset_names,
            variant=variant,
            missing_policy=missing_policy,
        )
        print(
            f"[lerobot_datasets] wrapping mixture with LatentActionAugmentedDataset: "
            f"{label_mgr}, out_key={out_key}, strict={strict}"
        )
        dataset = LatentActionAugmentedDataset(
            dataset,
            label_mgr,
            out_key=out_key,
            strict=strict,
        )

    return dataset

if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./semanticvla/config/training/cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        pass
