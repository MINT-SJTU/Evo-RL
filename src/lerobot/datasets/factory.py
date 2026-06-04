#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.rl.acp_instance_table import ACPInstanceTableDataset
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts if len(parts) > 1 else None


def _dataset_root_for_multi(root: str | Path | None, repo_id: str) -> Path | None:
    if root is None:
        return None
    return Path(root) / repo_id


def _repo_id_from_root(root: Path) -> str:
    return f"local/{root.name}"


def _parse_dataset_roots(root: str | None) -> Path | list[Path] | None:
    roots = _split_csv(root)
    if roots is None:
        return Path(root) if root is not None else None
    return [Path(dataset_root) for dataset_root in roots]


def _parse_repo_ids(repo_id: str | list[str]) -> str | list[str]:
    if isinstance(repo_id, str):
        parsed_repo_ids = [part.strip() for part in repo_id.split(",") if part.strip()]
        if len(parsed_repo_ids) > 1:
            return parsed_repo_ids
    return repo_id


def _resolve_multi_dataset_ids_and_roots(
    repo_id: str | list[str],
    root: str | None,
) -> tuple[str | list[str], Path | list[Path] | None]:
    """Resolve logical dataset ids and physical roots.

    Comma-separated ``dataset.root`` values are treated as exact dataset roots. In that form, a single
    ``dataset.repo_id`` is only a placeholder and stable local ids are derived from root basenames.
    The older ``root=/parent`` + ``repo_id=ds0,ds1`` form is kept for compatibility.
    """
    parsed_repo_id = _parse_repo_ids(repo_id)
    parsed_root = _parse_dataset_roots(root)

    if isinstance(parsed_root, list):
        if isinstance(parsed_repo_id, list):
            if len(parsed_repo_id) != len(parsed_root):
                parsed_repo_id = [_repo_id_from_root(dataset_root) for dataset_root in parsed_root]
        else:
            parsed_repo_id = [_repo_id_from_root(dataset_root) for dataset_root in parsed_root]
        return parsed_repo_id, parsed_root

    return parsed_repo_id, parsed_root


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    repo_id, dataset_root = _resolve_multi_dataset_ids_and_roots(cfg.dataset.repo_id, cfg.dataset.root)
    cfg.dataset.repo_id = repo_id

    if isinstance(repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            repo_id, root=dataset_root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                repo_id,
                root=dataset_root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        if cfg.dataset.streaming:
            raise NotImplementedError("Multi-dataset streaming is not supported on this branch.")

        dataset_roots = dataset_root if isinstance(dataset_root, list) else None
        root_parent = None if isinstance(dataset_root, list) else dataset_root
        metas = [
            LeRobotDatasetMetadata(
                dataset_repo_id,
                root=dataset_roots[i]
                if dataset_roots is not None
                else _dataset_root_for_multi(root_parent, dataset_repo_id),
                revision=cfg.dataset.revision,
            )
            for i, dataset_repo_id in enumerate(repo_id)
        ]
        delta_timestamps = resolve_delta_timestamps(cfg.policy, metas[0])
        dataset = MultiLeRobotDataset(
            repo_id,
            root=root_parent,
            roots=dataset_roots,
            episodes=cfg.dataset.episodes if isinstance(cfg.dataset.episodes, dict) else None,
            delta_timestamps=delta_timestamps,
            tolerances_s={dataset_repo_id: cfg.tolerance_s for dataset_repo_id in repo_id},
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    acp_cfg = getattr(cfg, "acp", None)
    instance_table_path = getattr(acp_cfg, "instance_table_path", None)
    if instance_table_path:
        if cfg.dataset.streaming:
            raise ValueError("ACP instance tables are not supported with streaming datasets.")
        drop_n_last_frames = int(getattr(cfg.policy, "drop_n_last_frames", 0))
        dataset = ACPInstanceTableDataset(
            base_dataset=dataset,
            instance_table_path=instance_table_path,
            drop_n_last_frames=drop_n_last_frames,
        )

    return dataset
