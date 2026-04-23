#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.acp_tags import build_acp_tagged_task

ACP_INSTANCE_ROLE_FIELD = "prompt_role"
ACP_INSTANCE_SOURCE_INDEX_FIELD = "source_index"
ACP_INSTANCE_SOURCE_EPISODE_FIELD = "source_episode_index"
ACP_INSTANCE_SOURCE_FRAME_FIELD = "source_frame_index"
ACP_INSTANCE_SOURCE_ADVANTAGE_FIELD = "source_advantage"
ACP_INSTANCE_DUPLICATED_FIELD = "is_duplicate"

ACP_ROLE_UNLABELED = "unlabeled"
ACP_ROLE_POSITIVE = "positive"
ACP_ROLE_NEGATIVE = "negative"
ACP_VALID_ROLES = {ACP_ROLE_UNLABELED, ACP_ROLE_POSITIVE, ACP_ROLE_NEGATIVE}


def _flatten_float_column(values: Sequence[Any], field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2 and array.shape[1] == 1:
        return array.reshape(-1).astype(np.float32, copy=False)
    raise ValueError(f"Field '{field_name}' must have shape [N] or [N, 1], got {array.shape}.")


def build_acp_instance_table(
    source_indices: np.ndarray,
    source_episode_indices: np.ndarray,
    source_frame_indices: np.ndarray,
    advantages: np.ndarray,
    negative_bottom_ratio: float,
    positive_top_duplicate_ratio: float,
) -> pa.Table:
    if not 0.0 <= negative_bottom_ratio <= 1.0:
        raise ValueError("'negative_bottom_ratio' must be within [0, 1].")
    if not 0.0 <= positive_top_duplicate_ratio <= 1.0:
        raise ValueError("'positive_top_duplicate_ratio' must be within [0, 1].")
    if not (
        source_indices.shape
        == source_episode_indices.shape
        == source_frame_indices.shape
        == advantages.shape
    ):
        raise ValueError("All ACP instance table inputs must share the same shape.")

    num_rows = int(source_indices.shape[0])
    if num_rows == 0:
        raise ValueError("Cannot build ACP instance table from an empty dataset.")

    negative_threshold = float(np.quantile(advantages, negative_bottom_ratio))
    positive_threshold = float(np.quantile(advantages, 1.0 - positive_top_duplicate_ratio))

    base_roles = np.full(num_rows, ACP_ROLE_UNLABELED, dtype=object)
    if negative_bottom_ratio > 0.0:
        base_roles[advantages <= negative_threshold] = ACP_ROLE_NEGATIVE

    duplicate_mask = np.zeros(num_rows, dtype=np.bool_)
    if positive_top_duplicate_ratio > 0.0:
        duplicate_mask = advantages >= positive_threshold

    duplicated_count = int(np.sum(duplicate_mask))

    all_source_indices = np.concatenate([source_indices, source_indices[duplicate_mask]])
    all_episode_indices = np.concatenate([source_episode_indices, source_episode_indices[duplicate_mask]])
    all_frame_indices = np.concatenate([source_frame_indices, source_frame_indices[duplicate_mask]])
    all_advantages = np.concatenate([advantages, advantages[duplicate_mask]])
    all_roles = np.concatenate(
        [
            base_roles,
            np.full(duplicated_count, ACP_ROLE_POSITIVE, dtype=object),
        ]
    )
    is_duplicate = np.concatenate(
        [
            np.zeros(num_rows, dtype=np.bool_),
            np.ones(duplicated_count, dtype=np.bool_),
        ]
    )

    return pa.table(
        {
            ACP_INSTANCE_SOURCE_INDEX_FIELD: pa.array(all_source_indices.astype(np.int64, copy=False)),
            ACP_INSTANCE_SOURCE_EPISODE_FIELD: pa.array(all_episode_indices.astype(np.int64, copy=False)),
            ACP_INSTANCE_SOURCE_FRAME_FIELD: pa.array(all_frame_indices.astype(np.int64, copy=False)),
            ACP_INSTANCE_SOURCE_ADVANTAGE_FIELD: pa.array(all_advantages.astype(np.float32, copy=False)),
            ACP_INSTANCE_ROLE_FIELD: pa.array(all_roles.tolist(), type=pa.string()),
            ACP_INSTANCE_DUPLICATED_FIELD: pa.array(is_duplicate.tolist(), type=pa.bool_()),
        }
    )


def write_acp_instance_table(
    dataset: LeRobotDataset,
    output_path: str | Path,
    advantage_field: str,
    negative_bottom_ratio: float,
    positive_top_duplicate_ratio: float,
) -> Path:
    raw_frames = dataset.hf_dataset.with_format(None)
    if advantage_field not in raw_frames.column_names:
        raise KeyError(f"Missing advantage field '{advantage_field}' in dataset.")

    source_indices = np.asarray(raw_frames["index"], dtype=np.int64).reshape(-1)
    source_episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64).reshape(-1)
    source_frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64).reshape(-1)
    advantages = _flatten_float_column(raw_frames[advantage_field], advantage_field)

    table = build_acp_instance_table(
        source_indices=source_indices,
        source_episode_indices=source_episode_indices,
        source_frame_indices=source_frame_indices,
        advantages=advantages,
        negative_bottom_ratio=negative_bottom_ratio,
        positive_top_duplicate_ratio=positive_top_duplicate_ratio,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


def load_acp_instance_table(path: str | Path) -> pa.Table:
    table = pq.read_table(path)
    required_columns = {
        ACP_INSTANCE_SOURCE_INDEX_FIELD,
        ACP_INSTANCE_ROLE_FIELD,
    }
    missing_columns = required_columns.difference(table.column_names)
    if missing_columns:
        raise KeyError(f"ACP instance table is missing required columns: {sorted(missing_columns)}")

    roles = table[ACP_INSTANCE_ROLE_FIELD].to_pylist()
    invalid_roles = sorted({role for role in roles if role not in ACP_VALID_ROLES})
    if invalid_roles:
        raise ValueError(f"ACP instance table contains invalid roles: {invalid_roles}")
    return table


def _compute_valid_absolute_indices(dataset: LeRobotDataset, drop_n_last_frames: int) -> set[int]:
    valid_indices: set[int] = set()
    for start_index, end_index in zip(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        strict=True,
    ):
        effective_end = int(end_index) - max(0, int(drop_n_last_frames))
        valid_indices.update(range(int(start_index), max(int(start_index), effective_end)))
    return valid_indices


class ACPInstanceTableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: LeRobotDataset,
        instance_table_path: str | Path,
        drop_n_last_frames: int = 0,
    ):
        self.base_dataset = base_dataset
        self.meta = base_dataset.meta
        self.root = base_dataset.root
        self.repo_id = base_dataset.repo_id
        self.episodes = base_dataset.episodes
        self.image_transforms = base_dataset.image_transforms
        self.delta_timestamps = base_dataset.delta_timestamps
        self.tolerance_s = base_dataset.tolerance_s
        self.revision = base_dataset.revision
        self.video_backend = base_dataset.video_backend
        self.features = base_dataset.features
        self.instance_table_path = Path(instance_table_path)

        table = load_acp_instance_table(self.instance_table_path)
        source_indices = np.asarray(table[ACP_INSTANCE_SOURCE_INDEX_FIELD].to_numpy(), dtype=np.int64).reshape(-1)
        roles = np.asarray(table[ACP_INSTANCE_ROLE_FIELD].to_pylist(), dtype=object)

        if drop_n_last_frames > 0:
            valid_absolute_indices = _compute_valid_absolute_indices(base_dataset, drop_n_last_frames)
            keep_mask = np.asarray([int(idx) in valid_absolute_indices for idx in source_indices], dtype=np.bool_)
            source_indices = source_indices[keep_mask]
            roles = roles[keep_mask]

        absolute_to_relative = base_dataset._absolute_to_relative_idx
        if absolute_to_relative is None:
            relative_indices = source_indices.tolist()
        else:
            keep_mask = np.asarray([int(idx) in absolute_to_relative for idx in source_indices], dtype=np.bool_)
            source_indices = source_indices[keep_mask]
            roles = roles[keep_mask]
            relative_indices = [absolute_to_relative[int(idx)] for idx in source_indices]

        self._source_indices = source_indices.astype(np.int64, copy=False)
        self._source_relative_indices = np.asarray(relative_indices, dtype=np.int64)
        self._roles = roles.tolist()

    @property
    def num_frames(self) -> int:
        return len(self._source_relative_indices)

    @property
    def num_episodes(self) -> int:
        return self.base_dataset.num_episodes

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = int(self._source_relative_indices[idx])
        role = self._roles[idx]
        item = dict(self.base_dataset[source_idx])
        task = item.get("task")
        if not isinstance(task, str):
            raise TypeError(f"Expected task to be a string, got {type(task).__name__}.")

        if role == ACP_ROLE_POSITIVE:
            item["task"] = build_acp_tagged_task(task, is_positive=True)
        elif role == ACP_ROLE_NEGATIVE:
            item["task"] = build_acp_tagged_task(task, is_positive=False)
        elif role != ACP_ROLE_UNLABELED:
            raise ValueError(f"Unsupported ACP role '{role}'.")

        item["complementary_info.acp_prompt_role"] = role
        item["complementary_info.acp_source_index"] = torch.tensor(
            [int(self._source_indices[idx])], dtype=torch.int64
        )
        return item
