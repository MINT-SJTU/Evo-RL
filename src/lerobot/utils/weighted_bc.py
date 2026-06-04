#!/usr/bin/env python

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pa_ds
import torch

from lerobot.configs.train import WeightedBCConfig
from lerobot.datasets.trajectory_type import (
    EPISODE_INDEX_FIELD,
    IS_INTERVENTION_FIELD,
    TRAJECTORY_EXPERT,
    build_episode_trajectory_types,
)

logger = logging.getLogger(__name__)

FRAME_INDEX_FIELD = "frame_index"


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return _to_float(value[0]) if len(value) > 0 else 0.0
    if value is None:
        return 0.0
    return float(value)


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v.item() if hasattr(v, "item") else v) for v in value]
    return [int(value)]


def _to_float_list(value: Any, *, length: int, default: float = 0.0) -> list[float]:
    if value is None:
        return [default] * length
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v.item() if hasattr(v, "item") else _to_float(v)) for v in value]
    return [float(value)] * length


def _batch_column(record_batch, name: str):
    return record_batch.column(record_batch.schema.names.index(name))


class WeightedBCWeights:
    """Compute lightweight sample weights for DAgger-style weighted BC."""

    def __init__(
        self,
        dataset,
        config: WeightedBCConfig,
        *,
        device: torch.device | None = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-6
        self.fps = int(getattr(dataset, "fps"))
        self.pre_intervention_frames = int(round(config.pre_intervention_s * self.fps))

        self.episode_trajectory_types = self._resolve_episode_trajectory_types(dataset)
        self._pre_intervention_windows = self._empty_windows()
        self._intervention_windows = self._empty_windows()

        if self.pre_intervention_frames > 0 or config.intervention_weight != config.model_weight:
            self._build_windows(dataset)

        self._pre_intervention_windows = self._finalize_windows(self._pre_intervention_windows)
        self._intervention_windows = self._finalize_windows(self._intervention_windows)

        num_pre_windows = sum(len(starts) for starts, _ in self._pre_intervention_windows.values())
        num_intervention_windows = sum(len(starts) for starts, _ in self._intervention_windows.values())
        num_expert_episodes = sum(
            trajectory_type == TRAJECTORY_EXPERT
            for trajectory_type in self.episode_trajectory_types.values()
        )
        logger.info(
            "Weighted BC enabled: fps=%d pre_intervention_frames=%d pre_windows=%d "
            "intervention_windows=%d expert_episodes=%d",
            self.fps,
            self.pre_intervention_frames,
            num_pre_windows,
            num_intervention_windows,
            num_expert_episodes,
        )

    @staticmethod
    def _empty_windows() -> defaultdict[int, list[tuple[int, int]]]:
        return defaultdict(list)

    @staticmethod
    def _dataset_specs(dataset) -> list[tuple[Path, dict[int, int]]]:
        base_dataset = getattr(dataset, "base_dataset", dataset)
        sub_datasets = getattr(base_dataset, "_datasets", None)
        if sub_datasets is None:
            return [(Path(base_dataset.root), {})]

        episode_maps = getattr(base_dataset, "_episode_index_maps", [{} for _ in sub_datasets])
        return [(Path(sub_dataset.root), episode_maps[i]) for i, sub_dataset in enumerate(sub_datasets)]

    def _resolve_episode_trajectory_types(self, dataset) -> dict[int, str]:
        episode_types = getattr(dataset, "episode_trajectory_types", None)
        if episode_types is not None:
            return {int(k): str(v) for k, v in episode_types.items()}

        resolved: dict[int, str] = {}
        for root, episode_map in self._dataset_specs(dataset):
            local_types = build_episode_trajectory_types(root)
            for local_ep, trajectory_type in local_types.items():
                global_ep = episode_map.get(int(local_ep), int(local_ep))
                resolved[int(global_ep)] = trajectory_type
        return resolved

    def _build_windows(self, dataset) -> None:
        for root, episode_map in self._dataset_specs(dataset):
            self._scan_dataset_root(root, episode_map)

    def _scan_dataset_root(self, root: Path, episode_map: dict[int, int]) -> None:
        data_dir = root / "data"
        if not data_dir.exists():
            logger.warning("Weighted BC: dataset data directory not found: %s", data_dir)
            return

        data_dataset = pa_ds.dataset(data_dir, format="parquet")
        schema_names = set(data_dataset.schema.names)
        required_columns = {EPISODE_INDEX_FIELD, FRAME_INDEX_FIELD, IS_INTERVENTION_FIELD}
        missing_columns = sorted(required_columns - schema_names)
        if missing_columns:
            logger.warning(
                "Weighted BC: skipping intervention windows for %s; missing columns: %s",
                root,
                missing_columns,
            )
            return

        prev_intervention: dict[int, bool] = {}
        open_intervention_start: dict[int, int] = {}
        last_frame: dict[int, int] = {}

        scanner = data_dataset.scanner(
            columns=[EPISODE_INDEX_FIELD, FRAME_INDEX_FIELD, IS_INTERVENTION_FIELD],
            batch_size=65536,
        )
        for record_batch in scanner.to_batches():
            episode_indices = _batch_column(record_batch, EPISODE_INDEX_FIELD).to_pylist()
            frame_indices = _batch_column(record_batch, FRAME_INDEX_FIELD).to_pylist()
            intervention_values = _batch_column(record_batch, IS_INTERVENTION_FIELD).to_pylist()

            for local_ep_raw, frame_raw, intervention_raw in zip(
                episode_indices,
                frame_indices,
                intervention_values,
                strict=True,
            ):
                local_ep = int(local_ep_raw)
                global_ep = int(episode_map.get(local_ep, local_ep))
                frame_idx = int(frame_raw)
                is_intervention = _to_float(intervention_raw) > 0.5
                was_intervention = prev_intervention.get(global_ep, False)

                last_frame[global_ep] = frame_idx

                if is_intervention and not was_intervention:
                    open_intervention_start[global_ep] = frame_idx
                    if self.pre_intervention_frames > 0:
                        window_start = max(0, frame_idx - self.pre_intervention_frames)
                        if window_start < frame_idx:
                            self._pre_intervention_windows[global_ep].append((window_start, frame_idx))
                elif was_intervention and not is_intervention:
                    start = open_intervention_start.pop(global_ep, frame_idx)
                    if start < frame_idx:
                        self._intervention_windows[global_ep].append((start, frame_idx))

                prev_intervention[global_ep] = is_intervention

        for global_ep, start in open_intervention_start.items():
            end = last_frame.get(global_ep, start) + 1
            if start < end:
                self._intervention_windows[global_ep].append((start, end))

    @staticmethod
    def _finalize_windows(
        windows: defaultdict[int, list[tuple[int, int]]],
    ) -> dict[int, tuple[list[int], list[int]]]:
        finalized: dict[int, tuple[list[int], list[int]]] = {}
        for episode_index, intervals in windows.items():
            if not intervals:
                continue

            merged: list[tuple[int, int]] = []
            for start, end in sorted(intervals):
                if start >= end:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))

            if merged:
                starts, ends = zip(*merged, strict=True)
                finalized[int(episode_index)] = (list(starts), list(ends))
        return finalized

    @staticmethod
    def _in_windows(
        windows: dict[int, tuple[list[int], list[int]]],
        episode_index: int,
        frame_index: int,
    ) -> bool:
        starts_ends = windows.get(int(episode_index))
        if starts_ends is None:
            return False

        starts, ends = starts_ends
        pos = bisect.bisect_right(starts, int(frame_index)) - 1
        return pos >= 0 and int(frame_index) < ends[pos]

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        episode_indices = batch.get(EPISODE_INDEX_FIELD)
        frame_indices = batch.get(FRAME_INDEX_FIELD)

        if episode_indices is None or frame_indices is None:
            batch_size = self._get_batch_size(batch)
            weights = torch.full(
                (batch_size,),
                float(self.config.model_weight),
                device=self.device,
                dtype=torch.float32,
            )
            return weights, {
                "raw_mean_weight": float(self.config.model_weight),
                "raw_min_weight": float(self.config.model_weight),
                "raw_max_weight": float(self.config.model_weight),
                "model_count": batch_size,
                "pre_intervention_count": 0,
                "intervention_count": 0,
                "expert_count": 0,
            }

        episodes = _to_int_list(episode_indices)
        frames = _to_int_list(frame_indices)
        interventions = _to_float_list(batch.get(IS_INTERVENTION_FIELD), length=len(episodes), default=0.0)

        raw_weights: list[float] = []
        counts = {
            "model_count": 0,
            "pre_intervention_count": 0,
            "intervention_count": 0,
            "expert_count": 0,
        }

        for episode_index, frame_index, intervention_value in zip(
            episodes,
            frames,
            interventions,
            strict=True,
        ):
            is_expert_episode = self.episode_trajectory_types.get(int(episode_index)) == TRAJECTORY_EXPERT
            is_intervention = intervention_value > 0.5 or self._in_windows(
                self._intervention_windows,
                episode_index,
                frame_index,
            )

            if is_expert_episode:
                weight = float(self.config.expert_weight)
                counts["expert_count"] += 1
            elif is_intervention:
                weight = float(self.config.intervention_weight)
                counts["intervention_count"] += 1
            elif self._in_windows(self._pre_intervention_windows, episode_index, frame_index):
                weight = float(self.config.pre_intervention_weight)
                counts["pre_intervention_count"] += 1
            else:
                weight = float(self.config.model_weight)
                counts["model_count"] += 1
            raw_weights.append(weight)

        weights = torch.tensor(raw_weights, device=self.device, dtype=torch.float32)
        stats = {
            "raw_mean_weight": float(weights.mean().detach().item()) if raw_weights else 0.0,
            "raw_min_weight": float(weights.min().detach().item()) if raw_weights else 0.0,
            "raw_max_weight": float(weights.max().detach().item()) if raw_weights else 0.0,
            **counts,
        }

        if self.config.normalize and weights.numel() > 0:
            weights = weights * weights.numel() / (weights.sum() + self.epsilon)

        return weights, stats

    @staticmethod
    def _get_batch_size(batch: dict) -> int:
        for key in ("action", EPISODE_INDEX_FIELD, FRAME_INDEX_FIELD):
            value = batch.get(key)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return int(value.shape[0])
        return 1

