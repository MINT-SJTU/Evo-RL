#!/usr/bin/env python

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

from lerobot.utils.recording_annotations import normalize_episode_success_label


@dataclass
class ValueInferenceDatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    download_videos: bool = True
    success_field: str = "episode_success"
    default_success: str = "failure"

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("'dataset.repo_id' must be non-empty.")
        if not self.success_field:
            raise ValueError("'dataset.success_field' must be non-empty.")

        normalized = normalize_episode_success_label(self.default_success)
        if normalized is None:
            raise ValueError("'dataset.default_success' must be either 'success' or 'failure'.")
        self.default_success = normalized


@dataclass
class ValueInferenceCheckpointConfig:
    checkpoint_path: str
    checkpoint_ref: str = "last"

    def validate(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("'inference.checkpoint_path' must be non-empty.")
        if not self.checkpoint_ref:
            raise ValueError("'inference.checkpoint_ref' must be non-empty.")


@dataclass
class ValueInferenceRuntimeConfig:
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("'runtime.batch_size' must be > 0.")
        if self.num_workers < 0:
            raise ValueError("'runtime.num_workers' must be >= 0.")


@dataclass
class ValueInferenceACPConfig:
    enable: bool = False
    n_step: int = 50
    positive_ratio: float = 0.3
    force_intervention_positive: bool = True
    length_scale_quantile: float = 0.8

    intervention_field: str = "complementary_info.is_intervention"
    value_field: str = "complementary_info.value"
    advantage_field: str = "complementary_info.advantage"
    indicator_field: str = "complementary_info.acp_indicator"

    c_fail_coef: float = 1.0

    def validate(self) -> None:
        if self.n_step <= 0:
            raise ValueError("'acp.n_step' must be > 0.")
        if not 0.0 <= self.positive_ratio <= 1.0:
            raise ValueError("'acp.positive_ratio' must be within [0, 1].")
        if not 0.0 < self.length_scale_quantile <= 1.0:
            raise ValueError("'acp.length_scale_quantile' must be within (0, 1].")
        if self.c_fail_coef < 0:
            raise ValueError("'acp.c_fail_coef' must be non-negative.")
        if not self.value_field:
            raise ValueError("'acp.value_field' must be non-empty.")
        if self.enable and (not self.advantage_field or not self.indicator_field):
            raise ValueError(
                "'acp.advantage_field' and 'acp.indicator_field' must be non-empty when 'acp.enable=true'."
            )


@dataclass
class ValueInferenceVizConfig:
    enable: bool = False
    episodes: str = "all"
    video_key: str | None = None
    video_keys: str | None = None
    overwrite: bool = False
    vcodec: str = "libsvtav1"
    frame_storage_mode: str = "memory"
    smooth_window: int = 1

    def validate(self) -> None:
        if not self.episodes:
            raise ValueError("'viz.episodes' must be non-empty.")
        if not self.vcodec:
            raise ValueError("'viz.vcodec' must be non-empty.")
        if self.frame_storage_mode not in {"memory", "disk"}:
            raise ValueError("'viz.frame_storage_mode' must be one of {'memory', 'disk'}.")
        if self.smooth_window < 1:
            raise ValueError("'viz.smooth_window' must be >= 1. Use 1 to disable smoothing.")


@dataclass
class ValueInferencePipelineConfig:
    dataset: ValueInferenceDatasetConfig
    inference: ValueInferenceCheckpointConfig = field(
        default_factory=lambda: ValueInferenceCheckpointConfig(checkpoint_path="")
    )
    runtime: ValueInferenceRuntimeConfig = field(default_factory=ValueInferenceRuntimeConfig)
    acp: ValueInferenceACPConfig = field(default_factory=ValueInferenceACPConfig)
    viz: ValueInferenceVizConfig = field(default_factory=ValueInferenceVizConfig)

    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000

    rename_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.dataset.validate()
        self.inference.validate()
        self.runtime.validate()
        self.acp.validate()
        self.viz.validate()

        if not self.job_name:
            repo_tag = self.dataset.repo_id.replace("/", "_")
            self.job_name = f"value_infer_{repo_tag}"

        if self.output_dir is None:
            now = dt.datetime.now()
            out_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/value_infer") / out_dir

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]
