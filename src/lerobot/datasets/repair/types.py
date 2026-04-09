"""Shared types and constants for dataset repair."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


SKIP_FRAME_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


class DamageType(Enum):
    HEALTHY = "healthy"
    EMPTY_SHELL = "empty_shell"
    CRASH_NO_SAVE = "crash_no_save"
    TMP_VIDEOS_STUCK = "tmp_videos_stuck"
    PARQUET_NO_VIDEO = "parquet_no_video"
    META_STALE = "meta_stale"
    FRAME_MISMATCH = "frame_mismatch"
    MISSING_CP = "missing_cp"


@dataclass
class DiagnosisResult:
    dataset_dir: Path
    damage_type: DamageType
    repairable: bool
    details: dict[str, Any]


@dataclass
class RepairResult:
    dataset_dir: Path
    damage_type: DamageType | None
    outcome: str  # "healthy", "repaired", "skipped", "failed"
    error: str | None = None
