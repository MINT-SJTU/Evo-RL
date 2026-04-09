"""I/O helpers: JSON, JSONL, parquet, image, and directory utilities."""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON / recovery helpers
# ---------------------------------------------------------------------------


def sanitize_jsonl_line(line: str) -> str:
    """Strip null bytes and whitespace from a JSONL line."""
    return line.replace("\x00", "").strip()


# Regex for: [CP] END at episode 0, frame 496 (segment: 333-496, 163 frames, outcome=success)
_CP_END_RE = re.compile(
    r"\[CP\] END at episode (\d+), frame (\d+) "
    r"\(segment: (\d+)-(\d+), \d+ frames(?:, outcome=(\w+))?\)"
)


def parse_cp_from_log(log_path: Path) -> list[dict[str, Any]]:
    """Parse critical phase intervals from a recording log file."""
    intervals: list[dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            m = _CP_END_RE.search(line)
            if not m:
                continue
            intervals.append({
                "episode_index": int(m.group(1)),
                "start_frame": int(m.group(3)),
                "end_frame": int(m.group(4)),
                "outcome": m.group(5),  # None if no outcome
            })
    return intervals


def find_log_for_dataset(dataset_dir: Path) -> Path | None:
    """Find the .log file for a dataset (sibling with same name + .log suffix)."""
    log_path = dataset_dir.parent / f"{dataset_dir.name}.log"
    return log_path if log_path.exists() else None


def load_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as handle:
        return json.load(handle)


def write_info(dataset_dir: Path, info: dict[str, Any]) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, "w") as handle:
        json.dump(info, handle, indent=4)
        handle.write("\n")


def read_recovery_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    recovery_path = dataset_dir / "recovery_frames.jsonl"
    if not recovery_path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with open(recovery_path) as handle:
        for line_num, line in enumerate(handle, 1):
            sanitized = sanitize_jsonl_line(line)
            if not sanitized:
                break  # pure null bytes or whitespace -- end of valid data
            try:
                rows.append(json.loads(sanitized))
            except json.JSONDecodeError as exc:
                log.warning(
                    "Corrupt JSON in %s at line %d: %s",
                    recovery_path,
                    line_num,
                    exc,
                )
                break

    return rows


# ---------------------------------------------------------------------------
# Feature normalization and value coercion
# ---------------------------------------------------------------------------


def normalize_feature_shapes(features: dict) -> dict:
    """Deep-copy features dict, convert all shape values from lists to tuples.

    Prevents numpy tuple (12,) vs JSON list [12] mismatch in lerobot's
    validate_frame.
    """
    normalized = copy.deepcopy(features)
    for feat in normalized.values():
        if "shape" in feat:
            feat["shape"] = tuple(feat["shape"])
    return normalized


def coerce_recovery_value(value: Any, feature: dict) -> Any:
    """Convert a recovery row value using the feature's dtype."""
    if isinstance(value, list):
        return np.array(value, dtype=np.dtype(feature["dtype"]))
    return value


# ---------------------------------------------------------------------------
# Image / directory helpers
# ---------------------------------------------------------------------------


def list_episode_dirs(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return [path for path in sorted(parent.iterdir()) if path.is_dir()]


def list_frame_pngs(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("frame-*.png"))


def count_images_per_camera(dataset_dir: Path) -> dict[str, int]:
    images_dir = dataset_dir / "images"
    counts: dict[str, int] = {}

    for camera_dir in list_episode_dirs(images_dir):
        total = 0
        for episode_dir in list_episode_dirs(camera_dir):
            total += len(list_frame_pngs(episode_dir))
        counts[camera_dir.name] = total

    return counts


def min_images_per_camera(images_per_camera: dict[str, int]) -> int:
    if not images_per_camera:
        return 0
    return min(images_per_camera.values())


def count_video_files(dataset_dir: Path) -> int:
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        return 0
    return len(list(videos_dir.rglob("*.mp4")))


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def get_visual_keys(info: dict[str, Any]) -> list[str]:
    """Return feature keys with dtype 'image' or 'video'."""
    return [key for key, feature in info["features"].items() if feature.get("dtype") in {"image", "video"}]


# ---------------------------------------------------------------------------
# Safe parquet helpers
# ---------------------------------------------------------------------------

_PARQUET_ERRORS = (OSError, pa.lib.ArrowException)


def safe_read_parquet_metadata(path: Path) -> pq.FileMetaData | None:
    """Read parquet file metadata, returning None on corrupt files."""
    try:
        return pq.read_metadata(path)
    except _PARQUET_ERRORS as exc:
        log.warning("Corrupt parquet file (cannot read metadata): %s: %s", path, exc)
        return None


def safe_read_parquet_table(path: Path, columns: list[str] | None = None) -> pa.Table | None:
    """Read parquet table, returning None on corrupt files."""
    try:
        return pq.read_table(path, columns=columns)
    except _PARQUET_ERRORS as exc:
        log.warning("Corrupt parquet file (cannot read table): %s: %s", path, exc)
        return None


def scan_parquet_files(dataset_dir: Path) -> tuple[int, int, int]:
    """Scan all parquet files under data/.

    Returns (n_valid_files, n_episodes, n_total_rows).
    Reads the episode_index column to count distinct episodes.
    """
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []
    total_rows = 0
    valid_files = 0
    episode_indices: set[int] = set()

    for parquet_path in parquet_files:
        metadata = safe_read_parquet_metadata(parquet_path)
        if metadata is None:
            continue
        total_rows += metadata.num_rows
        valid_files += 1
        table = safe_read_parquet_table(parquet_path, columns=["episode_index"])
        if table is not None:
            episode_indices.update(int(value) for value in table["episode_index"].to_pylist())

    return valid_files, len(episode_indices), total_rows


def load_png_copy(png_path: Path) -> PIL.Image.Image | None:
    """Load a PNG and return a copy. Returns None if the file is corrupt."""
    try:
        with PIL.Image.open(png_path) as image:
            return image.copy()
    except (OSError, PIL.UnidentifiedImageError):
        log.warning("Corrupt PNG: %s", png_path)
        return None


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def build_video_path(dataset_dir: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
    """Build the expected video file path for an episode."""
    chunks_size = int(info.get("chunks_size", 1000))
    video_path_template = (
        info.get("video_path")
        or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    chunk_index = episode_index // chunks_size
    file_index = episode_index % chunks_size
    return dataset_dir / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def is_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def find_datasets(target: Path) -> list[Path]:
    if is_dataset_dir(target):
        return [target]
    if not target.is_dir():
        return []
    return [path for path in sorted(target.iterdir()) if path.is_dir() and is_dataset_dir(path)]
