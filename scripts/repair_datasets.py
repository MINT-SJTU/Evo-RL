"""Diagnose and auto-repair damaged LeRobot recording datasets.

Usage:
    PYTHONPATH=src python scripts/repair_datasets.py --dataset-dir /path/to/dataset_or_parent
    PYTHONPATH=src python scripts/repair_datasets.py --dataset-dir /path/to/parent --force
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames

log = logging.getLogger(__name__)

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
    intervals = []
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
                break  # pure null bytes or whitespace — end of valid data
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

def get_image_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") in {"image", "video"}]

# ---------------------------------------------------------------------------
# Safe parquet helpers
# ---------------------------------------------------------------------------

def safe_read_parquet_metadata(path: Path) -> pq.FileMetaData | None:
    """Read parquet file metadata, returning None on corrupt files."""
    try:
        return pq.read_metadata(path)
    except Exception as exc:
        log.warning("Corrupt parquet file (cannot read metadata): %s: %s", path, exc)
        return None

def safe_read_parquet_table(path: Path, columns: list[str] | None = None) -> pq.Table | None:
    """Read parquet table, returning None on corrupt files."""
    try:
        return pq.read_table(path, columns=columns)
    except Exception as exc:
        log.warning("Corrupt parquet file (cannot read table): %s: %s", path, exc)
        return None

def count_parquet_rows(dataset_dir: Path) -> tuple[int, int]:
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []
    total_rows = 0
    valid_files = 0

    for parquet_path in parquet_files:
        metadata = safe_read_parquet_metadata(parquet_path)
        if metadata is None:
            continue
        table = safe_read_parquet_table(parquet_path, columns=[])
        if table is None:
            continue
        total_rows += metadata.num_rows
        valid_files += 1

    return valid_files, total_rows

def collect_parquet_counts(dataset_dir: Path) -> tuple[int, int]:
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []
    total_rows = 0
    episode_indices: set[int] = set()

    for parquet_path in parquet_files:
        metadata = safe_read_parquet_metadata(parquet_path)
        if metadata is None:
            continue
        table = safe_read_parquet_table(parquet_path, columns=["episode_index"])
        if table is None:
            continue
        total_rows += metadata.num_rows
        episode_indices.update(int(value) for value in table["episode_index"].to_pylist())

    return len(episode_indices), total_rows

# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def find_tmp_videos(dataset_dir: Path) -> dict[str, Path]:
    """Find mp4 files in tmp* subdirectories. Returns {video_key: mp4_path}.

    lerobot encodes videos to tmp dirs (e.g. tmp8p8sh7a4/observation.images.left_wrist_000.mp4)
    before moving them to videos/. If the process crashes after encoding but before moving,
    these tmp dirs remain.
    """
    result = {}
    for tmp_dir in sorted(dataset_dir.iterdir()):
        if not tmp_dir.is_dir() or not tmp_dir.name.startswith("tmp"):
            continue
        for mp4 in tmp_dir.glob("*.mp4"):
            # filename: observation.images.left_wrist_000.mp4
            # video_key: observation.images.left_wrist
            stem = mp4.stem  # observation.images.left_wrist_000
            # strip trailing _NNN episode suffix
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                video_key = parts[0]
            else:
                video_key = stem
            result[video_key] = mp4
    return result

def has_frame_mismatch(recovery_count: int, images_per_camera: dict[str, int]) -> bool:
    if recovery_count <= 0 or not images_per_camera:
        return False
    return any(count != recovery_count for count in images_per_camera.values())

def is_repairable(damage_type: DamageType, details: dict[str, Any]) -> bool:
    if damage_type == DamageType.CRASH_NO_SAVE:
        return details["n_recovery_lines"] > 0 and details["min_images_per_camera"] > 0
    if damage_type == DamageType.TMP_VIDEOS_STUCK:
        return details["n_recovery_lines"] > 0 and details["n_tmp_videos"] > 0
    if damage_type == DamageType.PARQUET_NO_VIDEO:
        return details["n_parquet_rows"] > 0 and details["min_images_per_camera"] > 0
    if damage_type == DamageType.META_STALE:
        return details["n_parquet_rows"] > 0
    if damage_type == DamageType.FRAME_MISMATCH:
        return details["truncate_target_frames"] > 0
    if damage_type == DamageType.MISSING_CP:
        return details.get("n_log_cp", 0) > 0
    return False

def truncate_target_frames(n_recovery_lines: int, image_floor: int, n_parquet_rows: int) -> int:
    candidates = [count for count in [n_recovery_lines, image_floor, n_parquet_rows] if count > 0]
    if not candidates:
        return 0
    return min(candidates)

def diagnose_dataset(dataset_dir: Path) -> DiagnosisResult:
    info = load_info(dataset_dir)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    n_recovery_lines = len(read_recovery_rows(dataset_dir))
    images_per_camera = count_images_per_camera(dataset_dir)
    image_floor = min_images_per_camera(images_per_camera)
    n_parquet_files, n_parquet_rows = count_parquet_rows(dataset_dir)
    n_video_files = count_video_files(dataset_dir)
    video_keys = get_video_keys(info)
    tmp_videos = find_tmp_videos(dataset_dir)
    has_cp = (dataset_dir / "critical_phase_intervals.json").exists()
    log_path = find_log_for_dataset(dataset_dir)
    log_cp_intervals = parse_cp_from_log(log_path) if log_path and not has_cp else []

    details = {
        "info_total_episodes": total_episodes,
        "info_total_frames": total_frames,
        "n_recovery_lines": n_recovery_lines,
        "images_per_camera": images_per_camera,
        "min_images_per_camera": image_floor,
        "n_parquet_files": n_parquet_files,
        "n_parquet_rows": n_parquet_rows,
        "n_video_files": n_video_files,
        "n_video_keys": len(video_keys),
        "n_tmp_videos": len(tmp_videos),
        "tmp_videos": tmp_videos,
        "truncate_target_frames": truncate_target_frames(
            n_recovery_lines=n_recovery_lines,
            image_floor=image_floor,
            n_parquet_rows=n_parquet_rows,
        ),
        "has_cp": has_cp,
        "n_log_cp": len(log_cp_intervals),
        "log_cp_intervals": log_cp_intervals,
        "log_path": log_path,
    }

    if total_episodes == 0 and n_parquet_rows == 0 and n_recovery_lines == 0 and image_floor == 0 and not tmp_videos:
        damage_type = DamageType.EMPTY_SHELL
    elif n_video_files == 0 and tmp_videos and n_recovery_lines > 0:
        damage_type = DamageType.TMP_VIDEOS_STUCK
    elif n_parquet_rows == 0 and n_video_files == 0 and (n_recovery_lines > 0 or image_floor > 0):
        damage_type = DamageType.CRASH_NO_SAVE
    elif n_parquet_rows > 0 and n_video_files == 0 and len(video_keys) > 0:
        damage_type = DamageType.PARQUET_NO_VIDEO
    elif total_episodes == 0 and n_parquet_rows > 0 and n_video_files > 0:
        damage_type = DamageType.META_STALE
    elif has_frame_mismatch(n_recovery_lines, images_per_camera):
        damage_type = DamageType.FRAME_MISMATCH
    elif not has_cp and log_cp_intervals and n_parquet_rows > 0:
        damage_type = DamageType.MISSING_CP
    else:
        damage_type = DamageType.HEALTHY

    repairable = is_repairable(damage_type, details)
    return DiagnosisResult(
        dataset_dir=dataset_dir,
        damage_type=damage_type,
        repairable=repairable,
        details=details,
    )

# ---------------------------------------------------------------------------
# Frame building
# ---------------------------------------------------------------------------

def get_single_episode_name(images_dir: Path, image_key: str) -> str:
    episode_dirs = list_episode_dirs(images_dir / image_key)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode image directories found for {image_key} under {images_dir}")
    return episode_dirs[0].name

def load_png_copy(png_path: Path) -> Image.Image | None:
    """Load a PNG and return a copy. Returns None if the file is corrupt."""
    try:
        with Image.open(png_path) as image:
            return image.copy()
    except Exception:
        log.warning("Corrupt PNG: %s", png_path)
        return None

def build_frame_dict(
    recovery_row: dict[str, Any],
    features: dict[str, Any],
    images_dir: Path,
    image_keys: list[str],
    episode_name_by_key: dict[str, str],
    frame_index: int,
    task: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": task}

    for key, feature in features.items():
        if key in SKIP_FRAME_KEYS:
            continue
        if key in image_keys:
            png_path = images_dir / key / episode_name_by_key[key] / f"frame-{frame_index:06d}.png"
            frame[key] = load_png_copy(png_path)
            continue
        if key not in recovery_row:
            continue
        frame[key] = coerce_recovery_value(recovery_row[key], feature)

    return frame

# ---------------------------------------------------------------------------
# Repair: crash-no-save
# ---------------------------------------------------------------------------

def copy_critical_phase_intervals(src_dir: Path, dst_dir: Path, max_frames: int | None = None) -> None:
    """Copy CP intervals, optionally truncating to max_frames."""
    src_path = src_dir / "critical_phase_intervals.json"
    if not src_path.exists():
        return
    with open(src_path) as f:
        intervals = json.load(f)
    if max_frames is not None:
        truncated = []
        for iv in intervals:
            if iv["start_frame"] >= max_frames:
                continue
            if iv["end_frame"] > max_frames:
                iv["end_frame"] = max_frames
            truncated.append(iv)
        intervals = truncated
    with open(dst_dir / "critical_phase_intervals.json", "w") as f:
        json.dump(intervals, f, indent=2)

def repair_crash_no_save(
    dataset_dir: Path,
    diagnosis: DiagnosisResult,
    task: str,
    vcodec: str,
    force: bool,
) -> tuple[Path, bool]:
    info = load_info(dataset_dir)
    recovery_rows = read_recovery_rows(dataset_dir)
    features = normalize_feature_shapes(copy.deepcopy(info["features"]))
    images_dir = dataset_dir / "images"
    image_keys = get_image_keys(info)
    n_usable = min(len(recovery_rows), diagnosis.details["min_images_per_camera"])

    if n_usable <= 0:
        raise ValueError(f"No usable frames available to rebuild {dataset_dir}")

    out_dir = dataset_dir.parent / f"{dataset_dir.name}_repaired"
    if out_dir.exists():
        if force:
            log.info("Force mode: removing existing %s", out_dir.name)
            shutil.rmtree(out_dir)
        else:
            log.info("Skipping %s — _repaired already exists (use --force to rebuild)", dataset_dir.name)
            return out_dir, False

    dataset = LeRobotDataset.create(
        repo_id=f"local/{out_dir.name}",
        fps=int(info["fps"]),
        root=out_dir,
        robot_type=info.get("robot_type"),
        features=features,
        use_videos=bool(image_keys),
        vcodec=vcodec,
    )
    episode_name_by_key = {key: get_single_episode_name(images_dir, key) for key in image_keys}

    log.info("Rebuilding %s into %s with %d usable frames", dataset_dir.name, out_dir.name, n_usable)

    actual_frames = 0
    for frame_index, recovery_row in enumerate(recovery_rows[:n_usable]):
        frame = build_frame_dict(
            recovery_row=recovery_row,
            features=features,
            images_dir=images_dir,
            image_keys=image_keys,
            episode_name_by_key=episode_name_by_key,
            frame_index=frame_index,
            task=task,
        )
        # Stop if any image is corrupt (truncated file from crash)
        if any(frame.get(k) is None for k in image_keys):
            log.warning("Corrupt PNG at frame %d, stopping rebuild at %d frames", frame_index, actual_frames)
            break
        dataset.add_frame(frame)
        actual_frames += 1
        if actual_frames % 500 == 0:
            log.info("Rebuild progress: %d/%d", actual_frames, n_usable)

    dataset.save_episode()
    dataset.finalize()
    copy_critical_phase_intervals(dataset_dir, out_dir, max_frames=actual_frames)
    return out_dir, True

# ---------------------------------------------------------------------------
# Repair: tmp-videos-stuck
# ---------------------------------------------------------------------------

def repair_tmp_videos_stuck(
    dataset_dir: Path,
    diagnosis: DiagnosisResult,
    task: str,
    vcodec: str,
    force: bool,
) -> tuple[Path, bool]:
    """Move tmp-encoded videos to proper locations and rebuild parquet from recovery.

    lerobot encodes videos to tmp dirs before moving them. If the process crashes
    after encoding, the videos are stranded in tmp dirs. This repair:
    1. Creates a _repaired dataset using recovery JSONL (for state/action parquet)
    2. Moves tmp videos to the proper videos/ directory structure
    3. Skips video re-encoding since we already have encoded videos
    """
    info = load_info(dataset_dir)
    tmp_videos: dict[str, Path] = diagnosis.details["tmp_videos"]
    recovery_rows = read_recovery_rows(dataset_dir)
    features = normalize_feature_shapes(copy.deepcopy(info["features"]))
    video_keys = get_video_keys(info)

    out_dir = dataset_dir.parent / f"{dataset_dir.name}_repaired"
    if out_dir.exists() and not force:
        log.info("Skipping %s — _repaired already exists", dataset_dir.name)
        return out_dir, False
    if out_dir.exists() and force:
        log.info("Removing existing %s for forced re-repair", out_dir.name)
        shutil.rmtree(out_dir)

    n_usable = len(recovery_rows)
    log.info("Rebuilding %s with %d frames + tmp videos", dataset_dir.name, n_usable)

    # Create dataset WITHOUT video features (to avoid needing image data),
    # then manually place tmp videos and patch info.json afterward.
    non_video_features = {k: v for k, v in features.items() if v.get("dtype") not in ("video", "image")}

    out_ds = LeRobotDataset.create(
        repo_id=f"local/{out_dir.name}",
        fps=int(info["fps"]),
        root=out_dir,
        robot_type=info.get("robot_type"),
        features=non_video_features,
        use_videos=False,
    )

    for frame_index, recovery_row in enumerate(recovery_rows[:n_usable]):
        frame: dict[str, Any] = {"task": task}
        for key, feature in non_video_features.items():
            if key in SKIP_FRAME_KEYS:
                continue
            if key not in recovery_row:
                continue
            frame[key] = coerce_recovery_value(recovery_row[key], feature)
        out_ds.add_frame(frame)
        if (frame_index + 1) % 2000 == 0:
            log.info("  progress: %d/%d", frame_index + 1, n_usable)

    out_ds.save_episode()
    out_ds.finalize()

    # Copy tmp videos to proper locations
    for vkey in video_keys:
        if vkey not in tmp_videos:
            log.warning("No tmp video found for key %s", vkey)
            continue
        src_mp4 = tmp_videos[vkey]
        dst_mp4 = build_video_path(out_dir, info, vkey, episode_index=0)
        dst_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_mp4, dst_mp4)
        log.info("Copied tmp video: %s -> %s", src_mp4.name, dst_mp4)

    # Patch info.json: restore full features, video_path, and correct totals
    out_info = load_info(out_dir)
    out_info["features"] = {k: v for k, v in info["features"].items()}
    out_info["total_episodes"] = 1
    out_info["total_frames"] = n_usable
    out_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    write_info(out_dir, out_info)

    # Patch episodes metadata: add video chunk/file/timestamp columns
    _patch_episodes_video_columns(out_dir, video_keys, n_usable, int(info["fps"]))

    copy_critical_phase_intervals(dataset_dir, out_dir)
    log.info("Repaired (tmp_videos_stuck): %s -> %s (%d frames)", dataset_dir.name, out_dir.name, n_usable)
    return out_dir, True

def _patch_episodes_video_columns(
    dataset_dir: Path, video_keys: list[str], n_frames: int, fps: int,
) -> None:
    """Add video chunk/file/timestamp columns to episodes metadata parquet.

    When a dataset is created with use_videos=False, the episodes metadata
    lacks video columns. This patches them in so aggregate_datasets works.
    """
    import pyarrow as pa

    ep_parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_parquets:
        return

    for ep_path in ep_parquets:
        table = pq.read_table(ep_path)
        n_rows = len(table)
        to_ts = (n_frames - 1) / fps if fps > 0 else 0.0

        for vkey in video_keys:
            prefix = f"videos/{vkey}"
            if f"{prefix}/chunk_index" in table.column_names:
                continue
            table = table.append_column(f"{prefix}/chunk_index", pa.array([0] * n_rows, type=pa.int64()))
            table = table.append_column(f"{prefix}/file_index", pa.array([0] * n_rows, type=pa.int64()))
            table = table.append_column(f"{prefix}/from_timestamp", pa.array([0.0] * n_rows, type=pa.float64()))
            table = table.append_column(f"{prefix}/to_timestamp", pa.array([to_ts] * n_rows, type=pa.float64()))

        pq.write_table(table, ep_path)
    log.info("Patched episodes metadata with video columns for %d video keys", len(video_keys))

# ---------------------------------------------------------------------------
# Repair: parquet-no-video
# ---------------------------------------------------------------------------

def parse_episode_index(episode_dir: Path) -> int:
    return int(episode_dir.name.split("-")[-1])

def build_video_path(dataset_dir: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
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

def repair_parquet_no_video(
    dataset_dir: Path,
    diagnosis: DiagnosisResult,
    vcodec: str,
) -> None:
    del diagnosis
    info = load_info(dataset_dir)
    images_dir = dataset_dir / "images"
    fps = int(info["fps"])

    for video_key in get_video_keys(info):
        video_images_dir = images_dir / video_key
        episode_dirs = list_episode_dirs(video_images_dir)
        if not episode_dirs:
            raise FileNotFoundError(
                f"No PNG episode directories found for video key {video_key} in {dataset_dir}"
            )

        for episode_dir in episode_dirs:
            episode_index = parse_episode_index(episode_dir)
            video_path = build_video_path(dataset_dir, info, video_key, episode_index)
            log.info("Encoding %s episode %s -> %s", video_key, episode_dir.name, video_path)
            encode_video_frames(episode_dir, video_path, fps, vcodec=vcodec, overwrite=True)

# ---------------------------------------------------------------------------
# Repair: meta-stale
# ---------------------------------------------------------------------------

def patch_info_totals_from_parquet(dataset_dir: Path) -> tuple[int, int]:
    info = load_info(dataset_dir)
    total_episodes, total_frames = collect_parquet_counts(dataset_dir)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"} if total_episodes > 0 else {}
    write_info(dataset_dir, info)
    return total_episodes, total_frames

def repair_meta_stale(dataset_dir: Path, diagnosis: DiagnosisResult) -> None:
    del diagnosis
    total_episodes, total_frames = patch_info_totals_from_parquet(dataset_dir)
    log.info(
        "Patched %s info.json totals: total_episodes=%d total_frames=%d",
        dataset_dir.name,
        total_episodes,
        total_frames,
    )

# ---------------------------------------------------------------------------
# Repair: missing-cp (recover from log)
# ---------------------------------------------------------------------------

def repair_missing_cp(dataset_dir: Path, diagnosis: DiagnosisResult) -> None:
    """Recover critical_phase_intervals.json from the recording log file."""
    intervals = diagnosis.details["log_cp_intervals"]
    log_path = diagnosis.details["log_path"]
    cp_path = dataset_dir / "critical_phase_intervals.json"
    with open(cp_path, "w") as f:
        json.dump(intervals, f, indent=2)
    log.info(
        "Repaired (missing_cp): recovered %d CP intervals from %s -> %s",
        len(intervals), log_path.name, cp_path.name,
    )

# ---------------------------------------------------------------------------
# Repair: frame-mismatch (truncation)
# ---------------------------------------------------------------------------

def truncate_recovery_jsonl(dataset_dir: Path, n_keep: int) -> None:
    recovery_path = dataset_dir / "recovery_frames.jsonl"
    if not recovery_path.exists():
        return

    kept_lines: list[str] = []
    with open(recovery_path) as handle:
        for line in handle:
            sanitized = sanitize_jsonl_line(line)
            if not sanitized:
                break
            kept_lines.append(f"{sanitized}\n")
            if len(kept_lines) >= n_keep:
                break

    with open(recovery_path, "w") as handle:
        handle.writelines(kept_lines)

def truncate_camera_pngs(camera_dir: Path, n_keep: int) -> None:
    seen = 0

    for episode_dir in list_episode_dirs(camera_dir):
        for png_path in list_frame_pngs(episode_dir):
            seen += 1
            if seen <= n_keep:
                continue
            png_path.unlink()

def truncate_images(dataset_dir: Path, n_keep: int) -> None:
    images_dir = dataset_dir / "images"
    for camera_dir in list_episode_dirs(images_dir):
        truncate_camera_pngs(camera_dir, n_keep)

def truncate_parquet(dataset_dir: Path, n_keep: int) -> None:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return

    remaining = n_keep

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        metadata = safe_read_parquet_metadata(parquet_path)
        if metadata is None:
            log.warning("Deleting corrupt parquet file: %s", parquet_path)
            parquet_path.unlink()
            continue

        row_count = metadata.num_rows
        if remaining <= 0:
            parquet_path.unlink()
            continue
        if row_count <= remaining:
            remaining -= row_count
            continue

        table = safe_read_parquet_table(parquet_path)
        if table is None:
            log.warning("Deleting corrupt parquet file: %s", parquet_path)
            parquet_path.unlink()
            continue
        pq.write_table(table.slice(0, remaining), parquet_path)
        remaining = 0

def repair_frame_mismatch(dataset_dir: Path, diagnosis: DiagnosisResult) -> None:
    n_keep = diagnosis.details["truncate_target_frames"]
    if n_keep <= 0:
        raise ValueError(f"No positive truncate target for {dataset_dir}")

    log.info("Truncating %s to %d frames", dataset_dir.name, n_keep)
    truncate_recovery_jsonl(dataset_dir, n_keep)
    truncate_images(dataset_dir, n_keep)
    truncate_parquet(dataset_dir, n_keep)

    if diagnosis.details["n_parquet_rows"] > 0:
        total_episodes, total_frames = patch_info_totals_from_parquet(dataset_dir)
        log.info(
            "Updated %s info.json after truncation: total_episodes=%d total_frames=%d",
            dataset_dir.name,
            total_episodes,
            total_frames,
        )

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_repaired_dataset(dataset_dir: Path) -> list[str]:
    """Return list of error strings (empty = healthy)."""
    errors: list[str] = []

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        errors.append("info.json missing")
        return errors

    try:
        info = load_info(dataset_dir)
    except (json.JSONDecodeError, OSError) as exc:
        log.exception("Unable to read %s", info_path)
        errors.append(f"info.json unreadable: {exc}")
        return errors

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0:
        errors.append(f"total_episodes={total_episodes} (expected > 0)")
    if total_frames <= 0:
        errors.append(f"total_frames={total_frames} (expected > 0)")

    parquet_rows = 0
    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    for parquet_path in parquet_files:
        metadata = safe_read_parquet_metadata(parquet_path)
        table = safe_read_parquet_table(parquet_path)
        if metadata is None or table is None:
            errors.append(f"unreadable parquet: {parquet_path.relative_to(dataset_dir)}")
            continue
        parquet_rows += metadata.num_rows

    if parquet_rows != total_frames:
        errors.append(f"parquet row sum {parquet_rows} != info total_frames {total_frames}")

    # Check expected video files exist
    video_keys = get_video_keys(info)
    for video_key in video_keys:
        for ep_idx in range(total_episodes):
            video_path = build_video_path(dataset_dir, info, video_key, ep_idx)
            if not video_path.exists():
                errors.append(f"missing video: {video_path.relative_to(dataset_dir)}")

    return errors

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def is_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()

def find_datasets(target: Path) -> list[Path]:
    if is_dataset_dir(target):
        return [target]
    if not target.is_dir():
        return []
    return [path for path in sorted(target.iterdir()) if path.is_dir() and is_dataset_dir(path)]

# ---------------------------------------------------------------------------
# Logging and repair dispatch
# ---------------------------------------------------------------------------

def log_diagnosis(diagnosis: DiagnosisResult) -> None:
    details = diagnosis.details
    tmp_tag = f" tmp_vid={details['n_tmp_videos']}" if details.get("n_tmp_videos", 0) > 0 else ""
    cp_tag = f" log_cp={details['n_log_cp']}" if details.get("n_log_cp", 0) > 0 else ""
    log.info(
        "%-40s %-18s repairable=%s recovery=%d min_images=%d parquet=%d videos=%d%s%s",
        diagnosis.dataset_dir.name,
        diagnosis.damage_type.value,
        diagnosis.repairable,
        details["n_recovery_lines"],
        details["min_images_per_camera"],
        details["n_parquet_rows"],
        details["n_video_files"],
        tmp_tag,
        cp_tag,
    )

def repair_dataset(
    diagnosis: DiagnosisResult,
    task: str,
    vcodec: str,
    dry_run: bool,
    force: bool,
) -> RepairResult:
    dataset_dir = diagnosis.dataset_dir
    damage = diagnosis.damage_type

    if damage == DamageType.HEALTHY:
        return RepairResult(dataset_dir, damage, "healthy")
    if damage == DamageType.EMPTY_SHELL:
        return RepairResult(dataset_dir, damage, "skipped", error="empty shell — nothing to recover")
    if not diagnosis.repairable:
        log.warning("Skipping unrepairable dataset %s", dataset_dir)
        return RepairResult(dataset_dir, damage, "skipped", error="unrepairable")
    if dry_run:
        log.info("Dry run: would repair %s as %s", dataset_dir.name, damage.value)
        return RepairResult(dataset_dir, damage, "skipped", error="dry run")

    if damage == DamageType.CRASH_NO_SAVE:
        _, rebuilt = repair_crash_no_save(dataset_dir, diagnosis, task, vcodec, force)
        if not rebuilt:
            return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
    elif damage == DamageType.TMP_VIDEOS_STUCK:
        _, rebuilt = repair_tmp_videos_stuck(dataset_dir, diagnosis, task, vcodec, force)
        if not rebuilt:
            return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
    elif damage == DamageType.PARQUET_NO_VIDEO:
        repair_parquet_no_video(dataset_dir, diagnosis, vcodec)
    elif damage == DamageType.META_STALE:
        repair_meta_stale(dataset_dir, diagnosis)
    elif damage == DamageType.FRAME_MISMATCH:
        repair_frame_mismatch(dataset_dir, diagnosis)
    elif damage == DamageType.MISSING_CP:
        repair_missing_cp(dataset_dir, diagnosis)

    return RepairResult(dataset_dir, damage, "repaired")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and repair damaged LeRobot datasets")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset dir or parent directory")
    parser.add_argument(
        "--task",
        default="Insert the copper screw into the black sleeve",
        help="Task string used when rebuilding crash-no-save datasets",
    )
    parser.add_argument("--dry-run", action="store_true", help="Diagnose without applying repairs")
    parser.add_argument("--vcodec", default="h264", help="Video codec for rebuilt or encoded videos")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if _repaired already exists")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.dataset_dir.exists():
        log.error("Path does not exist: %s", args.dataset_dir)
        sys.exit(1)

    datasets = find_datasets(args.dataset_dir)
    if not datasets:
        log.error("No dataset directories with meta/info.json found in %s", args.dataset_dir)
        sys.exit(1)

    log.info("Found %d dataset(s) to diagnose", len(datasets))
    results: list[RepairResult] = []

    for dataset_dir in datasets:
        try:
            diagnosis = diagnose_dataset(dataset_dir)
            log_diagnosis(diagnosis)
            result = repair_dataset(
                diagnosis, task=args.task, vcodec=args.vcodec, dry_run=args.dry_run, force=args.force,
            )

            if result.outcome == "repaired":
                verify_dir = diagnosis.dataset_dir
                if diagnosis.damage_type in (DamageType.CRASH_NO_SAVE, DamageType.TMP_VIDEOS_STUCK):
                    verify_dir = diagnosis.dataset_dir.parent / f"{diagnosis.dataset_dir.name}_repaired"
                verify_errors = verify_repaired_dataset(verify_dir)
                if verify_errors:
                    for err in verify_errors:
                        log.error("Verification failed for %s: %s", verify_dir.name, err)
                    result = RepairResult(
                        diagnosis.dataset_dir,
                        diagnosis.damage_type,
                        "failed",
                        error="; ".join(verify_errors),
                    )
        except Exception:
            log.exception("Failed to process %s", dataset_dir.name)
            result = RepairResult(
                dataset_dir,
                None,
                "failed",
                error=traceback.format_exc(),
            )

        results.append(result)

    healthy = sum(1 for r in results if r.outcome == "healthy")
    repaired = sum(1 for r in results if r.outcome == "repaired")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    failed = sum(1 for r in results if r.outcome == "failed")
    log.info(
        "Summary: %d healthy, %d repaired, %d skipped, %d failed out of %d total",
        healthy, repaired, skipped, failed, len(results),
    )

    if failed > 0:
        for r in results:
            if r.outcome != "failed":
                continue
            damage_type = r.damage_type.value if r.damage_type else "unknown"
            log.error("  FAILED: %s (%s)", r.dataset_dir.name, damage_type)

if __name__ == "__main__":
    main()
