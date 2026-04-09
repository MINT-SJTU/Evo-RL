"""Dataset diagnosis: detect damage type and determine repairability."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lerobot.datasets.repair.io_helpers import (
    build_video_path,
    count_images_per_camera,
    count_video_files,
    find_log_for_dataset,
    get_video_keys,
    load_info,
    min_images_per_camera,
    parse_cp_from_log,
    read_recovery_rows,
    safe_read_parquet_metadata,
    safe_read_parquet_table,
    scan_parquet_files,
)
from lerobot.datasets.repair.types import DamageType, DiagnosisResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------


def find_tmp_videos(dataset_dir: Path) -> dict[str, Path]:
    """Find mp4 files in tmp* subdirectories. Returns {video_key: mp4_path}.

    lerobot encodes videos to tmp dirs (e.g. tmp8p8sh7a4/observation.images.left_wrist_000.mp4)
    before moving them to videos/. If the process crashes after encoding but before moving,
    these tmp dirs remain.
    """
    result: dict[str, Path] = {}
    for tmp_dir in sorted(dataset_dir.iterdir()):
        if not tmp_dir.is_dir() or not tmp_dir.name.startswith("tmp"):
            continue
        for mp4 in tmp_dir.glob("*.mp4"):
            stem = mp4.stem  # observation.images.left_wrist_000
            parts = stem.rsplit("_", 1)
            video_key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            result[video_key] = mp4
    return result


def has_frame_mismatch(recovery_count: int, images_per_camera: dict[str, int]) -> bool:
    if recovery_count <= 0 or not images_per_camera:
        return False
    return any(count != recovery_count for count in images_per_camera.values())


def truncate_target_frames(n_recovery_lines: int, image_floor: int, n_parquet_rows: int) -> int:
    candidates = [count for count in [n_recovery_lines, image_floor, n_parquet_rows] if count > 0]
    if not candidates:
        return 0
    return min(candidates)


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


# ---------------------------------------------------------------------------
# Main diagnosis
# ---------------------------------------------------------------------------


def _classify_damage(
    total_episodes: int,
    n_recovery_lines: int,
    image_floor: int,
    n_parquet_rows: int,
    n_video_files: int,
    video_keys: list[str],
    tmp_videos: dict[str, Path],
    images_per_camera: dict[str, int],
    has_cp: bool,
    log_cp_intervals: list[dict[str, Any]],
) -> DamageType:
    """Classify damage type. CRITICAL: preserve exact if/elif ordering (overlapping conditions)."""
    if total_episodes == 0 and n_parquet_rows == 0 and n_recovery_lines == 0 and image_floor == 0 and not tmp_videos:
        return DamageType.EMPTY_SHELL
    if n_video_files == 0 and tmp_videos and n_recovery_lines > 0:
        return DamageType.TMP_VIDEOS_STUCK
    if n_parquet_rows == 0 and n_video_files == 0 and (n_recovery_lines > 0 or image_floor > 0):
        return DamageType.CRASH_NO_SAVE
    if n_parquet_rows > 0 and n_video_files == 0 and len(video_keys) > 0:
        return DamageType.PARQUET_NO_VIDEO
    if total_episodes == 0 and n_parquet_rows > 0 and n_video_files > 0:
        return DamageType.META_STALE
    if has_frame_mismatch(n_recovery_lines, images_per_camera):
        return DamageType.FRAME_MISMATCH
    if not has_cp and log_cp_intervals and n_parquet_rows > 0:
        return DamageType.MISSING_CP
    return DamageType.HEALTHY


def diagnose_dataset(dataset_dir: Path) -> DiagnosisResult:
    info = load_info(dataset_dir)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    n_recovery_lines = len(read_recovery_rows(dataset_dir))
    images_per_camera = count_images_per_camera(dataset_dir)
    image_floor = min_images_per_camera(images_per_camera)
    n_parquet_files, _n_episodes, n_parquet_rows = scan_parquet_files(dataset_dir)
    n_video_files = count_video_files(dataset_dir)
    video_keys = get_video_keys(info)
    tmp_videos = find_tmp_videos(dataset_dir)
    has_cp = (dataset_dir / "critical_phase_intervals.json").exists()
    log_path = find_log_for_dataset(dataset_dir)
    log_cp_intervals = parse_cp_from_log(log_path) if log_path and not has_cp else []

    details: dict[str, Any] = {
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

    damage_type = _classify_damage(
        total_episodes=total_episodes,
        n_recovery_lines=n_recovery_lines,
        image_floor=image_floor,
        n_parquet_rows=n_parquet_rows,
        n_video_files=n_video_files,
        video_keys=video_keys,
        tmp_videos=tmp_videos,
        images_per_camera=images_per_camera,
        has_cp=has_cp,
        log_cp_intervals=log_cp_intervals,
    )

    repairable = is_repairable(damage_type, details)
    return DiagnosisResult(
        dataset_dir=dataset_dir,
        damage_type=damage_type,
        repairable=repairable,
        details=details,
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

    video_keys = get_video_keys(info)
    for video_key in video_keys:
        for ep_idx in range(total_episodes):
            video_path = build_video_path(dataset_dir, info, video_key, ep_idx)
            if not video_path.exists():
                errors.append(f"missing video: {video_path.relative_to(dataset_dir)}")

    return errors
