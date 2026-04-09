"""Repair functions for each damage type."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .io_helpers import (
    build_video_path,
    coerce_recovery_value,
    get_video_keys,
    get_visual_keys,
    list_episode_dirs,
    list_frame_pngs,
    load_info,
    load_png_copy,
    normalize_feature_shapes,
    read_recovery_rows,
    safe_read_parquet_metadata,
    safe_read_parquet_table,
    scan_parquet_files,
    write_info,
)
from .types import (
    SKIP_FRAME_KEYS,
    DamageType,
    DiagnosisResult,
    RepairResult,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def prepare_output_dir(dataset_dir: Path, force: bool) -> tuple[Path, bool]:
    """Prepare {dataset_dir}_repaired output directory.

    Returns (out_dir, should_proceed).
    If exists + force: rmtree and True.
    If exists + no force: return (dir, False).
    If not exists: return (dir, True).
    """
    out_dir = dataset_dir.parent / f"{dataset_dir.name}_repaired"
    if out_dir.exists() and force:
        log.info("Force mode: removing existing %s", out_dir.name)
        shutil.rmtree(out_dir)
        return out_dir, True
    if out_dir.exists():
        log.info("Skipping %s -- _repaired already exists (use --force to rebuild)", dataset_dir.name)
        return out_dir, False
    return out_dir, True


def get_single_episode_name(images_dir: Path, image_key: str) -> str:
    episode_dirs = list_episode_dirs(images_dir / image_key)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode image directories found for {image_key} under {images_dir}")
    return episode_dirs[0].name


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


def parse_episode_index(episode_dir: Path) -> int:
    return int(episode_dir.name.split("-")[-1])



def _patch_episodes_video_columns(
    dataset_dir: Path, video_keys: list[str], n_frames: int, fps: int,
) -> None:
    """Add video chunk/file/timestamp columns to episodes metadata parquet.

    When a dataset is created with use_videos=False, the episodes metadata
    lacks video columns. This patches them in so aggregate_datasets works.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

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
# Repair: crash-no-save
# ---------------------------------------------------------------------------


def repair_crash_no_save(
    dataset_dir: Path,
    diagnosis: DiagnosisResult,
    task: str,
    vcodec: str,
    force: bool,
) -> tuple[Path, bool]:
    out_dir, should_proceed = prepare_output_dir(dataset_dir, force)
    if not should_proceed:
        return out_dir, False

    info = load_info(dataset_dir)
    recovery_rows = read_recovery_rows(dataset_dir)
    features = normalize_feature_shapes(info["features"])
    images_dir = dataset_dir / "images"
    image_keys = get_visual_keys(info)
    n_usable = min(len(recovery_rows), diagnosis.details["min_images_per_camera"])

    if n_usable <= 0:
        raise ValueError(f"No usable frames available to rebuild {dataset_dir}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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

    actual_frames = _add_frames_from_recovery(
        dataset, recovery_rows[:n_usable], features, images_dir, image_keys, episode_name_by_key, task,
    )

    dataset.save_episode()
    dataset.finalize()
    copy_critical_phase_intervals(dataset_dir, out_dir, max_frames=actual_frames)
    return out_dir, True


def _add_frames_from_recovery(
    dataset: LeRobotDataset,
    recovery_rows: list[dict[str, Any]],
    features: dict[str, Any],
    images_dir: Path,
    image_keys: list[str],
    episode_name_by_key: dict[str, str],
    task: str,
) -> int:
    """Add frames from recovery rows to dataset, stopping at first corrupt PNG."""
    actual_frames = 0
    for frame_index, recovery_row in enumerate(recovery_rows):
        frame = build_frame_dict(
            recovery_row=recovery_row,
            features=features,
            images_dir=images_dir,
            image_keys=image_keys,
            episode_name_by_key=episode_name_by_key,
            frame_index=frame_index,
            task=task,
        )
        if any(frame.get(k) is None for k in image_keys):
            log.warning("Corrupt PNG at frame %d, stopping rebuild at %d frames", frame_index, actual_frames)
            break
        dataset.add_frame(frame)
        actual_frames += 1
        if actual_frames % 500 == 0:
            log.info("Rebuild progress: %d/%d", actual_frames, len(recovery_rows))
    return actual_frames


# ---------------------------------------------------------------------------
# Repair: tmp-videos-stuck
# ---------------------------------------------------------------------------


def repair_tmp_videos_stuck(
    dataset_dir: Path,
    diagnosis: DiagnosisResult,
    task: str,
    force: bool,
) -> tuple[Path, bool]:
    """Move tmp-encoded videos to proper locations and rebuild parquet from recovery."""
    out_dir, should_proceed = prepare_output_dir(dataset_dir, force)
    if not should_proceed:
        return out_dir, False

    info = load_info(dataset_dir)
    tmp_videos: dict[str, Path] = diagnosis.details["tmp_videos"]
    recovery_rows = read_recovery_rows(dataset_dir)
    features = normalize_feature_shapes(info["features"])
    video_keys = get_video_keys(info)

    n_usable = len(recovery_rows)
    log.info("Rebuilding %s with %d frames + tmp videos", dataset_dir.name, n_usable)

    # Create dataset WITHOUT video features (to avoid needing image data),
    # then manually place tmp videos and patch info.json afterward.
    non_video_features = {k: v for k, v in features.items() if v.get("dtype") not in ("video", "image")}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_ds = LeRobotDataset.create(
        repo_id=f"local/{out_dir.name}",
        fps=int(info["fps"]),
        root=out_dir,
        robot_type=info.get("robot_type"),
        features=non_video_features,
        use_videos=False,
    )

    # Use build_frame_dict with image_keys=[] to build state-only frames
    for frame_index, recovery_row in enumerate(recovery_rows[:n_usable]):
        frame = build_frame_dict(
            recovery_row=recovery_row,
            features=non_video_features,
            images_dir=dataset_dir / "images",
            image_keys=[],
            episode_name_by_key={},
            frame_index=frame_index,
            task=task,
        )
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

    _patch_episodes_video_columns(out_dir, video_keys, n_usable, int(info["fps"]))
    copy_critical_phase_intervals(dataset_dir, out_dir)
    log.info("Repaired (tmp_videos_stuck): %s -> %s (%d frames)", dataset_dir.name, out_dir.name, n_usable)
    return out_dir, True


# ---------------------------------------------------------------------------
# Repair: parquet-no-video
# ---------------------------------------------------------------------------


def repair_parquet_no_video(
    dataset_dir: Path,
    vcodec: str,
) -> None:
    from lerobot.datasets.video_utils import encode_video_frames

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
    _n_files, total_episodes, total_frames = scan_parquet_files(dataset_dir)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"} if total_episodes > 0 else {}
    write_info(dataset_dir, info)
    return total_episodes, total_frames


def repair_meta_stale(dataset_dir: Path) -> None:
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
    from .io_helpers import sanitize_jsonl_line

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
    import pyarrow.parquet as pq

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
# Dispatch
# ---------------------------------------------------------------------------


def _run_verify(diagnosis: DiagnosisResult) -> list[str]:
    """Determine verify directory and run verification."""
    from .diagnosis import verify_repaired_dataset

    verify_dir = diagnosis.dataset_dir
    if diagnosis.damage_type in (DamageType.CRASH_NO_SAVE, DamageType.TMP_VIDEOS_STUCK):
        verify_dir = diagnosis.dataset_dir.parent / f"{diagnosis.dataset_dir.name}_repaired"
    return verify_repaired_dataset(verify_dir)


def _dispatch_repair(
    diagnosis: DiagnosisResult,
    task: str,
    vcodec: str,
    force: bool,
) -> RepairResult:
    """Execute the appropriate repair function. Returns RepairResult."""
    dataset_dir = diagnosis.dataset_dir
    damage = diagnosis.damage_type

    if damage == DamageType.CRASH_NO_SAVE:
        _, rebuilt = repair_crash_no_save(dataset_dir, diagnosis, task, vcodec, force)
        if not rebuilt:
            return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
    elif damage == DamageType.TMP_VIDEOS_STUCK:
        _, rebuilt = repair_tmp_videos_stuck(dataset_dir, diagnosis, task, force)
        if not rebuilt:
            return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
    elif damage == DamageType.PARQUET_NO_VIDEO:
        repair_parquet_no_video(dataset_dir, vcodec)
    elif damage == DamageType.META_STALE:
        repair_meta_stale(dataset_dir)
    elif damage == DamageType.FRAME_MISMATCH:
        repair_frame_mismatch(dataset_dir, diagnosis)
    elif damage == DamageType.MISSING_CP:
        repair_missing_cp(dataset_dir, diagnosis)

    return RepairResult(dataset_dir, damage, "repaired")


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
        return RepairResult(dataset_dir, damage, "skipped", error="empty shell -- nothing to recover")
    if not diagnosis.repairable:
        log.warning("Skipping unrepairable dataset %s", dataset_dir)
        return RepairResult(dataset_dir, damage, "skipped", error="unrepairable")
    if dry_run:
        log.info("Dry run: would repair %s as %s", dataset_dir.name, damage.value)
        return RepairResult(dataset_dir, damage, "skipped", error="dry run")

    result = _dispatch_repair(diagnosis, task, vcodec, force)
    if result.outcome != "repaired":
        return result

    verify_errors = _run_verify(diagnosis)
    if not verify_errors:
        return result

    for err in verify_errors:
        log.error("Verification failed for %s: %s", dataset_dir.name, err)
    return RepairResult(dataset_dir, damage, "failed", error="; ".join(verify_errors))
