"""Merge all datasets in a date folder into one autonomy dataset, then extract CP datasets.

Produces three outputs:
  - {folder_name}_autonomy:     all episodes merged
  - {folder_name}_cp_success:   success CP segments extracted from merged dataset
  - {folder_name}_cp_failure:   failure CP segments extracted from merged dataset

Usage:
    PYTHONPATH=src python scripts/merge_and_extract_cp.py \
        --dataset-dir /path/to/0408_271ep_sft

    # Dry-run: just show what would be merged
    PYTHONPATH=src python scripts/merge_and_extract_cp.py \
        --dataset-dir /path/to/0408_271ep_sft --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def transcode_videos_to_h264(datasets: list[Path]) -> None:
    """Re-encode any non-h264 videos to h264 for concat compatibility."""
    import subprocess

    for ds_dir in datasets:
        mp4s = list((ds_dir / "videos").rglob("*.mp4")) if (ds_dir / "videos").exists() else []
        if not mp4s:
            continue
        # Check first video's codec
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4s[0])],
            capture_output=True, text=True,
        )
        codec = probe.stdout.strip()
        if codec == "h264":
            continue

        log.info("Transcoding %s videos from %s to h264 (%d files)...", ds_dir.name, codec, len(mp4s))
        for mp4 in mp4s:
            tmp = mp4.with_suffix(".tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-an", str(tmp)],
                capture_output=True, check=True,
            )
            tmp.rename(mp4)

        # Update info.json video codec field
        info_path = ds_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        for feat in info["features"].values():
            if feat.get("dtype") == "video" and "info" in feat:
                feat["info"]["video.codec"] = "h264"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
            f.write("\n")
        log.info("Transcoded %s to h264", ds_dir.name)


def normalize_video_codec_metadata(datasets: list[Path]) -> None:
    """Ensure all datasets have identical video feature metadata.

    Reads the first dataset's video info as reference, then patches any
    dataset whose video info differs (e.g. av1 vs h264 codec field).
    This only changes metadata in info.json, not actual video files.
    """
    if not datasets:
        return

    # Use first dataset as reference
    ref_info = json.load(open(datasets[0] / "meta" / "info.json"))
    ref_video_infos = {}
    for key, feat in ref_info["features"].items():
        if feat.get("dtype") == "video" and "info" in feat:
            ref_video_infos[key] = feat["info"]

    if not ref_video_infos:
        return

    for ds_dir in datasets[1:]:
        info_path = ds_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        changed = False
        for key, ref_vi in ref_video_infos.items():
            feat = info["features"].get(key, {})
            if feat.get("dtype") != "video":
                continue
            current_vi = feat.get("info", {})
            if current_vi != ref_vi:
                feat["info"] = ref_vi
                changed = True

        if changed:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
                f.write("\n")
            log.info("Normalized video metadata: %s", ds_dir.name)


def find_healthy_datasets(parent_dir: Path) -> list[Path]:
    """Find all eval_autonomy_* datasets that have valid data."""
    results = []
    for entry in sorted(parent_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("eval_autonomy_"):
            continue
        info_path = entry / "meta" / "info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        if info.get("total_episodes", 0) > 0 and info.get("total_frames", 0) > 0:
            results.append(entry)
    return results


def load_all_cp_intervals(
    datasets: list[Path],
) -> list[dict[str, Any]]:
    """Load CP intervals from all datasets, adjusting episode_index for merge order.

    After aggregate_datasets, episodes are numbered sequentially across all source
    datasets in the order they appear in the repo_ids list. So dataset[0]'s episodes
    keep their indices, dataset[1]'s episodes get offset by dataset[0]'s episode count, etc.
    """
    all_intervals = []
    episode_offset = 0

    for ds_dir in datasets:
        info = json.load(open(ds_dir / "meta" / "info.json"))
        n_episodes = info.get("total_episodes", 0)

        cp_path = ds_dir / "critical_phase_intervals.json"
        if cp_path.exists():
            with open(cp_path) as f:
                intervals = json.load(f)
            for iv in intervals:
                all_intervals.append({
                    "episode_index": iv["episode_index"] + episode_offset,
                    "start_frame": iv["start_frame"],
                    "end_frame": iv["end_frame"],
                    "outcome": iv.get("outcome"),
                })

        episode_offset += n_episodes

    return all_intervals


def main():
    parser = argparse.ArgumentParser(description="Merge datasets and extract CP datasets")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Date folder (e.g. 0408_271ep_sft)")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged, don't execute")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: parent of dataset-dir)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src_dir = args.dataset_dir
    if not src_dir.exists():
        log.error("Path does not exist: %s", src_dir)
        sys.exit(1)

    datasets = find_healthy_datasets(src_dir)
    if not datasets:
        log.error("No healthy datasets found in %s", src_dir)
        sys.exit(1)

    # Compute stats
    total_frames = 0
    total_cp = 0
    for ds in datasets:
        info = json.load(open(ds / "meta" / "info.json"))
        total_frames += info.get("total_frames", 0)
        cp_path = ds / "critical_phase_intervals.json"
        if cp_path.exists():
            intervals = json.load(open(cp_path))
            total_cp += sum(iv["end_frame"] - iv["start_frame"] for iv in intervals)

    folder_name = src_dir.name
    output_base = args.output_dir or src_dir.parent
    autonomy_name = f"{folder_name}_autonomy"
    success_name = f"{folder_name}_cp_success"
    failure_name = f"{folder_name}_cp_failure"

    log.info("=== Merge Plan ===")
    log.info("Source: %s (%d datasets)", src_dir.name, len(datasets))
    log.info("Total frames: %d (%.1f min @ 30fps)", total_frames, total_frames / 30 / 60)
    log.info("Total CP frames: %d (%.1f min)", total_cp, total_cp / 30 / 60)
    log.info("Output autonomy: %s", autonomy_name)
    log.info("Output CP success: %s", success_name)
    log.info("Output CP failure: %s", failure_name)

    for ds in datasets:
        info = json.load(open(ds / "meta" / "info.json"))
        cp_path = ds / "critical_phase_intervals.json"
        n_cp = 0
        if cp_path.exists():
            n_cp = len(json.load(open(cp_path)))
        log.info("  %s: %d frames, %d CP intervals", ds.name, info["total_frames"], n_cp)

    if args.dry_run:
        log.info("Dry run — stopping here.")
        return

    # Step 0: Ensure all videos use the same codec (h264)
    # Some early datasets used av1, which can't be concat'd with h264.
    transcode_videos_to_h264(datasets)
    normalize_video_codec_metadata(datasets)

    # Step 1: Aggregate all datasets into one
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [f"local/{ds.name}" for ds in datasets]
    roots = [ds for ds in datasets]
    autonomy_root = output_base / autonomy_name

    if autonomy_root.exists():
        log.error("Output already exists: %s", autonomy_root)
        sys.exit(1)

    log.info("=== Step 1: Merging %d datasets into %s ===", len(datasets), autonomy_name)
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=f"local/{autonomy_name}",
        roots=roots,
        aggr_root=autonomy_root,
    )
    log.info("Merge complete: %s", autonomy_root)

    # Save merged CP intervals to the autonomy dataset
    all_intervals = load_all_cp_intervals(datasets)
    if all_intervals:
        cp_merged_path = autonomy_root / "critical_phase_intervals.json"
        with open(cp_merged_path, "w") as f:
            json.dump(all_intervals, f, indent=2)
        log.info("Saved %d merged CP intervals to %s", len(all_intervals), cp_merged_path.name)

    # Step 2: Extract CP success dataset
    success_intervals = [
        (iv["episode_index"], iv["start_frame"], iv["end_frame"], iv["outcome"])
        for iv in all_intervals if iv.get("outcome") == "success"
    ]
    failure_intervals = [
        (iv["episode_index"], iv["start_frame"], iv["end_frame"], iv["outcome"])
        for iv in all_intervals if iv.get("outcome") == "failure"
    ]

    from lerobot.utils.critical_phase_extraction_fast import extract_critical_phase_dataset_direct as extract_critical_phase_dataset

    if success_intervals:
        success_root = output_base / success_name
        if success_root.exists():
            log.error("Output already exists: %s", success_root)
        else:
            log.info("=== Step 2: Extracting %d success CP segments ===", len(success_intervals))
            extract_critical_phase_dataset(
                source_repo_id=f"local/{autonomy_name}",
                source_root=autonomy_root,
                output_repo_id=f"local/{success_name}",
                output_root=success_root,
                intervals=success_intervals,
                task=args.task,
            )
            log.info("CP success: %s (%d segments)", success_root, len(success_intervals))
    else:
        log.info("No success CP intervals found, skipping.")

    if failure_intervals:
        failure_root = output_base / failure_name
        if failure_root.exists():
            log.error("Output already exists: %s", failure_root)
        else:
            log.info("=== Step 3: Extracting %d failure CP segments ===", len(failure_intervals))
            extract_critical_phase_dataset(
                source_repo_id=f"local/{autonomy_name}",
                source_root=autonomy_root,
                output_repo_id=f"local/{failure_name}",
                output_root=failure_root,
                intervals=failure_intervals,
                task=args.task,
            )
            log.info("CP failure: %s (%d segments)", failure_root, len(failure_intervals))
    else:
        log.info("No failure CP intervals found, skipping.")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
