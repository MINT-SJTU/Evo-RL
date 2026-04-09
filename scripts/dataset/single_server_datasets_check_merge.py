#!/usr/bin/env python
"""
Check, validate, and merge multiple LeRobot sub-datasets on a single machine.

Given a dataset name prefix (e.g. "zzz") under a root directory, this script:
1. Scans for all matching sub-datasets
2. Computes per-episode lengths across all sub-datasets
3. Flags anomalous episodes (too short or too long vs. global average)
4. Extracts boundary frames (first/last) for every episode
5. Copies anomalous episode videos to a review folder
6. Merges all non-anomalous sub-datasets into a single output dataset
7. Optionally sets the task description in the merged dataset
8. Runs lerobot-dataset-report on the merged result

Examples:

```bash
# Merge all zzz_20260402_* datasets into "0402"
PYTHONPATH=src python -m lerobot.scripts.single_server_datasets_check_merge \
    --prefix zzz_20260402 --output-name 0402 \
    --task "Insert the copper screw into the black sleeve."

# Dry-run: only report anomalies, don't merge
PYTHONPATH=src python -m lerobot.scripts.single_server_datasets_check_merge \
    --prefix zzz_20260402 --output-name 0402 --dry-run
```
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pyarrow as pa

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.utils import load_info
from lerobot.scripts.lerobot_dataset_report import build_report, format_text_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_DATASETS_ROOT = Path.home() / ".roboclaw/workspace/embodied/datasets/local"
DEFAULT_SETUP_JSON = Path.home() / ".roboclaw/workspace/embodied/setup.json"
SHORT_RATIO = 0.3   # episode < avg * SHORT_RATIO is anomalous
LONG_RATIO = 3.0    # episode > avg * LONG_RATIO is anomalous


# ── Camera resolution from setup.json ──────────────────────────────────

def resolve_top_camera_key(setup_json: Path, ds_info: dict) -> str | None:
    """Read setup.json to find the 'top' camera alias, then match to a dataset feature key."""
    if not setup_json.exists():
        return None
    setup = json.loads(setup_json.read_text())
    top_alias = None
    for cam in setup.get("cameras", []):
        if cam.get("alias") == "top":
            top_alias = "top"
            break
    if not top_alias:
        return None
    # Match: feature key containing the alias (e.g. "observation.images.left_top")
    for key in ds_info.get("features", {}):
        if ds_info["features"][key].get("dtype") != "video":
            continue
        if top_alias in key:
            return key
    return None


# ── Step 1: discover sub-datasets ──────────────────────────────────────

def discover_datasets(root: Path, prefix: str) -> list[Path]:
    """Return sorted list of dataset dirs matching prefix that have meta/info.json."""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and (d / "meta/info.json").exists()
    )


# ── Step 2: per-episode length analysis ────────────────────────────────

def get_episode_lengths(ds_path: Path, fps: int) -> list[tuple[int, int, float]]:
    """Return [(ep_idx, frame_count, length_seconds), ...] for a dataset."""
    data_dir = ds_path / "data"
    if not data_dir.exists():
        return []
    dataset = pa_ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(columns=["episode_index"])
    ep_counts = Counter(table.column("episode_index").to_pylist())
    return [(ep, count, count / fps) for ep, count in sorted(ep_counts.items())]


def find_anomalies(
    all_episodes: list[tuple[str, int, int, float]],
    short_ratio: float,
    long_ratio: float,
) -> tuple[float, float, float, list[tuple[str, int, int, float]]]:
    """Compute thresholds and return (avg, min_thresh, max_thresh, anomalous_episodes)."""
    lengths = [e[3] for e in all_episodes]
    avg_len = sum(lengths) / len(lengths)
    min_thresh = avg_len * short_ratio
    max_thresh = avg_len * long_ratio
    anomalous = [e for e in all_episodes if e[3] < min_thresh or e[3] > max_thresh]
    return avg_len, min_thresh, max_thresh, anomalous


# ── Step 3: boundary frames extraction ─────────────────────────────────

def extract_boundary_frames(ds_path: Path, output_dir: Path, camera_key: str | None = None) -> bool:
    """Run lerobot_export_boundary_frames on a dataset. Returns True on success."""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_export_boundary_frames",
        "--dataset", str(ds_path),
        "--episodes", "all",
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    if camera_key:
        cmd.extend(["--camera-key", camera_key])
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"},
    )
    if result.returncode != 0:
        log.warning("  boundary frames FAILED: %s", result.stderr[-300:] if result.stderr else "unknown")
        return False
    return True


# ── Step 4: copy anomalous episode videos for review ───────────────────

def extract_anomalous_episode_clips(
    ds_path: Path,
    episode_indices: list[int],
    review_dir: Path,
    camera_key: str | None = None,
) -> None:
    """Use ffmpeg to extract only the anomalous episode segments from video files."""
    review_dir.mkdir(parents=True, exist_ok=True)
    info = load_info(ds_path)
    video_path_template = info.get("video_path", "")
    if not video_path_template:
        return

    ep_table = pa_ds.dataset(str(ds_path / "meta" / "episodes"), format="parquet").to_table()
    ep_df = ep_table.to_pandas()

    if camera_key is None:
        video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
        camera_key = video_keys[0] if video_keys else None
    if camera_key is None:
        return

    chunk_col = f"videos/{camera_key}/chunk_index"
    file_col = f"videos/{camera_key}/file_index"
    from_ts_col = f"videos/{camera_key}/from_timestamp"
    to_ts_col = f"videos/{camera_key}/to_timestamp"

    for ep_idx in episode_indices:
        matching = ep_df[ep_df["episode_index"] == ep_idx]
        if matching.empty:
            continue
        row = matching.iloc[0]

        chunk_idx = int(row[chunk_col]) if chunk_col in ep_df.columns else 0
        file_idx = int(row[file_col]) if file_col in ep_df.columns else 0
        from_ts = float(row[from_ts_col]) if from_ts_col in ep_df.columns else 0.0
        to_ts = float(row[to_ts_col]) if to_ts_col in ep_df.columns else 0.0

        src = ds_path / video_path_template.format(
            video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx,
        )
        if not src.exists():
            continue

        dst = review_dir / f"{ds_path.name}_ep{ep_idx}.mp4"
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(from_ts),
                "-to", str(to_ts),
                "-i", str(src),
                "-c", "copy",
                str(dst),
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("  extracted: %s (%.1fs-%.1fs)", dst.name, from_ts, to_ts)
        else:
            log.warning("  ffmpeg FAILED for ep %d: %s", ep_idx, result.stderr[-200:])


# ── Step 5: merge datasets ─────────────────────────────────────────────

def merge_datasets(
    root: Path,
    dataset_names: list[str],
    output_name: str,
) -> Path:
    """Merge selected datasets into output_name under root."""
    output_path = root / output_name
    if output_path.exists():
        shutil.rmtree(output_path)

    roots = [root / name for name in dataset_names]
    aggregate_datasets(
        repo_ids=dataset_names,
        aggr_repo_id=output_name,
        roots=roots,
        aggr_root=output_path,
    )
    return output_path


# ── Step 6: set task name ───────��──────────────────────────────────────

def set_task_name(ds_path: Path, task: str) -> None:
    """Overwrite tasks.parquet with a single task entry."""
    tasks_path = ds_path / "meta" / "tasks.parquet"
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    pq.write_table(pa.Table.from_pandas(df), tasks_path)
    log.info("Task set to: %s", task)


# ── Main pipeline ─────��────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    root = Path(args.datasets_root).expanduser().resolve()
    setup_json = Path(args.setup_json).expanduser().resolve()
    check_dir = root / f"{args.output_name}_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: discover
    all_ds = discover_datasets(root, args.prefix)
    log.info("Found %d directories matching prefix '%s'", len(all_ds), args.prefix)

    # Step 2: per-episode analysis + resolve top camera
    all_episodes: list[tuple[str, int, int, float]] = []  # (ds_name, ep_idx, frames, seconds)
    ds_with_episodes: list[Path] = []
    top_camera_key: str | None = None

    for ds_path in all_ds:
        info = json.loads((ds_path / "meta/info.json").read_text())
        n_eps = info["total_episodes"]
        if n_eps == 0:
            log.info("SKIP %s: 0 episodes", ds_path.name)
            continue

        ds_with_episodes.append(ds_path)
        fps = info.get("fps", 30)
        ep_lengths = get_episode_lengths(ds_path, fps)
        for ep_idx, frames, seconds in ep_lengths:
            all_episodes.append((ds_path.name, ep_idx, frames, seconds))
        log.info("%s: %d episodes, %d frames", ds_path.name, n_eps, info["total_frames"])

        # Resolve top camera from setup.json (once)
        if top_camera_key is None:
            top_camera_key = resolve_top_camera_key(setup_json, info)

    if top_camera_key:
        log.info("Top camera: %s (from setup.json)", top_camera_key)
    else:
        log.info("Top camera: auto-detect (setup.json not found or no 'top' alias)")

    if not all_episodes:
        log.error("No episodes found. Nothing to do.")
        return

    avg_len, min_thresh, max_thresh, anomalous = find_anomalies(
        all_episodes, args.short_ratio, args.long_ratio,
    )
    log.info("")
    log.info("=" * 60)
    log.info("GLOBAL: %d episodes, avg=%.1fs", len(all_episodes), avg_len)
    log.info("Thresholds: <%.1fs (short) or >%.1fs (long)", min_thresh, max_thresh)

    anomalous_ds_names: set[str] = set()
    if anomalous:
        log.info("")
        log.info("ANOMALOUS EPISODES (%d):", len(anomalous))
        for ds_name, ep_idx, frames, seconds in anomalous:
            reason = "SHORT" if seconds < min_thresh else "LONG"
            log.info("  %s: %s ep=%d %.1fs (%d frames)", reason, ds_name, ep_idx, seconds, frames)
            anomalous_ds_names.add(ds_name)
    else:
        log.info("\nNo anomalous episodes found.")

    # Step 3: boundary frames (top camera only)
    log.info("")
    log.info("Extracting boundary frames...")
    for ds_path in ds_with_episodes:
        out_dir = check_dir / ds_path.name
        log.info("  %s (%d eps)", ds_path.name, json.loads((ds_path / "meta/info.json").read_text())["total_episodes"])
        extract_boundary_frames(ds_path, out_dir, camera_key=top_camera_key)

    # Step 4: extract anomalous episode video clips (top camera only)
    if anomalous:
        review_dir = check_dir / "_anomalous_episodes"
        log.info("")
        log.info("Extracting anomalous episode clips to: %s", review_dir)
        # Group anomalous episodes by dataset
        from collections import defaultdict
        anomalous_by_ds: dict[str, list[int]] = defaultdict(list)
        for ds_name, ep_idx, _, _ in anomalous:
            anomalous_by_ds[ds_name].append(ep_idx)
        for ds_name, ep_indices in sorted(anomalous_by_ds.items()):
            ds_path = root / ds_name
            log.info("  %s eps=%s:", ds_name, ep_indices)
            extract_anomalous_episode_clips(
                ds_path, ep_indices, review_dir, camera_key=top_camera_key,
            )

    if args.dry_run:
        log.info("\n--dry-run: skipping merge.")
        return

    # Step 5: merge (exclude datasets with anomalous episodes)
    merge_names = [ds.name for ds in ds_with_episodes if ds.name not in anomalous_ds_names]
    excluded = [ds.name for ds in ds_with_episodes if ds.name in anomalous_ds_names]

    if excluded:
        log.info("")
        log.info("Excluding %d dataset(s) with anomalous episodes:", len(excluded))
        for name in excluded:
            log.info("  %s", name)

    log.info("")
    log.info("Merging %d dataset(s) into '%s'...", len(merge_names), args.output_name)
    output_path = merge_datasets(root, merge_names, args.output_name)

    # Step 6: set task name
    if args.task:
        set_task_name(output_path, args.task)

    # Step 7: final report
    log.info("")
    report = build_report(output_path)
    print(format_text_report(report))

    # Save report JSON
    report_path = check_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("\nReport JSON saved to: %s", report_path)
    log.info("Boundary frames at: %s", check_dir)
    log.info("Merged dataset at: %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check and merge LeRobot sub-datasets on a single machine.",
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Dataset name prefix to match (e.g. 'zzz_20260402').",
    )
    parser.add_argument(
        "--output-name", required=True,
        help="Name for the merged output dataset (e.g. '0402').",
    )
    parser.add_argument(
        "--datasets-root", default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory containing sub-datasets. Default: %(default)s",
    )
    parser.add_argument(
        "--setup-json", default=str(DEFAULT_SETUP_JSON),
        help="Path to roboclaw setup.json for camera config. Default: %(default)s",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description to set in the merged dataset.",
    )
    parser.add_argument(
        "--short-ratio", type=float, default=SHORT_RATIO,
        help="Episodes shorter than avg * ratio are anomalous. Default: %(default)s (avg=134s -> threshold=40s)",
    )
    parser.add_argument(
        "--long-ratio", type=float, default=LONG_RATIO,
        help="Episodes longer than avg * ratio are anomalous. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report anomalies and extract boundary frames; don't merge.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
