#!/usr/bin/env python3

"""
Run the full ARX5 raw-train dataset processing pipeline:

1. Merge consecutive same-source segments.
2. Delete episodes with missing/invalid `episode_success`.
3. Convert the cleaned recording into a LeRobotDataset.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_repo_id(raw_record_dir: Path) -> str:
    return raw_record_dir.name


def _run_step(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process ARX5 raw-train recordings end-to-end.")
    parser.add_argument("--raw-record-dir", type=Path, required=True, help="Input raw recording root.")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Intermediate merged root. Default: <raw-record-dir>_merged",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Final LeRobotDataset root. Default: <raw-record-dir>_lerobot",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset repo_id passed to convert. Default: basename of raw-record-dir",
    )
    parser.add_argument("--robot-type", type=str, default="arx5")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing merged-dir/output-root before processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    raw_record_dir = args.raw_record_dir.resolve()
    if not raw_record_dir.is_dir():
        raise FileNotFoundError(str(raw_record_dir))

    merged_dir = args.merged_dir.resolve() if args.merged_dir is not None else raw_record_dir.with_name(
        raw_record_dir.name + "_merged"
    )
    output_root = args.output_root.resolve() if args.output_root is not None else raw_record_dir.with_name(
        raw_record_dir.name + "_lerobot"
    )
    repo_id = args.repo_id if args.repo_id is not None else _default_repo_id(raw_record_dir)

    if args.overwrite and output_root.exists():
        logger.info("Deleting existing output root: %s", output_root)
        shutil.rmtree(output_root)

    merge_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.merge_arx5_raw_train_segments",
        "--raw-record-dir",
        str(raw_record_dir),
        "--output-dir",
        str(merged_dir),
    ]
    if args.overwrite:
        merge_cmd.append("--overwrite")
    _run_step(merge_cmd)

    prune_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.prune_arx5_raw_train_missing_success",
        "--raw-record-dir",
        str(merged_dir),
    ]
    _run_step(prune_cmd)

    convert_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.convert_arx5_raw_train_to_lerobot_dataset",
        "--raw-record-dir",
        str(merged_dir),
        "--output-root",
        str(output_root),
        "--repo-id",
        repo_id,
        "--robot-type",
        args.robot_type,
    ]
    _run_step(convert_cmd)

    logger.info("Merged copy kept at: %s", merged_dir)
    logger.info("LeRobotDataset written to: %s", output_root)


if __name__ == "__main__":
    main()
