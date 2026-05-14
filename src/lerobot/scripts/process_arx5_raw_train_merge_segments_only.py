#!/usr/bin/env python3

"""
Merge ARX5 raw-train recording segments only.

This script is a small wrapper around `merge_arx5_raw_train_segments`.
It writes a merged raw-recording copy and does not prune episodes or convert
the result into a LeRobotDataset.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_step(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ARX5 raw-train segments without LeRobot conversion.")
    parser.add_argument("--raw-record-dir", type=Path, required=True, help="Input raw recording root.")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Output merged raw recording root. Default: <raw-record-dir>_merged",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing merged-dir before processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    raw_record_dir = args.raw_record_dir.expanduser().resolve()
    if not raw_record_dir.is_dir():
        raise FileNotFoundError(str(raw_record_dir))

    merged_dir = (
        args.merged_dir.expanduser().resolve()
        if args.merged_dir is not None
        else raw_record_dir.with_name(raw_record_dir.name + "_merged")
    )

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
    logger.info("Merged raw recording written to: %s", merged_dir)


if __name__ == "__main__":
    main()
