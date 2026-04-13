#!/usr/bin/env python3

"""
Delete raw-training episodes whose `meta.json` does not contain a valid `episode_success` label.

This is intended for ARX5 raw recordings produced by `lerobot_arx5_infer.py --raw-train-record-dir`
or a merged copy produced by `merge_arx5_raw_train_segments.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_dirs(raw_root: Path) -> list[Path]:
    metas = sorted(raw_root.glob("episode_*/meta.json"), key=lambda p: p.parent.name)
    return [p.parent for p in metas]


def _is_valid_episode_success(value: object) -> bool:
    return value in (0, 1, "0", "1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete ARX5 raw-training episodes with missing or invalid episode_success labels."
    )
    parser.add_argument("--raw-record-dir", type=Path, required=True, help="Root containing episode_*/meta.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which episodes would be deleted.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    raw_root = args.raw_record_dir.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(str(raw_root))

    episode_dirs = _episode_dirs(raw_root)
    if not episode_dirs:
        raise RuntimeError(f"No episode_*/meta.json under {raw_root}")

    deleted = 0
    kept = 0
    for episode_dir in episode_dirs:
        meta = _load_json(episode_dir / "meta.json")
        episode_success = meta.get("episode_success")
        if _is_valid_episode_success(episode_success):
            kept += 1
            continue

        logger.info("Deleting %s: invalid episode_success=%r", episode_dir, episode_success)
        if not args.dry_run:
            shutil.rmtree(episode_dir)
        deleted += 1

    logger.info("Done. kept=%d deleted=%d dry_run=%s", kept, deleted, args.dry_run)


if __name__ == "__main__":
    main()
