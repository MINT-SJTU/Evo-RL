#!/usr/bin/env python3

"""
Fix historical ARX5 dual-arm raw-train recordings whose VR takeover segments were
saved in left-then-right order instead of the dataset convention right-then-left.

This script only rewrites segments whose source is `vr` and swaps the first 7 dims
with the last 7 dims in both `actions.json` and `states.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DUAL_ARM_DIM = 14
HALF_DIM = 7


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: list | dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _swap_dual_arm_order(step: list[float]) -> list[float]:
    if len(step) != DUAL_ARM_DIM:
        raise ValueError(f"Expected {DUAL_ARM_DIM} dims, got {len(step)}")
    return list(step[HALF_DIM:]) + list(step[:HALF_DIM])


def _is_vr_segment(seg_dir: Path) -> bool:
    meta_path = seg_dir / "meta.json"
    if not meta_path.is_file():
        return seg_dir.name.endswith("_vr")
    meta = _load_json(meta_path)
    return str(meta.get("source")) == "vr"


def _rewrite_segment(seg_dir: Path) -> tuple[int, int]:
    actions_path = seg_dir / "actions.json"
    states_path = seg_dir / "states.json"
    actions = _load_json(actions_path)
    states = _load_json(states_path)
    if not isinstance(actions, list) or not isinstance(states, list):
        raise TypeError(f"Expected list payloads in {seg_dir}")
    if len(actions) != len(states):
        raise ValueError(f"actions/states length mismatch in {seg_dir}: {len(actions)} != {len(states)}")

    fixed_actions = [_swap_dual_arm_order(step) for step in actions]
    fixed_states = [_swap_dual_arm_order(step) for step in states]

    _dump_json(actions_path, fixed_actions)
    _dump_json(states_path, fixed_states)
    return len(fixed_actions), DUAL_ARM_DIM


def _iter_episode_dirs(raw_root: Path) -> list[Path]:
    return sorted([p.parent for p in raw_root.glob("episode_*/meta.json")], key=lambda p: p.name)


def _prepare_output_dir(raw_record_dir: Path, output_dir: Path | None, overwrite: bool) -> Path:
    if output_dir is None:
        return raw_record_dir

    resolved = output_dir.resolve()
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"{resolved} exists; pass --overwrite to replace it.")
        shutil.rmtree(resolved)
    shutil.copytree(raw_record_dir, resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix VR takeover segment ordering in ARX5 dual-arm raw-train recordings."
    )
    parser.add_argument("--raw-record-dir", type=Path, required=True, help="Input dataset root with episode_*/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. If omitted, rewrite in place. "
            "If provided, the input dataset is copied first and only the copy is modified."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting an existing --output-dir before copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which VR segments would be fixed without writing any files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    raw_record_dir = args.raw_record_dir.resolve()
    if not raw_record_dir.is_dir():
        raise FileNotFoundError(str(raw_record_dir))

    work_dir = _prepare_output_dir(raw_record_dir, args.output_dir, args.overwrite)
    logger.info("Working dataset root: %s", work_dir)

    episode_dirs = _iter_episode_dirs(work_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episode_*/meta.json found under {work_dir}")

    fixed_segments = 0
    fixed_steps = 0
    for episode_dir in episode_dirs:
        segments_dir = episode_dir / "segments"
        if not segments_dir.is_dir():
            continue
        segment_dirs = sorted([p for p in segments_dir.glob("segment_*") if p.is_dir()], key=lambda p: p.name)
        for seg_dir in segment_dirs:
            if not _is_vr_segment(seg_dir):
                continue
            if args.dry_run:
                logger.info("Would fix VR segment: %s", seg_dir)
                fixed_segments += 1
                continue
            num_steps, step_dim = _rewrite_segment(seg_dir)
            fixed_segments += 1
            fixed_steps += num_steps
            logger.info("Fixed %s (%d steps, dim=%d)", seg_dir, num_steps, step_dim)

    if args.dry_run:
        logger.info("Dry run complete. VR segments to fix: %d", fixed_segments)
    else:
        logger.info("Done. Fixed %d VR segment(s), %d total step(s).", fixed_segments, fixed_steps)


if __name__ == "__main__":
    main()
