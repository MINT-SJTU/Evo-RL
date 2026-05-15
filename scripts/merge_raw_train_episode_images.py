#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Flatten camera PNGs under each episode's segments into one timeline.

Expects ``TrainRawEpisodeRecorder`` layout from ``lerobot_arx5_dual_infer`` /
``lerobot_arx5_infer``::

    episode_{idx:04d}/segments/segment_{seg:04d}_{policy|vr}/images/<camera>/frame_XXXXXX.png

Does not support LeRobot parquet/video datasets (use after export from raw only).

Outputs::

    <output_root>/episode_XXXX/images/<camera>/frame_XXXXXX.png

with frames renumbered contiguously across segments in segment-index order.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(r"^segment_(\d+)_(.+)$")
_FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def _parse_segment_dir(name: str) -> tuple[int, str] | None:
    m = _SEGMENT_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def _list_segment_dirs(segments_dir: Path) -> list[Path]:
    dirs: list[tuple[int, Path]] = []
    for p in segments_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_segment_dir(p.name)
        if parsed is None:
            logger.warning("Skip non-segment directory: %s", p)
            continue
        idx, _src = parsed
        dirs.append((idx, p))
    dirs.sort(key=lambda x: x[0])
    return [p for _, p in dirs]


def _episode_dirs(raw_root: Path) -> list[Path]:
    metas = sorted(
        raw_root.glob("episode_*/meta.json"),
        key=lambda p: int(p.parent.name.split("_", 1)[1]),
    )
    return [p.parent for p in metas]


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    if (root / "meta" / "info.json").is_file() and not any(root.glob("episode_*/meta.json")):
        raise ValueError(
            "Directory looks like a LeRobot dataset (meta/info.json) without episode_*/meta.json. "
            "This script only supports TrainRawEpisodeRecorder raw trees: "
            "episode_*/segments/segment_*_*/images/<camera>/frame_*.png"
        )


def _camera_names_from_segment(seg_images_root: Path) -> list[str]:
    if not seg_images_root.is_dir():
        return []
    names = sorted(p.name for p in seg_images_root.iterdir() if p.is_dir())
    return names


def _list_png_frames(cam_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    if not cam_dir.is_dir():
        return []
    for p in cam_dir.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if m is None:
            continue
        frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    ordered = [path for _, path in frames]
    step_indices = [idx for idx, _ in frames]
    if len(step_indices) > 1:
        diffs = [b - a for a, b in zip(step_indices, step_indices[1:], strict=False)]
        if any(d <= 0 for d in diffs):
            raise ValueError(f"Non-monotonic frame indices under {cam_dir}")
    return ordered


def _link_like(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def merge_episode_images(
    episode_dir: Path,
    output_episode_dir: Path,
    *,
    copy_mode: str,
    dry_run: bool,
) -> dict[str, int]:
    segments_dir = episode_dir / "segments"
    if not segments_dir.is_dir():
        raise FileNotFoundError(f"Missing segments/: {segments_dir}")

    segment_dirs = _list_segment_dirs(segments_dir)
    if not segment_dirs:
        raise RuntimeError(f"No segment_* dirs under {segments_dir}")

    camera_names: list[str] | None = None
    global_off = 0
    counts_per_cam: dict[str, int] = {}

    for seg in segment_dirs:
        img_root = seg / "images"
        cams_here = _camera_names_from_segment(img_root)

        if camera_names is None:
            if not cams_here:
                logger.warning("Segment %s has no camera directories under images/; skip", seg.name)
                continue
            camera_names = cams_here
            for c in camera_names:
                counts_per_cam[c] = 0
        else:
            if set(cams_here) != set(camera_names):
                raise ValueError(
                    f"Camera set mismatch in {seg}: expected {sorted(camera_names)}, got {sorted(cams_here)}"
                )

        frame_lists = {cam: _list_png_frames(img_root / cam) for cam in camera_names}
        lengths = {cam: len(frame_lists[cam]) for cam in camera_names}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Per-camera frame counts differ in segment {seg.name}: {lengths}"
            )
        n_frames = lengths[camera_names[0]]
        if n_frames == 0:
            logger.warning("Segment %s has zero PNG frames per camera; skip", seg.name)
            continue

        for cam in camera_names:
            paths = frame_lists[cam]
            for local_i, src in enumerate(paths):
                dst_name = f"frame_{global_off + local_i:06d}.png"
                dst = output_episode_dir / "images" / cam / dst_name
                if dry_run:
                    pass
                else:
                    _link_like(src, dst, copy_mode)
                counts_per_cam[cam] += 1

        global_off += n_frames

    if camera_names is None:
        raise RuntimeError(f"No usable images under any segment in {episode_dir}")

    return counts_per_cam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of TrainRawEpisodeRecorder dataset (episode_*/meta.json).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory (default: sibling folder named '<dataset-root.name>_merged_images'). "
            "Episode trees are written as <output-root>/episode_XXXX/images/..."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned merges without writing files.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place merged PNGs (default: copy).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = args.dataset_root.expanduser().resolve()
    _validate_dataset_root(root)

    out_root = args.output_root
    if out_root is None:
        out_root = root.parent / f"{root.name}_merged_images"
    else:
        out_root = out_root.expanduser().resolve()

    episodes = _episode_dirs(root)
    if not episodes:
        raise ValueError(f"No episode_*/meta.json found under {root}")

    logging.info("dataset_root=%s", root)
    logging.info("output_root=%s", out_root)
    logging.info("episodes=%d copy_mode=%s dry_run=%s", len(episodes), args.copy_mode, args.dry_run)

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    total_png = 0
    for ep in episodes:
        out_ep = out_root / ep.name
        if args.dry_run:
            counts = merge_episode_images(ep, out_ep, copy_mode=args.copy_mode, dry_run=True)
        else:
            out_ep.mkdir(parents=True, exist_ok=False)
            counts = merge_episode_images(ep, out_ep, copy_mode=args.copy_mode, dry_run=False)
        ep_total = sum(counts.values())
        total_png += ep_total
        cams_s = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logging.info("%s -> %d png writes (%s)", ep.name, ep_total, cams_s)

    logging.info("Done. total_png_operations=%d", total_png)


if __name__ == "__main__":
    main()
