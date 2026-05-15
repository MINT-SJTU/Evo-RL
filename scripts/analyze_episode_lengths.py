#!/usr/bin/env python3

"""Summarize per-episode `length` distribution for a LeRobot dataset or ARX5 raw train record tree.

Optional --min-length / --max-length print episode_index rows that fall outside the band:
length < min_length or length > max_length (episodes within [min_length, max_length] are not listed).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot.datasets.utils import load_episodes

logger = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(r"^segment_(\d+)_(.+)$")


def _resolve_format(root: Path, fmt: str) -> Literal["lerobot", "raw"]:
    if fmt == "lerobot":
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"LeRobot layout requires meta/info.json under {root}")
        return "lerobot"
    if fmt == "raw":
        if not any(root.glob("episode_*/meta.json")):
            raise FileNotFoundError(f"Raw layout requires at least one episode_*/meta.json under {root}")
        return "raw"
    if (root / "meta" / "info.json").is_file():
        return "lerobot"
    if any(root.glob("episode_*/meta.json")):
        return "raw"
    raise ValueError(
        f"Could not detect dataset layout under {root}. "
        "Use --format lerobot (expects meta/info.json) or --format raw (expects episode_*/meta.json)."
    )


def _load_lengths_lerobot(root: Path) -> tuple[np.ndarray, np.ndarray]:
    episodes = load_episodes(root)
    names = episodes.column_names
    if "length" not in names:
        raise KeyError(f"Episodes table has no 'length' column. Available: {sorted(names)}")
    n = len(episodes)
    if n == 0:
        raise ValueError(f"Episodes table is empty under {root / 'meta' / 'episodes'}")
    lengths = np.asarray(episodes["length"], dtype=np.int64).reshape(-1)
    if lengths.shape[0] != n:
        raise RuntimeError(f"Unexpected length column shape for {n} episodes")
    if "episode_index" in names:
        ep_ids = np.asarray(episodes["episode_index"], dtype=np.int64).reshape(-1)
    else:
        logger.warning("Episodes table has no 'episode_index'; using row indices 0..N-1 as ids.")
        ep_ids = np.arange(n, dtype=np.int64)
    return ep_ids, lengths


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


def _raw_episode_dirs(raw_root: Path) -> list[Path]:
    metas = sorted(raw_root.glob("episode_*/meta.json"), key=lambda p: int(p.parent.name.split("_", 1)[1]))
    return [p.parent for p in metas]


def _episode_length_from_raw_segments(episode_dir: Path) -> int:
    segments_dir = episode_dir / "segments"
    if not segments_dir.is_dir():
        logger.warning("Episode %s has no segments/ directory; length 0", episode_dir.name)
        return 0
    segment_dirs = _list_segment_dirs(segments_dir)
    if not segment_dirs:
        logger.warning("Episode %s has no valid segment_* dirs; length 0", episode_dir.name)
        return 0
    total = 0
    for seg in segment_dirs:
        act_path = seg / "actions.json"
        if not act_path.is_file():
            logger.warning("Skip segment %s: missing actions.json", seg)
            continue
        payload = json.loads(act_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"actions.json must be a JSON list: {act_path}")
        total += len(payload)
    return total


def _load_lengths_raw(root: Path) -> tuple[np.ndarray, np.ndarray]:
    episode_dirs = _raw_episode_dirs(root)
    if not episode_dirs:
        raise ValueError(f"No episode_*/meta.json under {root}")
    ep_ids: list[int] = []
    lengths_list: list[int] = []
    for ep in episode_dirs:
        ep_ids.append(int(ep.name.split("_", 1)[1]))
        lengths_list.append(_episode_length_from_raw_segments(ep))
    return np.asarray(ep_ids, dtype=np.int64), np.asarray(lengths_list, dtype=np.int64)


def _print_stats(lengths: np.ndarray) -> None:
    print(f"num_episodes: {lengths.shape[0]}")
    print(f"length min:    {int(lengths.min())}")
    print(f"length max:    {int(lengths.max())}")
    print(f"length mean:   {float(lengths.mean()):.4f}")
    print(f"length median: {float(np.median(lengths)):.4f}")
    print(f"length std:    {float(lengths.std()):.4f}")
    for q in (25, 75, 90, 95, 99):
        v = float(np.percentile(lengths, q))
        print(f"length p{q}:    {v:.4f}")


def _print_out_of_range(
    episode_ids: np.ndarray,
    lengths: np.ndarray,
    min_len: int | None,
    max_len: int | None,
) -> None:
    if min_len is None and max_len is None:
        return
    below: list[tuple[int, int]] = []
    above: list[tuple[int, int]] = []
    for eid, length in zip(episode_ids.tolist(), lengths.tolist(), strict=True):
        if min_len is not None and length < min_len:
            below.append((int(eid), int(length)))
        if max_len is not None and length > max_len:
            above.append((int(eid), int(length)))
    print("\nLength threshold check:")
    if min_len is not None:
        print(f"  min_length = {min_len}  ->  episodes with length < min: {len(below)}")
        for eid, length in sorted(below, key=lambda t: t[0]):
            print(f"    episode_index={eid}  length={length}")
    if max_len is not None:
        print(f"  max_length = {max_len}  ->  episodes with length > max: {len(above)}")
        for eid, length in sorted(above, key=lambda t: t[0]):
            print(f"    episode_index={eid}  length={length}")


def _print_histogram(lengths: np.ndarray, bins: int) -> None:
    if bins < 1:
        raise ValueError("--bins must be >= 1")
    counts, edges = np.histogram(lengths, bins=bins)
    print(f"\nHistogram ({bins} bins, [low, high)):")
    cmax = int(counts.max()) if counts.size else 1
    width = 50
    for i, c in enumerate(counts):
        lo, hi = float(edges[i]), float(edges[i + 1])
        bar_n = int(round(c * width / cmax)) if cmax > 0 else 0
        bar = "#" * bar_n
        print(f"  [{lo:10.2f}, {hi:10.2f}): {int(c):6d}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset root (meta/info.json) or raw record root (episode_* / ...).",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "lerobot", "raw"),
        default="auto",
        help="Dataset layout (default: auto-detect).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="If set, list episodes whose length is strictly less than this value.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set, list episodes whose length is strictly greater than this value.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary stats and histogram (warnings to stderr via logging).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    if args.min_length is not None and args.max_length is not None and args.min_length > args.max_length:
        parser.error("--min-length must be <= --max-length when both are set.")

    kind = _resolve_format(root, args.format)
    if kind == "lerobot":
        episode_ids, lengths = _load_lengths_lerobot(root)
    else:
        episode_ids, lengths = _load_lengths_raw(root)

    print(f"dataset_root: {root}")
    print(f"format:       {kind}")
    _print_stats(lengths)
    _print_out_of_range(episode_ids, lengths, args.min_length, args.max_length)
    _print_histogram(lengths, args.bins)


if __name__ == "__main__":
    main()
