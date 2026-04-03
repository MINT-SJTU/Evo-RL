#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Merge consecutive raw-training segments that share the same `source` (policy / vr).

`lerobot_arx5_infer.py` / `lerobot_arx5_dual_infer.py` write one segment per policy chunk
and per VR stretch; this script rewrites a copy of the recording with fewer segment folders
while preserving timestep order and JSON contents.

After merging, run `convert_arx5_raw_train_to_lerobot_dataset.py` on `--output-dir`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_SEGMENT_RE = re.compile(r"^segment_(\d+)_(.+)$")


def _parse_segment_dir(name: str) -> tuple[int, str] | None:
    m = _SEGMENT_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_dirs(raw_root: Path) -> list[Path]:
    metas = sorted(raw_root.glob("episode_*/meta.json"), key=lambda p: p.parent.name)
    return [p.parent for p in metas]


def _list_segment_dirs(segments_dir: Path) -> list[Path]:
    dirs: list[tuple[int, Path]] = []
    for p in segments_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_segment_dir(p.name)
        if parsed is None:
            logger.warning("Skip non-segment dir: %s", p)
            continue
        idx, _src = parsed
        dirs.append((idx, p))
    dirs.sort(key=lambda x: x[0])
    return [p for _, p in dirs]


def _merge_segment_group(group: list[Path], out_dir: Path, source: str, dt_s: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=False)

    all_actions: list[list[float]] = []
    all_states: list[list[float]] = []
    frame_off = 0
    camera_names: list[str] | None = None

    for seg in group:
        actions = _load_json(seg / "actions.json")
        states = _load_json(seg / "states.json")
        if not isinstance(actions, list) or not isinstance(states, list):
            raise TypeError(f"actions/states must be lists in {seg}")
        if len(actions) != len(states):
            raise ValueError(f"Length mismatch in {seg}: actions={len(actions)} states={len(states)}")

        seg_meta = _load_json(seg / "meta.json")
        if str(seg_meta.get("source")) != source:
            raise ValueError(f"meta.source in {seg} != dirname source {source!r}")
        if float(seg_meta.get("dt_s", dt_s)) != float(dt_s):
            raise ValueError(f"dt_s mismatch in {seg}: expected {dt_s}, got {seg_meta.get('dt_s')}")

        imgs = seg / "images"
        cams = sorted([p.name for p in imgs.iterdir() if p.is_dir()])
        if camera_names is None:
            camera_names = cams
            for cam in camera_names:
                (images_root / cam).mkdir(parents=True, exist_ok=False)
        elif cams != camera_names:
            raise ValueError(f"Camera set mismatch in {seg}: {cams} vs {camera_names}")

        for step_idx in range(len(actions)):
            for cam in camera_names:
                src_png = imgs / cam / f"frame_{step_idx:06d}.png"
                if not src_png.is_file():
                    raise FileNotFoundError(str(src_png))
                dst_png = images_root / cam / f"frame_{frame_off + step_idx:06d}.png"
                shutil.copy2(src_png, dst_png)

        all_actions.extend(actions)
        all_states.extend(states)
        frame_off += len(actions)

    (out_dir / "actions.json").write_text(json.dumps(all_actions, indent=2) + "\n", encoding="utf-8")
    (out_dir / "states.json").write_text(json.dumps(all_states, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"source": source, "dt_s": dt_s}, indent=2) + "\n",
        encoding="utf-8",
    )


def merge_episode(episode_dir: Path, out_episode: Path) -> None:
    meta_path = episode_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(str(meta_path))
    episode_meta = _load_json(meta_path)
    if not isinstance(episode_meta, dict):
        raise TypeError(f"episode meta must be object: {meta_path}")
    dt_s = float(episode_meta["dt_s"])

    out_episode.mkdir(parents=True, exist_ok=False)
    shutil.copy2(meta_path, out_episode / "meta.json")

    segments_in = episode_dir / "segments"
    if not segments_in.is_dir():
        raise FileNotFoundError(str(segments_in))

    segment_dirs = _list_segment_dirs(segments_in)
    if not segment_dirs:
        raise RuntimeError(f"No segments under {segments_in}")

    out_segments = out_episode / "segments"
    out_segments.mkdir(parents=True, exist_ok=False)

    out_idx = 0
    i = 0
    while i < len(segment_dirs):
        seg = segment_dirs[i]
        parsed = _parse_segment_dir(seg.name)
        if parsed is None:
            i += 1
            continue
        _idx, source = parsed
        group = [seg]
        j = i + 1
        while j < len(segment_dirs):
            nxt = segment_dirs[j]
            p2 = _parse_segment_dir(nxt.name)
            if p2 is None:
                break
            _i2, src2 = p2
            if src2 != source:
                break
            nmeta = _load_json(nxt / "meta.json")
            if str(nmeta.get("source")) != source:
                break
            group.append(nxt)
            j += 1

        out_name = f"segment_{out_idx:04d}_{source}"
        _merge_segment_group(group, out_segments / out_name, source=source, dt_s=dt_s)
        logger.info(
            "Merged %d segment(s) -> %s (%s)",
            len(group),
            out_episode.name + "/segments/" + out_name,
            source,
        )
        out_idx += 1
        i = j


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge consecutive same-source segments in ARX5 raw train recordings."
    )
    parser.add_argument("--raw-record-dir", type=Path, required=True, help="Input root with episode_*/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root (created; must not exist unless --overwrite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir if it exists, then write merged data.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    src = args.raw_record_dir.resolve()
    dst = args.output_dir.resolve()
    if not src.is_dir():
        raise FileNotFoundError(str(src))
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    eps = _episode_dirs(src)
    if not eps:
        raise RuntimeError(f"No episode_*/meta.json under {src}")

    for ep in eps:
        merge_episode(ep, dst / ep.name)

    logger.info("Done. Merged dataset root: %s", dst)


if __name__ == "__main__":
    main()
