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
List PNG sizes (H, W) for ARX5 raw train recordings and find frames that differ from the
reference used by `convert_arx5_raw_train_to_lerobot_dataset.py`:
first episode, first segment (lexicographic `segment_*`), `frame_000000.png` per camera.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_dirs(raw_root: Path) -> list[Path]:
    metas = sorted(raw_root.glob("episode_*/meta.json"), key=lambda p: p.parent.name)
    return [p.parent for p in metas]


def _png_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return (int(h), int(w))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-record-dir", type=Path, required=True)
    parser.add_argument(
        "--list-mismatch",
        action="store_true",
        help="Print every PNG path whose (H,W) differs from the convert-script reference.",
    )
    args = parser.parse_args()

    raw_root: Path = args.raw_record_dir.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(str(raw_root))

    eps = _episode_dirs(raw_root)
    if not eps:
        raise RuntimeError(f"No episode_*/meta.json under {raw_root}")

    first_ep = eps[0]
    ep_meta = _load_json(first_ep / "meta.json")
    camera_names = list(ep_meta["camera_names"])

    segments_dir = first_ep / "segments"
    first_segments = sorted(segments_dir.glob("segment_*"), key=lambda p: p.name)
    if not first_segments:
        raise RuntimeError(f"No segments under {segments_dir}")

    ref_segment = first_segments[0]
    ref_by_cam: dict[str, tuple[int, int]] = {}
    for cam in camera_names:
        ref_png = ref_segment / "images" / cam / "frame_000000.png"
        if not ref_png.is_file():
            raise FileNotFoundError(
                f"Missing reference frame for convert script: {ref_png}"
            )
        ref_by_cam[cam] = _png_hw(ref_png)

    # shape -> paths, per camera
    by_cam_shape: dict[str, dict[tuple[int, int], list[Path]]] = {
        cam: defaultdict(list) for cam in camera_names
    }

    for ep in eps:
        segs = sorted((ep / "segments").glob("segment_*"), key=lambda p: p.name)
        for seg in segs:
            for cam in camera_names:
                cam_dir = seg / "images" / cam
                if not cam_dir.is_dir():
                    continue
                for png in sorted(cam_dir.glob("frame_*.png")):
                    hw = _png_hw(png)
                    by_cam_shape[cam][hw].append(png)

    print("Reference (same as convert: first episode, first segment, frame_000000):")
    for cam in camera_names:
        h, w = ref_by_cam[cam]
        print(f"  {cam}: HxW = {h} x {w}  (numpy array shape ({h}, {w}, 3))")

    print("\nPer-camera size histogram (all episodes/segments):")
    mismatch_total = 0
    for cam in camera_names:
        shapes = by_cam_shape[cam]
        if not shapes:
            print(f"  {cam}: no PNGs found")
            continue
        ref = ref_by_cam[cam]
        print(f"  {cam}:")
        for hw in sorted(shapes.keys(), key=lambda t: (-len(shapes[t]), t)):
            paths = shapes[hw]
            tag = "  REF" if hw == ref else "  !!!"
            print(f"    {tag} ({hw[0]}, {hw[1]}): {len(paths)} frame(s)")
            if hw != ref:
                mismatch_total += len(paths)

    print("\nMismatch vs reference (count):")
    for cam in camera_names:
        ref = ref_by_cam[cam]
        bad = 0
        for hw, paths in by_cam_shape[cam].items():
            if hw != ref:
                bad += len(paths)
        print(f"  {cam}: {bad}")

    # One line per segment folder that contains any wrong-sized frame (any camera).
    seg_mismatch: dict[Path, set[tuple[int, int]]] = defaultdict(set)
    for cam in camera_names:
        ref = ref_by_cam[cam]
        for hw, paths in by_cam_shape[cam].items():
            if hw == ref:
                continue
            for p in paths:
                seg_dir = p.parent.parent.parent
                seg_mismatch[seg_dir].add(hw)

    if seg_mismatch:
        print("\nSegments containing non-reference sizes (path -> distinct (H,W) seen):")
        for seg_dir in sorted(seg_mismatch.keys(), key=lambda p: str(p)):
            shapes = ", ".join(f"({h},{w})" for h, w in sorted(seg_mismatch[seg_dir]))
            print(f"  {seg_dir}")
            print(f"    shapes: {shapes}")

    if args.list_mismatch:
        print("\nMismatch file paths:")
        for cam in camera_names:
            ref = ref_by_cam[cam]
            for hw, paths in sorted(by_cam_shape[cam].items(), key=lambda x: x[0]):
                if hw == ref:
                    continue
                print(f"  --- {cam} ({hw[0]}, {hw[1]}) expected {ref} ---")
                for p in paths:
                    print(p)

    if mismatch_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
