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
Convert ARX5 raw training recordings (recorded by `lerobot_arx5_infer.py --raw-train-record-dir`)
into a LeRobotDataset directory that can be used by training scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_episode_dirs(raw_root: Path) -> list[Path]:
    eps = sorted([p for p in raw_root.glob("episode_*/meta.json")], key=lambda x: x.parent.name)
    return [p.parent for p in eps]


def _episode_success_to_label(success_value: int) -> str:
    # LeRobot uses canonical labels: "success"/"failure"
    if int(success_value) == 1:
        return "success"
    return "failure"


def _build_dataset_features(
    *,
    action_keys: list[str],
    camera_shapes: dict[str, tuple[int, int, int]],
    images_only: bool,
) -> dict:
    # Hardware observation features:
    # - state: joint/gripper values
    # - images: camera frames
    obs_hw_features = {k: float for k in action_keys}
    obs_hw_features.update(camera_shapes)

    action_hw_features = {k: float for k in action_keys}

    use_video = not images_only
    obs_features = hw_to_dataset_features(obs_hw_features, prefix=OBS_STR, use_video=use_video)
    action_features = hw_to_dataset_features(action_hw_features, prefix=ACTION, use_video=use_video)
    dataset_features = combine_feature_dicts(obs_features, action_features)

    # Human-in-loop compatible features (stable schema for ACP / intervention conditioning).
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_keys),),
        "names": action_keys,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }
    dataset_features["complementary_info.collector_policy_id"] = {
        "dtype": "string",
        "shape": (1,),
        "names": ["collector_policy_id"],
    }

    return dataset_features


def _read_camera_shape(png_path: Path) -> tuple[int, int, int]:
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape for {png_path}: {arr.shape}")
    return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-record-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Where LeRobotDataset will be written.")
    parser.add_argument("--repo-id", type=str, required=True, help="Local dataset repo_id name.")
    parser.add_argument("--robot-type", type=str, default="arx5")
    parser.add_argument("--vcodec", type=str, default="libsvtav1")
    parser.add_argument("--skip-missing-success", action="store_true", default=False)
    parser.add_argument(
        "--images-only",
        action="store_true",
        default=False,
        help="Store camera observations as images (PNG) instead of videos.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)

    raw_record_dir: Path = args.raw_record_dir
    if not raw_record_dir.exists():
        raise FileNotFoundError(raw_record_dir)

    episode_dirs = _load_episode_dirs(raw_record_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* dirs found under {raw_record_dir}")

    first_meta = _load_json(episode_dirs[0] / "meta.json")
    action_keys = list(first_meta["action_keys"])
    camera_names = list(first_meta["camera_names"])
    fps = int(first_meta.get("fps", int(round(1.0 / first_meta["dt_s"]))))
    dt_s = float(first_meta["dt_s"])

    # Infer camera shapes from the first segment's frame_000000 images.
    first_segments_dir = episode_dirs[0] / "segments"
    first_segment_dirs = sorted([p for p in first_segments_dir.glob("segment_*")], key=lambda x: x.name)
    if not first_segment_dirs:
        raise RuntimeError("First episode has no segments; cannot infer camera shapes.")
    first_segment = first_segment_dirs[0]
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    for cam in camera_names:
        png_path = first_segment / "images" / cam / "frame_000000.png"
        camera_shapes[cam] = _read_camera_shape(png_path)

    dataset_features = _build_dataset_features(
        action_keys=action_keys,
        camera_shapes=camera_shapes,
        images_only=args.images_only,
    )

    dataset = LeRobotDataset.create(
        args.repo_id,
        fps,
        root=args.output_root,
        robot_type=args.robot_type,
        features=dataset_features,
        use_videos=not args.images_only,
        vcodec=args.vcodec,
    )

    # Keep dataset frames consistent with the recorded dt.
    logger.info("Start converting %d episode(s) with fps=%d (dt_s=%.6f)", len(episode_dirs), fps, dt_s)

    for ep_idx, episode_dir in enumerate(episode_dirs):
        meta = _load_json(episode_dir / "meta.json")
        episode_success_int = meta.get("episode_success", None)
        if episode_success_int is None:
            msg = f"Episode {episode_dir} has episode_success=None."
            if args.skip_missing_success:
                logger.warning("%s Skipping.", msg)
                continue
            raise RuntimeError(msg)

        episode_success_label = _episode_success_to_label(int(episode_success_int))
        task = str(meta["task"])
        collector_policy_id_policy = str(meta.get("collector_policy_id_policy", "policy"))
        collector_policy_id_human = str(meta.get("collector_policy_id_human", "human"))

        segments_dir = episode_dir / "segments"
        segment_dirs = sorted([p for p in segments_dir.glob("segment_*")], key=lambda x: x.name)

        logger.info("Converting episode %s (%d segments)", episode_dir.name, len(segment_dirs))

        for seg_dir in segment_dirs:
            seg_meta = _load_json(seg_dir / "meta.json")
            source = str(seg_meta["source"])
            is_intervention = 1.0 if source == "vr" else 0.0
            collector_policy_id = collector_policy_id_human if source == "vr" else collector_policy_id_policy

            actions_vectors = _load_json(seg_dir / "actions.json")
            states_vectors = _load_json(seg_dir / "states.json")
            if len(actions_vectors) != len(states_vectors):
                raise ValueError(f"Actions/states length mismatch in {seg_dir}")

            for step_idx, (action_vector, state_vector) in enumerate(zip(actions_vectors, states_vectors)):
                action_values = {k: float(action_vector[i]) for i, k in enumerate(action_keys)}
                obs_state_values = {k: float(state_vector[i]) for i, k in enumerate(action_keys)}
                observation_images: dict[str, np.ndarray] = {}
                for cam in camera_names:
                    img_path = seg_dir / "images" / cam / f"frame_{step_idx:06d}.png"
                    img_arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                    observation_images[cam] = img_arr

                observation_values: dict = {**obs_state_values, **observation_images}

                observation_frame = build_dataset_frame(dataset.features, observation_values, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)

                # Store policy_action as policy output when source is policy, else zeros.
                policy_action_vec = (
                    np.asarray(action_vector, dtype=np.float32)
                    if source == "policy"
                    else np.zeros((len(action_keys),), dtype=np.float32)
                )

                frame = {
                    **observation_frame,
                    **action_frame,
                    "complementary_info.policy_action": policy_action_vec,
                    "complementary_info.is_intervention": np.asarray([is_intervention], dtype=np.float32),
                    "complementary_info.state": np.asarray([is_intervention], dtype=np.float32),
                    "complementary_info.collector_policy_id": str(collector_policy_id),
                    "task": task,
                }

                dataset.add_frame(frame)

        dataset.save_episode(extra_episode_metadata={"episode_success": episode_success_label})

    dataset.finalize()
    logger.info("Conversion finished. Dataset written to %s", args.output_root)


if __name__ == "__main__":
    main()

