#!/usr/bin/env python

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

"""Run LeRobot Pi0.5 inference directly on an ARX5 arm."""

import argparse
import atexit
import json
import logging
import queue
import select
import shutil
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from enum import Enum, auto
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.arx5_follower import ARX5_REAL_STATE_KEYS, ARX5Follower, ARX5FollowerConfig
from lerobot.robots.arx5_follower.arx5_runtime import Arx5Runtime, SharedARXX5Interface
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device
from lerobot.configs.policies import PreTrainedConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "lerobot/pi05_base"
DEFAULT_LOCAL_CAMERA_MAP = {
    "base_0_rgb": "side",
    "left_wrist_0_rgb": "wrist",
    "right_wrist_0_rgb": "front",
}
ARX5_MULTI_CUPS_CAMERA_MAP = {
    "base": "front",
    "right_wrist": "wrist",
}
DEFAULT_STATS_PATH = files("lerobot.robots.arx5_follower").joinpath("pi05_arx5_default_stats.json")
OBS_IMAGE_PREFIX = "observation.images."


class LoopState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    TEACHING = auto()


class KeyboardListener:
    """Non-blocking keyboard input for Linux terminals."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings: list[int] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        atexit.register(self._atexit_restore)
        self.resume()

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (ValueError, OSError):
                break
            if self._stop.is_set():
                break
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char:
                self._queue.put(char.lower())

    def get_key(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def pause(self) -> None:
        """Stop the listener thread and restore cooked terminal mode (e.g. before VR teleop)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except termios.error:
                pass
            self._old_settings = None

    def resume(self) -> None:
        """Re-enable cbreak mode and restart the listener thread."""
        self.pause()
        self._stop.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error:
            self._old_settings = None
            logger.warning(
                "KeyboardListener: stdin is not a TTY; keyboard hotkeys ([R]/[D]/[Space]/...) are disabled. "
                "Run the script attached to a real terminal (e.g. ssh/pty), not piped or detached stdin."
            )
            return
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def restore(self) -> None:
        """Alias for full terminal cleanup (atexit)."""
        self.pause()

    def _atexit_restore(self) -> None:
        self.pause()


def _parse_camera_specs(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid camera spec '{spec}', expected name:value")
        name, value = spec.split(":", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def _get_policy_image_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    return [
        key.removeprefix(OBS_IMAGE_PREFIX)
        for key in policy_cfg.input_features
        if key.startswith(OBS_IMAGE_PREFIX)
    ]


def _get_required_policy_camera_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    return [key for key in _get_policy_image_keys(policy_cfg) if not key.startswith("empty_camera_")]


def _resolve_local_camera_map(policy_cfg: PreTrainedConfig) -> dict[str, str]:
    required_policy_keys = _get_required_policy_camera_keys(policy_cfg)
    required_policy_key_set = set(required_policy_keys)

    if required_policy_key_set.issubset(DEFAULT_LOCAL_CAMERA_MAP):
        return {policy_key: DEFAULT_LOCAL_CAMERA_MAP[policy_key] for policy_key in required_policy_keys}

    if required_policy_key_set.issubset(ARX5_MULTI_CUPS_CAMERA_MAP):
        return {policy_key: ARX5_MULTI_CUPS_CAMERA_MAP[policy_key] for policy_key in required_policy_keys}

    raise ValueError(
        "Unsupported ARX5 camera layout for this checkpoint. "
        f"Policy expects {sorted(required_policy_key_set)} but the runtime only knows how to map "
        f"{sorted(DEFAULT_LOCAL_CAMERA_MAP)} or {sorted(ARX5_MULTI_CUPS_CAMERA_MAP)}."
    )


def _list_realsense_cameras() -> None:
    try:
        cameras = RealSenseCamera.find_cameras()
    except Exception as error:
        print(f"Unable to query RealSense cameras: {error}")
        return
    if not cameras:
        print("No RealSense cameras found.")
        return
    print(f"Found {len(cameras)} RealSense camera(s):")
    for camera in cameras:
        default_stream = camera.get("default_stream_profile", {})
        print(
            f"  {camera['name']} serial={camera['id']} "
            f"default={default_stream.get('width')}x{default_stream.get('height')}@{default_stream.get('fps')}"
        )


def _make_camera_configs(
    *,
    camera_specs: dict[str, str],
    local_camera_map: dict[str, str],
    required_policy_camera_keys: list[str],
    use_usb_cams: bool,
    width: int,
    height: int,
    fps: int,
    flipped_local_cameras: set[str],
) -> dict[str, Any]:
    configs = {}
    local_to_policy = {local_name: policy_name for policy_name, local_name in local_camera_map.items()}
    for local_name, source in camera_specs.items():
        if local_name not in local_to_policy:
            raise ValueError(
                f"Unknown local camera '{local_name}'. Expected one of {sorted(local_to_policy)}."
            )
        policy_name = local_to_policy[local_name]
        if policy_name not in required_policy_camera_keys:
            logger.info(
                "Ignoring local camera '%s' because policy does not consume '%s'.",
                local_name,
                policy_name,
            )
            continue
        rotation = 180 if local_name in flipped_local_cameras else 0
        if use_usb_cams:
            configs[policy_name] = OpenCVCameraConfig(
                index_or_path=int(source),
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
        else:
            configs[policy_name] = RealSenseCameraConfig(
                serial_number_or_name=source,
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
    missing = set(required_policy_camera_keys) - set(configs)
    if missing:
        raise ValueError(
            "Missing required cameras for this policy: "
            f"{sorted(local_camera_map[name] for name in missing)}"
        )
    return configs


def _load_stats(stats_path: Path) -> dict[str, dict[str, list[float]]]:
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _load_policy_bundle(
    policy_path: str,
    device_override: str | None,
    stats_path: Path | None,
    policy_cfg: PreTrainedConfig | None = None,
):
    if policy_cfg is None:
        logger.info("Loading PI05 policy config from %s", policy_path)
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    else:
        logger.info("Using preloaded PI05 policy config from %s", policy_path)
    if policy_cfg.type != "pi05":
        raise ValueError(f"Expected a pi05 policy, got '{policy_cfg.type}'.")

    target_device = torch.device(device_override) if device_override else auto_select_torch_device()
    policy_cfg.device = str(target_device)
    policy_cfg.pretrained_path = policy_path

    logger.info("Instantiating policy on device=%s", target_device)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=policy_path, config=policy_cfg)
    policy.to(target_device)
    policy.eval()

    if stats_path is None:
        logger.info("Loading pre/post processors from checkpoint: %s", policy_path)
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=policy_path)
    else:
        logger.info("Loading normalization stats from %s", stats_path)
        stats = _load_stats(stats_path)
        preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=stats)
    logger.info("Policy bundle ready.")
    return policy_cfg, policy, preprocessor, postprocessor, target_device


def _build_dataset_features(robot: ARX5Follower) -> dict[str, dict[str, Any]]:
    return combine_feature_dicts(
        hw_to_dataset_features(robot.observation_features, OBS_STR),
        hw_to_dataset_features(robot.action_features, ACTION),
    )


def _predict_action_chunk(
    *,
    robot_observation: dict[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    execution_horizon: int,
    use_amp: bool,
) -> list[dict[str, float]]:
    observation_frame = build_dataset_frame(dataset_features, robot_observation, prefix=OBS_STR)
    processed_observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task=task,
        robot_type=robot_type,
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        processed_observation = preprocessor(processed_observation)
        action_chunk = policy.predict_action_chunk(processed_observation)
        action_chunk = postprocessor(action_chunk)

    action_names = dataset_features[ACTION]["names"]
    action_chunk = action_chunk.squeeze(0).to("cpu")
    horizon = min(execution_horizon, action_chunk.shape[0])
    actions = []
    for index in range(horizon):
        row = action_chunk[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions


def _clip_safe_actions(
    actions: list[dict[str, float]],
    current_pose: list[float],
    max_joint_step: float,
) -> list[dict[str, float]]:
    previous_joint = np.asarray(current_pose[:6], dtype=np.float64)
    safe_actions = []
    for action in actions:
        safe_action = dict(action)
        target_joint = np.asarray([safe_action[key] for key in ARX5_REAL_STATE_KEYS[:6]], dtype=np.float64)
        delta = target_joint - previous_joint
        clipped_delta = np.clip(delta, -max_joint_step, max_joint_step)
        if not np.allclose(delta, clipped_delta):
            target_joint = previous_joint + clipped_delta
            for axis, key in enumerate(ARX5_REAL_STATE_KEYS[:6]):
                safe_action[key] = float(target_joint[axis])
            logger.warning(
                "SAFE MODE: capped joint step, max per-joint delta=%.4f.",
                max_joint_step,
            )
        previous_joint = target_joint
        safe_actions.append(safe_action)
    return safe_actions


def _save_chunk_io(
    *,
    record_dir: Path,
    round_index: int,
    observation: dict[str, Any],
    current_joint_state: list[float],
    actions: list[dict[str, float]],
    camera_names: list[str],
) -> Path:
    round_dir = record_dir / f"round_{round_index:04d}"
    input_dir = round_dir / "input"
    output_dir = round_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_name in camera_names:
        image = observation[camera_name]
        Image.fromarray(image).save(input_dir / f"{camera_name}.png")

    (input_dir / "current_joint_state.json").write_text(
        json.dumps({"current_joint_state": current_joint_state}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
    return round_dir


def _discover_next_round_index(record_dir: Path) -> int:
    """
    Discover the next available `round_XXXX` index inside `record_dir`.

    This allows continuing recording into an existing `--record-dir` without overwriting
    previous rounds.
    """
    if not record_dir.exists():
        return 0
    round_dirs = [p for p in record_dir.glob("round_*") if p.is_dir()]
    indices: list[int] = []
    for p in round_dirs:
        try:
            indices.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return (max(indices) + 1) if indices else 0


class TrainRawEpisodeRecorder:
    """
    Record robot data in a lightweight "raw" format for later conversion to LeRobotDataset.

    Recording is controlled by keyboard keys:
      - Start a recording window in the runtime script, then resume policy separately.
      - End recording with 'D' and finalize with '0'/'1'/'2' (failure/success/abandon).

    Raw format layout (inside raw_record_root):
      episode_{episode_idx:04d}/
        meta.json
        segments/
          segment_{seg_idx:04d}_{source}/
            images/
              {camera_name}/
                frame_{t:06d}.png            # one frame per recorded step
            actions.json                       # shape: [T, action_dim]
            states.json                        # shape: [T, action_dim]
    """

    def __init__(
        self,
        *,
        raw_record_root: Path,
        action_keys: tuple[str, ...],
        camera_names: list[str],
        task: str,
        dt_s: float,
        image_save_size_hw: tuple[int, int] | None = None,
        collector_policy_id_policy: str = "policy",
        collector_policy_id_human: str = "human",
        force_success_on_vr: bool = True,
    ) -> None:
        self.raw_record_root = raw_record_root
        self.action_keys = action_keys
        self.camera_names = camera_names
        self.task = task
        self.dt_s = float(dt_s)
        self.fps = int(round(1.0 / self.dt_s)) if self.dt_s > 0 else 1
        self.image_save_size_hw = image_save_size_hw
        self.collector_policy_id_policy = collector_policy_id_policy
        self.collector_policy_id_human = collector_policy_id_human
        self.force_success_on_vr = force_success_on_vr

        self.enabled = raw_record_root is not None
        self._episode_dir: Path | None = None
        self._segments_dir: Path | None = None
        self._episode_index = self._discover_next_episode_index()

        self.recording_active = False
        self.waiting_for_success_label = False
        self.success_forced = False
        self.episode_success: int | None = None

        self._seg_idx = 0
        self._active_segment_dir: Path | None = None
        self._active_segment_source: str | None = None
        self._step_idx = 0
        self._active_actions: list[list[float]] = []
        self._active_states: list[list[float]] = []
        self._segment_lock = threading.RLock()
        self._writer_queue: queue.Queue[tuple[Path, np.ndarray] | None] | None = None
        self._writer_thread: threading.Thread | None = None

    def _ensure_writer_started(self) -> None:
        if not self.enabled:
            return
        if self._writer_queue is None:
            self._writer_queue = queue.Queue(maxsize=2048)
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._writer_thread = threading.Thread(
                target=self._image_writer_worker,
                name="train-raw-image-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def _image_writer_worker(self) -> None:
        assert self._writer_queue is not None
        while True:
            item = self._writer_queue.get()
            try:
                if item is None:
                    return
                img_path, img = item
                Image.fromarray(img).save(img_path)
            finally:
                self._writer_queue.task_done()

    def _flush_image_queue(self) -> None:
        if self._writer_queue is not None:
            self._writer_queue.join()

    def _shutdown_writer(self) -> None:
        if self._writer_queue is None:
            return
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_queue.put(None)
            self._writer_queue.join()
            self._writer_thread.join(timeout=2.0)
        self._writer_thread = None
        self._writer_queue = None

    def _discover_next_episode_index(self) -> int:
        """
        Find the next episode index by scanning existing `episode_*/` folders.

        Important: do not depend on `meta.json` existence. If a previous run crashed
        mid-episode, the directory may exist without meta.json, and we must not
        reuse the same episode index.
        """
        if not self.raw_record_root.exists():
            return 0
        episode_dirs = [p for p in self.raw_record_root.glob("episode_*") if p.is_dir()]
        indices: list[int] = []
        for p in episode_dirs:
            try:
                # episode_0000 -> 0000
                idx = int(p.name.split("_", 1)[1])
            except Exception:
                continue
            indices.append(idx)
        return (max(indices) + 1) if indices else 0

    def can_start_new_episode(self) -> bool:
        return self.enabled and not self.recording_active and not self.waiting_for_success_label

    def start_episode(self) -> None:
        if not self.can_start_new_episode():
            return
        self._ensure_writer_started()
        self.raw_record_root.mkdir(parents=True, exist_ok=True)
        # In case we detected a stale episode index (e.g. crashed run left a folder behind),
        # keep searching until we find a directory that doesn't exist yet.
        while True:
            self._episode_dir = self.raw_record_root / f"episode_{self._episode_index:04d}"
            try:
                self._episode_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                self._episode_index += 1
        self._segments_dir = self._episode_dir / "segments"
        self._segments_dir.mkdir(parents=True, exist_ok=False)

        self._seg_idx = 0
        self._active_segment_dir = None
        self._active_segment_source = None
        self._active_actions = []
        self._active_states = []

        self.recording_active = True
        self.waiting_for_success_label = False
        self.success_forced = False
        self.episode_success = None

        # Images/shapes are stored per segment; meta tracks only schema-level info.
        meta = {
            "task": self.task,
            "dt_s": self.dt_s,
            "fps": self.fps,
            "action_keys": list(self.action_keys),
            "camera_names": self.camera_names,
            "collector_policy_id_policy": self.collector_policy_id_policy,
            "collector_policy_id_human": self.collector_policy_id_human,
            "success_forced": False,
            "episode_success": None,
            "has_vr_takeover": False,
        }
        (self._episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _require_episode_dir(self) -> Path:
        if self._episode_dir is None or self._segments_dir is None:
            raise RuntimeError("TrainRawEpisodeRecorder is not in an episode.")
        return self._episode_dir

    def mark_vr_takeover(self) -> None:
        if not self.recording_active:
            return
        self.success_forced = self.success_forced or self.force_success_on_vr
        meta_path = self._require_episode_dir() / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["success_forced"] = self.success_forced
        meta["has_vr_takeover"] = True
        # Success can still be finalized on 'D' (or forced immediately if desired).
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _start_segment(self, *, source: str) -> None:
        with self._segment_lock:
            if not self.recording_active:
                return
            episode_dir = self._require_episode_dir()
            segments_dir = self._segments_dir
            assert segments_dir is not None

            self._active_segment_dir = segments_dir / f"segment_{self._seg_idx:04d}_{source}"
            self._active_segment_dir.mkdir(parents=True, exist_ok=False)
            self._active_segment_source = source
            self._active_actions = []
            self._active_states = []
            self._seg_idx += 1
            self._step_idx = 0

            images_dir = self._active_segment_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=False)
            for camera_name in self.camera_names:
                (images_dir / camera_name).mkdir(parents=True, exist_ok=False)

            seg_meta = {
                "source": source,
                "dt_s": self.dt_s,
            }
            (self._active_segment_dir / "meta.json").write_text(
                json.dumps(seg_meta, indent=2) + "\n", encoding="utf-8"
            )

    def start_policy_segment(self) -> None:
        self._start_segment(source="policy")

    def start_vr_segment(self) -> None:
        self._start_segment(source="vr")

    def is_segment_active(self) -> bool:
        with self._segment_lock:
            return self._active_segment_dir is not None

    def _append_step(
        self,
        *,
        action_vector: list[float],
        state_vector: list[float],
        images: dict[str, np.ndarray],
    ) -> None:
        with self._segment_lock:
            if not self.recording_active or self._active_segment_dir is None:
                return
            if len(action_vector) != len(self.action_keys):
                raise ValueError("action_vector length mismatch")
            if len(state_vector) != len(self.action_keys):
                raise ValueError("state_vector length mismatch")
            for cam in self.camera_names:
                if cam not in images:
                    raise ValueError(f"Missing image for camera '{cam}' in recorder.record_*_step().")
            self._active_actions.append([float(x) for x in action_vector])
            self._active_states.append([float(x) for x in state_vector])

            images_dir = self._active_segment_dir / "images"
            for cam in self.camera_names:
                img = np.asarray(images[cam])
                if img.dtype != np.uint8:
                    img_f = img.astype(np.float32, copy=False)
                    if img_f.max() <= 1.0:
                        img_f = img_f * 255.0
                    img = np.clip(img_f, 0.0, 255.0).astype(np.uint8)
                if self.image_save_size_hw is not None:
                    target_h, target_w = self.image_save_size_hw
                    pil_img = Image.fromarray(img).resize(
                        (int(target_w), int(target_h)),
                        Image.Resampling.BILINEAR,
                    )
                    img = np.asarray(pil_img, dtype=np.uint8)
                img_path = images_dir / cam / f"frame_{self._step_idx:06d}.png"
                if self._writer_queue is None:
                    Image.fromarray(img).save(img_path)
                else:
                    # Step thread only enqueues image jobs; writer thread handles disk IO.
                    self._writer_queue.put((img_path, img.copy()))
            self._step_idx += 1

    def record_policy_step(self, *, action_dict: dict[str, float], observation: dict[str, Any]) -> None:
        action_vector = [float(action_dict[k]) for k in self.action_keys]
        state_vector = [float(observation[k]) for k in self.action_keys]
        images = {cam: observation[cam] for cam in self.camera_names}
        self._append_step(action_vector=action_vector, state_vector=state_vector, images=images)

    def record_vr_step(
        self,
        *,
        action_vector: list[float],
        state_vector: list[float],
        images: dict[str, np.ndarray],
    ) -> None:
        self._append_step(action_vector=action_vector, state_vector=state_vector, images=images)

    def finish_active_segment(self) -> None:
        with self._segment_lock:
            if self._active_segment_dir is None:
                return
            active_segment_dir = self._active_segment_dir
            active_actions = list(self._active_actions)
            active_states = list(self._active_states)
            self._flush_image_queue()
            (active_segment_dir / "actions.json").write_text(
                json.dumps(active_actions, indent=2) + "\n",
                encoding="utf-8",
            )
            (active_segment_dir / "states.json").write_text(
                json.dumps(active_states, indent=2) + "\n",
                encoding="utf-8",
            )
            self._active_segment_dir = None
            self._active_segment_source = None
            self._active_actions = []
            self._active_states = []

    def stop_episode(self, *, success_value: int) -> None:
        if not self.recording_active and not self.waiting_for_success_label:
            return
        if success_value not in (0, 1):
            raise ValueError("success_value must be 0/1")

        # Flush any active segment first.
        self.finish_active_segment()

        meta_path = self._require_episode_dir() / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["episode_success"] = int(success_value)
        meta["success_forced"] = bool(self.success_forced)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        self.recording_active = False
        self.waiting_for_success_label = False
        self.success_forced = False
        self.episode_success = int(success_value)
        self._episode_index += 1
        if not self.waiting_for_success_label:
            self._shutdown_writer()

    def abandon_episode(self) -> None:
        if not self.recording_active and not self.waiting_for_success_label:
            return

        self.finish_active_segment()

        episode_dir = self._require_episode_dir()
        shutil.rmtree(episode_dir)

        self.recording_active = False
        self.waiting_for_success_label = False
        self.success_forced = False
        self.episode_success = None

        self._episode_dir = None
        self._segments_dir = None
        self._seg_idx = 0
        self._active_segment_dir = None
        self._active_segment_source = None
        self._step_idx = 0
        self._active_actions = []
        self._active_states = []
        self._shutdown_writer()

    def request_finish_episode(self) -> None:
        """Called on 'D'. Wait for the user to press '0'/'1'/'2' to finalize the episode."""
        if not self.recording_active:
            return

        self.waiting_for_success_label = True
        self.recording_active = False
        if self.success_forced:
            logger.info(
                "Raw recording ended on 'D': VR takeover detected. "
                "Press '1' (success) to keep it, or '2' (abandon) to delete it. "
                "Pressing '0' (failure) is not allowed after VR takeover."
            )
            return
        logger.info("Raw recording ended on 'D': waiting for success label. Press '0' (failure), '1' (success), or '2' (abandon).")

    def handle_success_label_hotkey(self, key: str) -> bool:
        """Returns True if handled (and episode was finalized)."""
        if not self.waiting_for_success_label:
            return False
        if key not in {"0", "1", "2"}:
            return False

        if key == "2":
            self.abandon_episode()
            logger.info("Raw episode abandoned and deleted.")
            return True

        if key == "0" and self.success_forced:
            logger.warning(
                "This episode had VR takeover, so it cannot be labeled as failure. "
                "Press '1' to keep it as success, or '2' to abandon and delete it."
            )
            return True

        success_value = int(key)
        self.stop_episode(success_value=success_value)
        logger.info("Raw episode finalized: episode_success=%d.", success_value)
        return True

def _log_keyboard_help(safe_mode: bool) -> None:
    base_message = (
        "Keyboard: [Space] stop | [H] home | [B] teach | [N] record pose | [M] goto pose | "
        "[V] VR teleop | [R] resume | [Q] quit"
    )
    if safe_mode:
        base_message += " | [I] next chunk"
    logger.info(base_message)


def _ensure_vr2robot_import() -> None:
    """Make `xrobotoolkit_teleop` importable (pip install -e vr2robot or repo next to workspace)."""
    try:
        import xrobotoolkit_teleop  # noqa: F401
    except ImportError:
        pass
    else:
        return

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "vr2robot"
        if (candidate / "xrobotoolkit_teleop").is_dir():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            import xrobotoolkit_teleop  # noqa: F401
            return

    raise ImportError(
        "Cannot import xrobotoolkit_teleop. Install the vr2robot package "
        "(e.g. pip install -e /path/to/vr2robot) or place the vr2robot repo where it can be found "
        "next to an ancestor directory of this script."
    )


def _run_vr_teleop_session(
    *,
    robot: ARX5Follower,
    keyboard: KeyboardListener | None,
    args: argparse.Namespace,
    recorder: TrainRawEpisodeRecorder | None = None,
) -> None:
    """
    Hold current joints, release the infer stack's CAN client, then run ARXX5TeleopController (VR + Placo IK)
    without dataset logging.

    Matches vr2robot single-arm VR teleop (`teleop_arx_x5_hardware.py` / `record_lerobot_single_arx_x5.py` stack).
    """
    if args.use_stub:
        logger.error("VR teleop requires a real arm; omit --use-stub.")
        return

    _ensure_vr2robot_import()
    from xrobotoolkit_teleop.hardware.arx_x5_teleop_controller import (
        ARXX5TeleopController,
        DEFAULT_ARX_X5_MANIPULATOR_CONFIG,
        DEFAULT_ARX_X5_URDF_PATH,
    )

    if keyboard is not None:
        keyboard.pause()

    arx_runtime = Arx5Runtime.get_or_create(
        can_port=args.can_port,
        arm_type=args.arm_type,
        dt=1.0 / max(1, args.vr_control_hz),
        use_stub=args.use_stub,
        recorded_pose_path=robot.recorded_pose_path,
    )

    was_connected = robot.is_connected
    pre_switch_q: np.ndarray | None = None
    if was_connected:
        logger.info("Holding current joint pose before VR teleop (freeze infer targets).")
        robot.hold_position()
        time.sleep(0.1)
        pre_switch_q = np.asarray(robot.get_joint_vector(), dtype=np.float64)
        if recorder is not None and recorder.recording_active:
            logger.info("Recording VR segment.")
            recorder.start_vr_segment()
            recorder.mark_vr_takeover()
        logger.info(
            "Handing off the same ARX5 client to VR (no CAN reconnect; no protect_mode on arm)."
        )
        robot.disconnect(protect=False, release_arm_to=arx_runtime)

    class _VRHoldController(ARXX5TeleopController):
        """Same-process VR: no go_home; Placo + gripper match hardware until user uses VR."""

        def _robot_setup(self):
            self.arm_controllers = {}
            self._hold_joint_targets = {}
            self._initial_gripper_by_arm: dict[str, float] = {}
            self._vr_gripper_hold_until_trigger = True
            self._record_arm_name = next(iter(self.can_ports.keys())) if self.can_ports else None
            self._last_vr_images: dict[str, np.ndarray] = {}
            for arm_name, can_port in self.can_ports.items():
                logger.info(
                    "Binding ARX X5 %s to shared runtime (was CAN %s; no second SingleArm).",
                    arm_name,
                    can_port,
                )
                arm = SharedARXX5Interface(arx_runtime, dt=self.dt)
                self.arm_controllers[arm_name] = arm
                if pre_switch_q is not None and pre_switch_q.size >= 7:
                    current = pre_switch_q[:6].astype(np.float64, copy=True)
                    gname = self.manipulator_config[arm_name]["gripper_config"]["joint_names"][0]
                    self.gripper_pos_target[arm_name][gname] = float(pre_switch_q[6])
                    self._initial_gripper_by_arm[arm_name] = float(pre_switch_q[6])
                else:
                    current = np.asarray(arm.get_joint_positions()[:6], dtype=np.float64)
                    gname = self.manipulator_config[arm_name]["gripper_config"]["joint_names"][0]
                    self._initial_gripper_by_arm[arm_name] = float(
                        self.gripper_pos_target[arm_name][gname]
                    )
                self._hold_joint_targets[arm_name] = current.copy()
                arm.set_joint_positions(current)
                if pre_switch_q is not None and pre_switch_q.size >= 7:
                    arm.set_catch_pos(float(pre_switch_q[6]))
            time.sleep(0.05)

        def _placo_setup(self):
            super()._placo_setup()
            for arm_name, controller in self.arm_controllers.items():
                q_slice = self.placo_arm_joint_slice[arm_name]
                self.placo_robot.state.q[q_slice] = np.asarray(
                    controller.get_joint_positions()[:6],
                    dtype=np.float64,
                )
            self.placo_robot.update_kinematics()
            self.sync_end_effector_poses_to_placo_tasks()

        def _update_gripper_target(self):
            if self._vr_gripper_hold_until_trigger:
                for gripper_name in self.manipulator_config:
                    if "gripper_config" not in self.manipulator_config[gripper_name]:
                        continue
                    gc = self.manipulator_config[gripper_name]["gripper_config"]
                    if self.xr_client.get_key_value_by_name(gc["gripper_trigger"]) > 0.12:
                        self._vr_gripper_hold_until_trigger = False
                        break
                if self._vr_gripper_hold_until_trigger:
                    for gripper_name in self.manipulator_config:
                        if "gripper_config" not in self.manipulator_config[gripper_name]:
                            continue
                        gc = self.manipulator_config[gripper_name]["gripper_config"]
                        jn = gc["joint_names"][0]
                        self.gripper_pos_target[gripper_name][jn] = self._initial_gripper_by_arm[
                            gripper_name
                        ]
                    return
            super()._update_gripper_target()

        def _send_command(self):
            for arm_name, controller in self.arm_controllers.items():
                if self.active.get(arm_name, False):
                    q_cmd = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()
                    self._hold_joint_targets[arm_name] = q_cmd.copy()
                else:
                    q_cmd = self._hold_joint_targets.get(arm_name)
                    if q_cmd is None:
                        q_cmd = np.asarray(controller.get_joint_positions()[:6], dtype=np.float64)
                        self._hold_joint_targets[arm_name] = q_cmd.copy()
                controller.set_joint_positions(q_cmd)

                if "gripper_config" in self.manipulator_config[arm_name]:
                    gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                    joint_name = gripper_config["joint_names"][0]
                    gripper_target = self.gripper_pos_target[arm_name][joint_name]
                    controller.set_catch_pos(gripper_target)
                else:
                    gripper_target = float(0.0)

                # Record VR actions + full step images.
                if (
                    recorder is not None
                    and recorder.recording_active
                    and self._record_arm_name is not None
                    and arm_name == self._record_arm_name
                ):
                    images: dict[str, np.ndarray] = {}
                    if self.camera_interface is not None:
                        frames_by_serial = self.camera_interface.get_frames()
                        # Map vr2robot camera names -> Evo-RL local camera names.
                        teleop_to_local = {"base": "side", "left_wrist": "wrist", "right_wrist": "front"}
                        for serial, frame_data in frames_by_serial.items():
                            teleop_name = self.camera_serial_to_name.get(serial)
                            if teleop_name is None:
                                continue
                            local_name = teleop_to_local.get(teleop_name)
                            if local_name is None or local_name not in recorder.camera_names:
                                continue
                            color = frame_data.get("color", None)
                            if color is None:
                                continue
                            images[local_name] = color
                    # Cache last images so we can still record every control tick.
                    self._last_vr_images.update(images)
                    if all(cam in self._last_vr_images for cam in recorder.camera_names):
                        full_images = {cam: self._last_vr_images[cam] for cam in recorder.camera_names}
                        action_vector = (
                            q_cmd[:6].astype(np.float64, copy=False).tolist()
                            + [float(gripper_target)]
                        )
                        recorder.record_vr_step(
                            action_vector=action_vector,
                            state_vector=action_vector,
                            images=full_images,
                        )

        def _shutdown_robot(self):
            for arm_name, controller in self.arm_controllers.items():
                q = np.asarray(controller.get_joint_positions()[:6], dtype=np.float64)
                controller.set_joint_positions(q)
                if "gripper_config" in self.manipulator_config[arm_name]:
                    gc = self.manipulator_config[arm_name]["gripper_config"]
                    jn = gc["joint_names"][0]
                    controller.set_catch_pos(float(self.gripper_pos_target[arm_name][jn]))

    logger.info(
        "Starting VR teleop (optionally records raw training segments). Hold grip to move; trigger for gripper. "
        "Press Ctrl+C here to stop VR and return to infer."
    )
    if not arx_runtime.is_connected and not args.use_stub:
        logger.info("VR without infer handoff: connecting Arx5Runtime once.")
        arx_runtime.connect(recorded_pose_path=robot.recorded_pose_path)

    try:
        vr_cam = not args.vr_no_camera
        controller = _VRHoldController(
            robot_urdf_path=DEFAULT_ARX_X5_URDF_PATH,
            manipulator_config=DEFAULT_ARX_X5_MANIPULATOR_CONFIG,
            scale_factor=args.vr_scale_factor,
            enable_camera=vr_cam,
            enable_camera_display=vr_cam and not args.vr_no_camera_display,
            camera_width=args.cam_width,
            camera_height=args.cam_height,
            camera_fps=args.fps,
            enable_log_data=False,
            can_ports={"right_arm": args.can_port},
            control_rate_hz=args.vr_control_hz,
            visualize_placo=args.vr_visualize_placo,
        )
        controller.run()
    except KeyboardInterrupt:
        logger.info("VR teleop stopped (Ctrl+C).")
    except Exception:
        logger.exception("VR teleop failed.")
    finally:
        if recorder is not None:
            recorder.finish_active_segment()
        arm_back = arx_runtime.take_client()
        if arm_back is not None:
            logger.info("Returning arm client to infer and reconnecting cameras...")
            try:
                robot.connect(reuse_arm_client=arm_back)
                robot.hold_position()
            except Exception:
                logger.exception("Failed to reconnect ARX5 after VR teleop.")
        elif was_connected:
            try:
                robot.connect()
                robot.hold_position()
            except Exception:
                logger.exception("Failed to reconnect ARX5 after VR teleop.")
        if keyboard is not None:
            keyboard.resume()
        logger.info("Infer keyboard active again. Still STOPPED until you press [R] to run policy.")


def _run_keyboard_command(
    *,
    key: str | None,
    robot: ARX5Follower,
    state: LoopState,
    safe_mode: bool,
    request_next_chunk: bool,
    recorder: TrainRawEpisodeRecorder | None = None,
) -> tuple[LoopState, bool, bool, bool]:
    """Returns (state, request_next_chunk, running, vr_teleop_requested)."""
    running = True
    vr = False

    if recorder is not None and recorder.handle_success_label_hotkey(key or ""):
        # Episode finalized; keep the loop stopped until the user explicitly resumes policy.
        return LoopState.STOPPED, False, True, False

    if key == "d":
        if recorder is None:
            logger.warning(
                "[D] Raw training recording is not enabled (omit --raw-train-record-dir). "
                "Stopping policy like [Space]. Re-run with --raw-train-record-dir <dir> to use [D] to end episodes."
            )
            robot.hold_position()
            time.sleep(0.05)
            return LoopState.STOPPED, False, True, False
        if recorder.recording_active:
            logger.info("Stopping raw training recording (D).")
            robot.hold_position()
            time.sleep(0.05)
            recorder.request_finish_episode()
            return LoopState.STOPPED, False, True, False
        if recorder.waiting_for_success_label:
            logger.info("[D] Ignored: already waiting for success label; press [0], [1], or [2] to finalize.")
            return state, request_next_chunk, running, vr
        logger.info("[D] Ignored: raw recording is not active (press [R] to start an episode).")
        return state, request_next_chunk, running, vr

    if key == " ":
        robot.hold_position()
        logger.warning("Emergency stop: holding current pose.")
        return LoopState.STOPPED, False, True, False
    if key == "q":
        logger.info("Quit requested.")
        return state, request_next_chunk, False, False
    if state == LoopState.STOPPED:
        if key == "v":
            logger.info("VR teleop requested from STOPPED state.")
            if recorder is not None and recorder.recording_active:
                recorder.mark_vr_takeover()
            return state, request_next_chunk, running, True
        if key == "h":
            logger.info("Moving ARX5 to the home pose.")
            robot.hold_position()
            time.sleep(0.1)
            robot.go_home()
            time.sleep(2.0)
            robot.hold_position()
        elif key == "r":
            if recorder is not None and recorder.waiting_for_success_label:
                logger.info("Waiting for success label (press '0', '1', or '2'); ignoring [R] resume.")
                return LoopState.STOPPED, False, True, False
            if recorder is not None and recorder.can_start_new_episode():
                logger.info("Starting raw training recording and resuming policy (R).")
                recorder.start_episode()
            robot.hold_position()
            time.sleep(0.1)
            logger.info("Resumed policy control.")
            return LoopState.RUNNING, not safe_mode, running, False
        elif key == "b":
            robot.enter_teach_mode()
            logger.info("Teach mode enabled. Drag the arm, then press [N] to save the pose.")
            return LoopState.TEACHING, request_next_chunk, running, False
        elif key == "m":
            if robot.has_recorded_pose():
                robot.move_to_recorded()
                logger.info("Moved to the recorded pose.")
            else:
                logger.warning("No recorded pose found. Use [B] then [N] first.")
    elif state == LoopState.TEACHING:
        if key == "n":
            recorded_pose_path = robot.exit_teach_mode_and_record()
            logger.info("Recorded pose saved to %s", recorded_pose_path)
            return LoopState.STOPPED, False, running, False
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        logger.info("SAFE MODE: next chunk requested.")
        return state, True, running, False
    return state, request_next_chunk, running, vr


def _execute_chunk(
    *,
    robot: ARX5Follower,
    actions: list[dict[str, float]],
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
    recorder: TrainRawEpisodeRecorder | None = None,
) -> tuple[LoopState, bool, bool]:
    state = LoopState.RUNNING
    request_next_chunk = not safe_mode
    running = True
    for action in actions:
        step_start = time.perf_counter()
        if keyboard is not None:
            key = keyboard.get_key()
            state, request_next_chunk, running, _vr = _run_keyboard_command(
                key=key,
                robot=robot,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
                recorder=recorder,
            )
            if state != LoopState.RUNNING or not running:
                break
        observation_for_step: dict[str, Any] | None = None
        if recorder is not None and recorder.recording_active:
            observation_for_step = robot.get_observation()
        robot.send_action(action)
        if recorder is not None and recorder.recording_active and observation_for_step is not None:
            recorder.record_policy_step(action_dict=action, observation=observation_for_step)
        precise_sleep(max(step_duration_s - (time.perf_counter() - step_start), 0.0))

    if recorder is not None:
        recorder.finish_active_segment()
    return state, request_next_chunk, running


def _log_predicted_actions(actions: list[dict[str, float]]) -> None:
    for action_index, action in enumerate(actions):
        logger.debug("Action[%d]: %s", action_index, action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot pi05_base directly on an ARX5 arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=None, help="Natural language task instruction.")
    parser.add_argument("--policy-path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help=(
            "Optional normalization stats path. If omitted, load policy_preprocessor/postprocessor "
            "directly from --policy-path."
        ),
    )
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step.")

    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:254522071216", "wrist:150622073629", "front:409122272986"],
        help="Local camera specs as name:serial_or_index using the dexbotic names side/wrist/front.",
    )
    parser.add_argument("--use-usb-cams", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-cameras", nargs="*", default=[])
    parser.add_argument("--list-cameras", action="store_true")

    parser.add_argument("--safe-mode", action="store_true", default=False)
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument(
        "--vr-scale-factor",
        type=float,
        default=1.5,
        help="VR teleop motion scale (same role as teleop_arx_x5_hardware.py).",
    )
    parser.add_argument(
        "--vr-no-camera",
        action="store_true",
        help="VR teleop without RealSense threads (arm-only IK control).",
    )
    parser.add_argument(
        "--vr-no-camera-display",
        action="store_true",
        help="In VR teleop, do not open OpenCV camera preview windows.",
    )
    parser.add_argument("--vr-control-hz", type=int, default=50, help="VR teleop IK / control loop rate.")
    parser.add_argument(
        "--vr-visualize-placo",
        action="store_true",
        help="Open Placo / MeshCat visualization during VR teleop.",
    )
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument(
        "--raw-train-record-dir",
        type=Path,
        default=None,
        help="If set, enable raw training recording controlled by keyboard: R starts (and resumes policy), D ends, 0/1/2 finalize failure/success/abandon.",
    )
    parser.add_argument("--robot-id", type=str, default="arx5")
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")

    args = parser.parse_args()

    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("--task is required unless --list-cameras is used.")
    if args.safe_mode and args.no_keyboard:
        parser.error("SAFE MODE requires keyboard control.")
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive.")

    logger.info("Loading policy config from %s", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    required_policy_camera_keys = _get_required_policy_camera_keys(policy_cfg)
    local_camera_map = _resolve_local_camera_map(policy_cfg)
    camera_specs = _parse_camera_specs(args.cameras)
    camera_configs = _make_camera_configs(
        camera_specs=camera_specs,
        local_camera_map=local_camera_map,
        required_policy_camera_keys=required_policy_camera_keys,
        use_usb_cams=args.use_usb_cams,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
        flipped_local_cameras=set(args.flip_cameras),
    )

    robot = ARX5Follower(
        ARX5FollowerConfig(
            id=args.robot_id,
            port=args.can_port,
            arm_type=args.arm_type,
            use_stub=args.use_stub,
            cameras=camera_configs,
            protect_on_disconnect=args.protect_on_disconnect,
        )
    )
    policy_cfg, policy, preprocessor, postprocessor, device = _load_policy_bundle(
        policy_path=args.policy_path,
        device_override=args.policy_device,
        stats_path=args.stats_path,
        policy_cfg=policy_cfg,
    )
    dataset_features = _build_dataset_features(robot)
    camera_names = list(robot.cameras)
    raw_recorder: TrainRawEpisodeRecorder | None = None
    if args.raw_train_record_dir is not None:
        raw_recorder = TrainRawEpisodeRecorder(
            raw_record_root=args.raw_train_record_dir,
            action_keys=ARX5_REAL_STATE_KEYS,
            camera_names=camera_names,
            task=args.task,
            dt_s=args.duration,
        )
    execution_horizon = args.execution_horizon or int(policy_cfg.n_action_steps)
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    round_index = _discover_next_round_index(args.record_dir) if args.record_dir is not None else 0
    running = True

    try:
        logger.info("Connecting ARX5 runtime.")
        logger.info("Camera sources: %s", {name: str(source) for name, source in camera_specs.items()})
        logger.info("Resolved local camera map: %s", local_camera_map)
        logger.info("Stub arm: %s", args.use_stub)
        robot.connect()
        logger.info("ARX5 runtime connected. Keyboard enabled: %s", keyboard is not None)
        if keyboard is not None:
            logger.info("Moving ARX5 to the home pose.")
            robot.go_home()
            time.sleep(2.0)
            robot.hold_position()
            _log_keyboard_help(args.safe_mode)
            if raw_recorder is not None:
                logger.info(
                    "Raw training recording: press [R] to start/resume policy and record, press [D] to end recording. "
                    "After [D], press [0] (failure), [1] (success), or [2] (abandon). "
                    "If VR takeover happened, [0] is rejected and you must choose [1] or [2]."
                )
            if args.safe_mode:
                logger.info("SAFE MODE is armed. Press [R] to resume, then [I] for each chunk.")
        else:
            logger.info("Starting continuous ARX5 inference loop without keyboard control.")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while running:
            if keyboard is not None:
                previous_state = state
                key = keyboard.get_key()
                state, request_next_chunk, running, vr_req = _run_keyboard_command(
                    key=key,
                    robot=robot,
                    state=state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                    recorder=raw_recorder,
                )
                if vr_req:
                    try:
                        _run_vr_teleop_session(
                            robot=robot, keyboard=keyboard, args=args, recorder=raw_recorder
                        )
                    except ImportError as err:
                        logger.error("VR teleop unavailable: %s", err)
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    state = LoopState.STOPPED
                    request_next_chunk = False if args.safe_mode else True
                    continue
                if previous_state != LoopState.RUNNING and state == LoopState.RUNNING:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
            if not running:
                break
            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if args.safe_mode and not request_next_chunk:
                time.sleep(0.05)
                continue

            observation = robot.get_observation()
            current_pose = [observation[key] for key in ARX5_REAL_STATE_KEYS]
            actions = _predict_action_chunk(
                robot_observation=observation,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type=robot.robot_type,
                execution_horizon=execution_horizon,
                use_amp=policy_cfg.use_amp,
            )
            if args.safe_mode:
                actions = _clip_safe_actions(actions, current_pose=current_pose, max_joint_step=args.max_joint_step)
            if not actions:
                logger.warning("Policy returned no actions. Retrying.")
                time.sleep(0.1)
                continue

            grippers = [round(action[ARX5_REAL_STATE_KEYS[-1]], 4) for action in actions]
            logger.info("Predicted %d actions. Gripper values: %s", len(actions), grippers)
            _log_predicted_actions(actions)
            request_next_chunk = False

            if args.record_dir is not None:
                round_dir = _save_chunk_io(
                    record_dir=args.record_dir,
                    round_index=round_index,
                    observation=observation,
                    current_joint_state=robot.get_joint_vector(),
                    actions=actions,
                    camera_names=camera_names,
                )
                logger.info("Saved chunk %d to %s", round_index, round_dir)
                round_index += 1
                if args.safe_mode:
                    logger.info("SAFE MODE: press [I] for the next chunk.")
                if raw_recorder is None:
                    continue

            if raw_recorder is not None and raw_recorder.recording_active:
                raw_recorder.start_policy_segment()

            state, request_next_chunk, running = _execute_chunk(
                robot=robot,
                actions=actions,
                step_duration_s=args.duration,
                keyboard=keyboard,
                safe_mode=args.safe_mode,
                recorder=raw_recorder,
            )
            if args.safe_mode and state == LoopState.RUNNING:
                logger.info("SAFE MODE: press [I] for the next chunk.")
    finally:
        if raw_recorder is not None:
            # Best-effort flush; episode success label may still be missing if you exit before pressing [D].
            try:
                raw_recorder.finish_active_segment()
            except Exception:
                logger.exception("Failed to flush raw recorder segment on exit.")
            if raw_recorder.recording_active:
                logger.warning("Raw recording was still active on exit. Press [D] to finalize an episode.")
            if raw_recorder.waiting_for_success_label:
                logger.warning("Waiting for success label (press [0], [1], or [2]) but exited.")
        if keyboard is not None:
            keyboard.restore()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
