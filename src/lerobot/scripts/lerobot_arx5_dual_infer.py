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

"""Run dual-arm LeRobot Pi0.5 inference directly on two ARX5 arms."""

import argparse
import dataclasses
import datetime
import json
import logging
import sys
import threading
import time
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient
from lerobot.robots.arx5_follower.config_arx5_follower import ARX5FollowerConfigBase
from lerobot.robots.arx5_follower.arx5_runtime import Arx5Runtime, SharedARXX5Interface
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.scripts.lerobot_arx5_infer import (
    KeyboardListener,
    LoopState,
    TrainRawEpisodeRecorder,
    _discover_next_round_index,
    _ensure_vr2robot_import,
    _list_realsense_cameras,
    _load_policy_bundle,
    _log_predicted_actions,
    _predict_action_chunk,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "lerobot/pi05_bimanual"
DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"
DEFAULT_STATS_PATH = files("lerobot.robots.arx5_follower").joinpath("pi05_arx5_default_stats.json")
# Hardware / dataset convention: gripper.pos is clipped to [gripper_min, gripper_max]; the upper bound is fully open.
ARX5_GRIPPER_FULLY_OPEN = float(
    next(f.default for f in dataclasses.fields(ARX5FollowerConfigBase) if f.name == "gripper_max")
)
DUAL_STATE_DIM = 14
STATE_KEYS = (
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
)
# TODO: action
LEFT_STATE_KEYS = STATE_KEYS[7:]
RIGHT_STATE_KEYS = STATE_KEYS[:7]
# LEFT_STATE_KEYS = STATE_KEYS[7:]
# RIGHT_STATE_KEYS = STATE_KEYS[:7]
ARM_ORDER = ("left_arm", "right_arm")


class EndEffectorTrajectoryRecorder:
    """Record dual-arm end-effector xyz trajectory and save to JSON."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._samples: list[dict[str, Any]] = []
        self._enabled = False

    @staticmethod
    def _read_arm_xyz(arm: ARX5ArmClient) -> list[float]:
        # Prefer SDK end-effector pose API; fallback to NaN when unavailable.
        getter = getattr(arm.arm, "get_ee_pose_xyzrpy", None)
        if getter is None:
            return [float("nan"), float("nan"), float("nan")]
        try:
            pose = getter()
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose_array.size < 3:
                return [float("nan"), float("nan"), float("nan")]
            return [float(pose_array[0]), float(pose_array[1]), float(pose_array[2])]
        except Exception:
            return [float("nan"), float("nan"), float("nan")]

    def record(self, *, step_index: int, left_arm: ARX5ArmClient, right_arm: ARX5ArmClient) -> None:
        if not self._enabled:
            return
        sample = {
            "step_index": int(step_index),
            "t_rel_s": float(time.perf_counter() - self._start_time),
            "left_xyz": self._read_arm_xyz(left_arm),
            "right_xyz": self._read_arm_xyz(right_arm),
        }
        self._samples.append(sample)

    def start(self) -> None:
        self._enabled = True
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"end_effector_traj_{timestamp}.json"
        payload = {
            "num_samples": len(self._samples),
            "samples": self._samples,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output_path


class JointTrajectoryRecorder:
    """Record commanded and measured dual-arm joint trajectories and save to JSON."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._samples: list[dict[str, Any]] = []
        self._enabled = False

    def start(self) -> None:
        self._enabled = True
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    def record(
        self,
        *,
        step_index: int,
        left_cmd: list[float],
        right_cmd: list[float],
        left_arm: ARX5ArmClient,
        right_arm: ARX5ArmClient,
    ) -> None:
        if not self._enabled:
            return
        left_actual = [float(value) for value in left_arm.get_state()[:7]]
        right_actual = [float(value) for value in right_arm.get_state()[:7]]
        sample = {
            "step_index": int(step_index),
            "t_rel_s": float(time.perf_counter() - self._start_time),
            "left_cmd": [float(value) for value in left_cmd[:7]],
            "right_cmd": [float(value) for value in right_cmd[:7]],
            "left_actual": left_actual,
            "right_actual": right_actual,
        }
        self._samples.append(sample)

    def save(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"joint_traj_{timestamp}.json"
        payload = {
            "num_samples": len(self._samples),
            "samples": self._samples,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output_path


def _is_any_vr_arm_active(active_by_arm: dict[str, bool]) -> bool:
    return any(bool(active_by_arm.get(arm_name, False)) for arm_name in ARM_ORDER)


def _visual_image_slot_names(policy_cfg: PreTrainedConfig) -> list[str]:
    """Names after `observation.images.` in policy input order."""
    if not policy_cfg.input_features:
        raise ValueError("Policy input_features is empty; cannot infer camera slot names.")

    prefix = f"{OBS_STR}.images."
    slots: list[str] = []
    for key, feat in policy_cfg.input_features.items():
        if not key.startswith(prefix) or feat.type is not FeatureType.VISUAL:
            continue
        slots.append(key.removeprefix(prefix))
    if not slots:
        raise ValueError("Policy has no VISUAL observation.images.* inputs.")
    return slots


def _required_image_slot_names(policy_cfg: PreTrainedConfig) -> list[str]:
    return [name for name in _visual_image_slot_names(policy_cfg) if not name.startswith("empty_camera_")]


def _parse_camera_specs(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid camera spec '{spec}', expected name:value")
        name, value = spec.split(":", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def _make_camera_configs(
    *,
    required_image_slot_names: list[str],
    camera_specs: dict[str, str],
    use_usb_cams: bool,
    width: int,
    height: int,
    fps: int,
    flipped_cameras: set[str],
) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for name in required_image_slot_names:
        if name not in camera_specs:
            raise ValueError(
                f"Missing camera spec for slot '{name}'. Required slots: {required_image_slot_names}. "
                f"Pass e.g. --cameras {name}:<serial> for each."
            )
        source = camera_specs[name]
        rotation = 180 if name in flipped_cameras else 0
        if use_usb_cams:
            configs[name] = OpenCVCameraConfig(
                index_or_path=int(source),
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
        else:
            configs[name] = RealSenseCameraConfig(
                serial_number_or_name=source,
                width=width,
                height=height,
                fps=fps,
                rotation=rotation,
            )
    return configs


def _build_dataset_features(camera_configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observation_features: dict[str, type | tuple[int, int, int]] = {key: float for key in STATE_KEYS}
    for name, camera_config in camera_configs.items():
        observation_features[name] = (camera_config.height, camera_config.width, 3)
    action_features = {key: float for key in STATE_KEYS}
    return combine_feature_dicts(
        hw_to_dataset_features(observation_features, OBS_STR),
        hw_to_dataset_features(action_features, ACTION),
    )


def _read_dual_state(left_arm: ARX5ArmClient, right_arm: ARX5ArmClient) -> list[float]:
    left_state = list(left_arm.get_state())
    right_state = list(right_arm.get_state())
    # TODO: state
    state = right_state[:7] + left_state[:7]
    #state = left_state[:7] + right_state[:7]
    if len(state) != DUAL_STATE_DIM:
        raise RuntimeError(f"Expected {DUAL_STATE_DIM}-dim state, got {len(state)}.")
    return state


def _build_dual_observation(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    cameras: dict[str, Any],
) -> dict[str, Any]:
    state = _read_dual_state(left_arm, right_arm)
    observation: dict[str, Any] = {key: float(state[index]) for index, key in enumerate(STATE_KEYS)}
    for name, camera in cameras.items():
        observation[name] = camera.async_read()
    return observation


def _serialize_dual_arm_targets(
    q_cmd_by_arm: dict[str, np.ndarray],
    gripper_by_arm: dict[str, float],
) -> list[float]:
    """Serialize dual-arm joint+gripper targets in dataset order: right, then left."""
    action_vector: list[float] = []
    for arm_name in ("right_arm", "left_arm"):
        q_cmd = np.asarray(q_cmd_by_arm[arm_name], dtype=np.float64).reshape(-1)
        action_vector.extend(q_cmd[:6].tolist())
        action_vector.append(float(gripper_by_arm[arm_name]))
    return action_vector


def _home_dual_arms_after_camera_failure(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    error: Exception,
) -> None:
    logger.error("相机读取失败，停止推理并将双臂移回 home。错误：%s", error)
    left_arm.hold_position()
    right_arm.hold_position()
    time.sleep(0.1)
    left_arm.go_home()
    right_arm.go_home()
    time.sleep(5.0)
    left_arm.hold_position()
    right_arm.hold_position()


def _split_dual_action(action: dict[str, float]) -> tuple[list[float], list[float]]:
    left_joint = [float(action[key]) for key in LEFT_STATE_KEYS]
    right_joint = [float(action[key]) for key in RIGHT_STATE_KEYS]
    return left_joint, right_joint


def _clip_safe_actions(
    actions: list[dict[str, float]],
    current_state: list[float],
    max_joint_step: float,
) -> list[dict[str, float]]:
    previous_state = np.asarray(current_state, dtype=np.float64)
    joint_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    safe_actions: list[dict[str, float]] = []
    for action in actions:
        safe_action = dict(action)
        target_state = np.asarray([safe_action[key] for key in STATE_KEYS], dtype=np.float64)
        delta = target_state[joint_indices] - previous_state[joint_indices]
        clipped_delta = np.clip(delta, -max_joint_step, max_joint_step)
        if not np.allclose(delta, clipped_delta):
            target_state[joint_indices] = previous_state[joint_indices] + clipped_delta
            for index, key in enumerate(STATE_KEYS):
                safe_action[key] = float(target_state[index])
            logger.warning(
                "**********安全模式：已限制双臂关节步长，单关节最大变化=%.4f。**********",
                max_joint_step,
            )
        previous_state = target_state
        safe_actions.append(safe_action)
    return safe_actions


def _save_chunk_io(
    *,
    record_dir: Path,
    round_index: int,
    observation: dict[str, Any],
    state: list[float],
    actions: list[dict[str, float]],
    camera_names: list[str],
) -> Path:
    round_dir = record_dir / f"round_{round_index:04d}"
    input_dir = round_dir / "input"
    output_dir = round_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_name in camera_names:
        Image.fromarray(observation[camera_name]).save(input_dir / f"{camera_name}.png")

    (input_dir / "state.json").write_text(
        json.dumps({"state": state}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
    return round_dir


def _log_keyboard_help(safe_mode: bool) -> None:
    base_message = (
        "键盘：[Space] 急停| [H] 回零 | [M] 前往记录位姿 | [V] VR 遥操作（VR 内按 [X] 退出） | "
        "[S] 开始录制 | [R] 执行推理 | [D] 结束录制 | [Q] 退出 | "
        "[O] open grippers (when stopped) | [B] teach | [N] record pose"
    )
    if safe_mode:
        base_message += " | [I] 下一段 chunk"
    logger.info(base_message)


def _run_keyboard_command(
    *,
    key: str | None,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    state: LoopState,
    safe_mode: bool,
    request_next_chunk: bool,
    recorder: TrainRawEpisodeRecorder | None = None,
    end_traj_recorder: EndEffectorTrajectoryRecorder | None = None,
    joint_traj_recorder: JointTrajectoryRecorder | None = None,
) -> tuple[LoopState, bool, bool, bool]:
    """Returns (state, request_next_chunk, running, vr_teleop_requested)."""
    if recorder is not None and recorder.handle_success_label_hotkey(key or ""):
        return LoopState.STOPPED, False, True, False

    def _warn_raw_recording_not_active(*, action_name: str, key_name: str) -> None:
        if recorder is None or recorder.recording_active or recorder.waiting_for_success_label:
            return
        logger.warning(
            "**********当前未在原始录制中仍触发了 %s（%s）。若要保存本回合，请先按 [S] 开始录制窗口。**********",
            action_name,
            key_name,
        )

    if key == "s":
        if recorder is None:
            logger.warning("**********[S] 未指定 --raw-train-record-dir，无法开始原始训练录制。**********")
            return state, request_next_chunk, True, False
        if recorder.waiting_for_success_label:
            logger.warning("**********[S] 已忽略：正在等待成功/失败标签，请先按 [0]、[1] 或 [2]。**********")
            return state, request_next_chunk, True, False
        if state != LoopState.STOPPED:
            logger.warning("**********[S] 已忽略：请先停止策略，再按 [S] 开始新的录制窗口。**********")
            return state, request_next_chunk, True, False
        if recorder.recording_active:
            logger.warning("**********[S] 已忽略：原始录制已在进行中。请按 [D] 结束当前录制。**********")
            return state, request_next_chunk, True, False
        if recorder.can_start_new_episode():
            recorder.start_episode()
            logger.info("已开始原始训练录制窗口（S）。请按 [R] 开始，本回合结束时按 [D]。")
        return state, request_next_chunk, True, False

    if key == "d":
        if recorder is None:
            logger.warning("**********[D] 未启用原始训练录制。请使用 --raw-train-record-dir <目录> 重新运行以使用 [S]/[D]。**********")
            return state, request_next_chunk, True, False
        if recorder.recording_active:
            logger.info("正在停止原始训练录制（D）。")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.05)
            recorder.request_finish_episode()
            return LoopState.STOPPED, False, True, False
        if recorder.waiting_for_success_label:
            logger.warning("**********[D] 已忽略：已在等待成功/失败标签；请按 [0]、[1] 或 [2] 完成标注。**********")
            return state, request_next_chunk, True, False
        logger.warning("**********[D] 已忽略：当前未在原始录制中。请先按 [S] 开始录制窗口。**********")
        return state, request_next_chunk, True, False

    if key == " ":
        left_arm.hold_position()
        right_arm.hold_position()
        if end_traj_recorder is not None and end_traj_recorder.enabled:
            end_traj_recorder.stop()
            logger.info("已停止末端轨迹记录（Space）。")
        if joint_traj_recorder is not None and joint_traj_recorder.enabled:
            joint_traj_recorder.stop()
            logger.info("已停止关节轨迹记录（Space）。")
        logger.info("急停：已保持当前姿态。")
        return LoopState.STOPPED, False, True, False
    if key == "q":
        logger.info("用户请求退出。")
        return state, request_next_chunk, False, False

    if key == "o":
        if state != LoopState.STOPPED:
            logger.warning(
                "**********[O] 已忽略：仅在停止状态下可张开夹爪；请先按 [Space] 急停。**********"
            )
            return state, request_next_chunk, True, False
        # send_joint -> set_joint_positions(6) + set_catch_pos(gripper). Fully open == gripper_max (not gripper_min).
        for arm in (left_arm, right_arm):
            pose = arm.get_state()
            arm.send_joint([float(pose[i]) for i in range(6)] + [ARX5_GRIPPER_FULLY_OPEN])
        logger.info("已张开双臂夹爪（O）。")
        return state, request_next_chunk, True, False

    if state == LoopState.STOPPED:
        if key == "v":
            _warn_raw_recording_not_active(action_name="请求 VR 遥操作", key_name="[V]")
            logger.info("在停止状态下请求进入 VR 遥操作。")
            return state, request_next_chunk, True, True
        if key == "h":
            logger.info("正在将双臂 ARX5 移动至 home 位姿。")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
        elif key == "r":
            if recorder is not None and recorder.waiting_for_success_label:
                logger.info("正在等待成功/失败标签（请按 '0'、'1' 或 '2'）；已忽略 [R] 恢复。")
                return LoopState.STOPPED, False, True, False
            _warn_raw_recording_not_active(action_name="恢复策略运行", key_name="[R]")
            if end_traj_recorder is not None and not end_traj_recorder.enabled:
                end_traj_recorder.start()
                logger.info("已开始末端轨迹记录（R）。")
            if joint_traj_recorder is not None and not joint_traj_recorder.enabled:
                joint_traj_recorder.start()
                logger.info("已开始关节轨迹记录（R）。")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            if recorder is not None and recorder.recording_active:
                logger.info("已开始推理（录制进行中）。")
            else:
                logger.info("已开始推理。")
            return LoopState.RUNNING, not safe_mode, True, False
        elif key == "b":
            left_arm.enter_teach_mode()
            right_arm.enter_teach_mode()
            logger.info("已进入示教模式。拖动双臂到位后按 [N] 保存位姿。")
            return LoopState.TEACHING, request_next_chunk, True, False
        elif key == "m":
            if left_arm.has_recorded_pose() and right_arm.has_recorded_pose():
                left_arm.move_to_recorded()
                right_arm.move_to_recorded()
                logger.info("双臂已移动至已记录位姿。")
            else:
                logger.warning("**********尚无已记录位姿。请先使用 [B] 再按 [N]。**********")
    elif state == LoopState.TEACHING:
        if key == "n":
            left_arm.save_recorded_pose()
            right_arm.save_recorded_pose()
            left_arm.hold_position()
            right_arm.hold_position()
            logger.info("已保存双臂位姿。")
            return LoopState.STOPPED, False, True, False
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        logger.info("安全模式：已请求下一段 chunk。")
        return state, True, True, False
    return state, request_next_chunk, True, False


def _execute_dual_chunk(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    cameras: dict[str, Any],
    actions: list[dict[str, float]],
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
    recorder: TrainRawEpisodeRecorder | None = None,
    end_traj_recorder: EndEffectorTrajectoryRecorder | None = None,
    joint_traj_recorder: JointTrajectoryRecorder | None = None,
) -> tuple[LoopState, bool, bool]:
    state = LoopState.RUNNING
    request_next_chunk = not safe_mode
    running = True
    for index, action in enumerate(actions):
        step_start = time.perf_counter()
        if keyboard is not None:
            key = keyboard.get_key()
            state, request_next_chunk, running, _vr = _run_keyboard_command(
                key=key,
                left_arm=left_arm,
                right_arm=right_arm,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
                recorder=recorder,
                end_traj_recorder=end_traj_recorder,
                joint_traj_recorder=joint_traj_recorder,
            )
            if state != LoopState.RUNNING or not running:
                break

        observation_for_step: dict[str, Any] | None = None
        if recorder is not None and recorder.recording_active:
            observation_for_step = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)

        left_joint, right_joint = _split_dual_action(action)
        left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
        right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()

        if end_traj_recorder is not None:
            end_traj_recorder.record(step_index=index, left_arm=left_arm, right_arm=right_arm)
        if joint_traj_recorder is not None:
            joint_traj_recorder.record(
                step_index=index,
                left_cmd=left_joint,
                right_cmd=right_joint,
                left_arm=left_arm,
                right_arm=right_arm,
            )

        if recorder is not None and recorder.recording_active and observation_for_step is not None:
            recorder.record_policy_step(action_dict=action, observation=observation_for_step)

        step_elapsed = time.perf_counter() - step_start
        if step_elapsed > step_duration_s:
            logger.warning(
                "控制步超时：step=%d elapsed=%.4fs > duration=%.4fs (overrun=%.4fs)",
                index,
                step_elapsed,
                step_duration_s,
                step_elapsed - step_duration_s,
            )
        precise_sleep(max(step_duration_s - step_elapsed, 0.0))

    if recorder is not None:
        recorder.finish_active_segment()
    return state, request_next_chunk, running


def _predict_action_chunk_with_kwargs(
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
    predict_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float]], torch.Tensor]:
    """Like `_predict_action_chunk`, but passes kwargs to `policy.predict_action_chunk` and returns raw tensor too."""
    predict_kwargs = predict_kwargs or {}
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
        action_chunk_raw = policy.predict_action_chunk(processed_observation, **predict_kwargs)
        action_chunk = postprocessor(action_chunk_raw)

    action_names = dataset_features[ACTION]["names"]
    action_chunk = action_chunk.squeeze(0).to("cpu")
    action_chunk_raw = action_chunk_raw.squeeze(0).detach().to("cpu")
    horizon = min(int(execution_horizon), int(action_chunk.shape[0]))
    actions: list[dict[str, float]] = []
    for index in range(horizon):
        row = action_chunk[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions, action_chunk_raw


class _AsyncChunkPredictor:
    """Background chunk predictor used to overlap inference and execution."""

    def __init__(
        self,
        *,
        dataset_features: dict[str, dict[str, Any]],
        policy,
        preprocessor,
        postprocessor,
        device,
        task: str,
        use_amp: bool,
        full_chunk_steps: int,
        rtc_enabled: bool,
    ) -> None:
        self._dataset_features = dataset_features
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._task = task
        self._use_amp = use_amp
        self._full_chunk_steps = full_chunk_steps
        self._rtc_enabled = bool(rtc_enabled)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._result: tuple[list[dict[str, float]], torch.Tensor | None] | None = None
        self._error: Exception | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        observation: dict[str, Any],
        rtc_prev_left_over: torch.Tensor | None = None,
        rtc_inference_delay: int | None = None,
        rtc_execution_horizon: int | None = None,
    ) -> bool:
        if self.is_running():
            return False

        def _worker() -> None:
            try:
                if self._rtc_enabled:
                    actions, raw = _predict_action_chunk_with_kwargs(
                        robot_observation=observation,
                        dataset_features=self._dataset_features,
                        policy=self._policy,
                        preprocessor=self._preprocessor,
                        postprocessor=self._postprocessor,
                        device=self._device,
                        task=self._task,
                        robot_type="arx5_dual",
                        execution_horizon=self._full_chunk_steps,
                        use_amp=self._use_amp,
                        predict_kwargs={
                            "prev_chunk_left_over": rtc_prev_left_over,
                            "inference_delay": rtc_inference_delay,
                            "execution_horizon": rtc_execution_horizon,
                        },
                    )
                    result = (actions, raw)
                else:
                    actions = _predict_action_chunk(
                        robot_observation=observation,
                        dataset_features=self._dataset_features,
                        policy=self._policy,
                        preprocessor=self._preprocessor,
                        postprocessor=self._postprocessor,
                        device=self._device,
                        task=self._task,
                        robot_type="arx5_dual",
                        execution_horizon=self._full_chunk_steps,
                        use_amp=self._use_amp,
                    )
                    result = (actions, None)
                with self._lock:
                    self._result = result
            except Exception as exc:
                with self._lock:
                    self._error = exc

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def consume_ready(self) -> tuple[list[dict[str, float]], torch.Tensor | None] | None:
        if self.is_running():
            return None
        with self._lock:
            if self._error is not None:
                exc = self._error
                self._error = None
                raise RuntimeError("Background action-chunk inference failed.") from exc
            if self._result is None:
                return None
            out = self._result
            self._result = None
            return out


def _execute_dual_chunk_async(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    cameras: dict[str, Any],
    actions: list[dict[str, float]],
    step_duration_s: float,
    keyboard: KeyboardListener | None,
    safe_mode: bool,
    trigger_next_prediction: callable,
    trigger_step: int,
    recorder: TrainRawEpisodeRecorder | None = None,
    end_traj_recorder: EndEffectorTrajectoryRecorder | None = None,
    joint_traj_recorder: JointTrajectoryRecorder | None = None,
) -> tuple[LoopState, bool, bool]:
    state = LoopState.RUNNING
    request_next_chunk = not safe_mode
    running = True
    started_next_prediction = False
    trigger_step = max(1, int(trigger_step))
    for index, action in enumerate(actions):
        step_start = time.perf_counter()
        if keyboard is not None:
            key = keyboard.get_key()
            state, request_next_chunk, running, _vr = _run_keyboard_command(
                key=key,
                left_arm=left_arm,
                right_arm=right_arm,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
                recorder=recorder,
                end_traj_recorder=end_traj_recorder,
                joint_traj_recorder=joint_traj_recorder,
            )
            if state != LoopState.RUNNING or not running:
                break

        observation_for_step: dict[str, Any] | None = None
        if recorder is not None and recorder.recording_active:
            observation_for_step = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)

        left_joint, right_joint = _split_dual_action(action)
        left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
        right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()

        if end_traj_recorder is not None:
            end_traj_recorder.record(step_index=index, left_arm=left_arm, right_arm=right_arm)
        if joint_traj_recorder is not None:
            joint_traj_recorder.record(
                step_index=index,
                left_cmd=left_joint,
                right_cmd=right_joint,
                left_arm=left_arm,
                right_arm=right_arm,
            )

        if recorder is not None and recorder.recording_active and observation_for_step is not None:
            recorder.record_policy_step(action_dict=action, observation=observation_for_step)

        # Trigger async inference while continuing to execute the remaining actions.
        if not started_next_prediction and (index + 1) >= trigger_step:
            latest_observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
            try:
                started_next_prediction = bool(trigger_next_prediction(latest_observation))
            except Exception:
                logger.exception("Failed to trigger background chunk inference.")

        step_elapsed = time.perf_counter() - step_start
        if step_elapsed > step_duration_s:
            logger.warning(
                "控制步超时：step=%d elapsed=%.4fs > duration=%.4fs (overrun=%.4fs)",
                index,
                step_elapsed,
                step_duration_s,
                step_elapsed - step_duration_s,
            )
        precise_sleep(max(step_duration_s - step_elapsed, 0.0))

    if recorder is not None:
        recorder.finish_active_segment()
    return state, request_next_chunk, running


def _run_vr_teleop_session(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    cameras: dict[str, Any],
    camera_names: list[str],
    camera_specs: dict[str, str],
    keyboard: KeyboardListener | None,
    args: argparse.Namespace,
    recorder: TrainRawEpisodeRecorder | None = None,
) -> tuple[ARX5ArmClient, ARX5ArmClient]:
    """
    Hold current joints, release the infer stack's clients, then run dual-arm VR teleop.

    This matches the single-arm infer->VR handoff style in the current repo: no second CAN open,
    no home reset on handoff, and return to infer in STOPPED state after [X].
    """
    if args.use_stub:
        logger.error("VR 遥操作需要真实机械臂；请勿使用 --use-stub。")
        return left_arm, right_arm

    _ensure_vr2robot_import()
    from xrobotoolkit_teleop.hardware.arx_x5_teleop_controller import (
        ARXX5TeleopController,
        DEFAULT_DUAL_ARX_X5_MANIPULATOR_CONFIG,
        DEFAULT_DUAL_ARX_X5_URDF_PATH,
    )

    vr_keyboard: KeyboardListener | None = None
    if keyboard is not None:
        keyboard.pause()
        vr_keyboard = KeyboardListener()

    cameras_to_reconnect = [name for name, camera in cameras.items() if camera.is_connected]
    for name in cameras_to_reconnect:
        logger.info("Disconnecting infer camera '%s' before VR teleop.", name)
        cameras[name].disconnect()

    left_runtime = Arx5Runtime.get_or_create(
        can_port=args.left_can_port,
        arm_type=args.arm_type,
        dt=1.0 / max(1, args.vr_control_hz),
        use_stub=args.use_stub,
        recorded_pose_path=left_arm.recorded_pose_path,
    )
    right_runtime = Arx5Runtime.get_or_create(
        can_port=args.right_can_port,
        arm_type=args.arm_type,
        dt=1.0 / max(1, args.vr_control_hz),
        use_stub=args.use_stub,
        recorded_pose_path=right_arm.recorded_pose_path,
    )
    runtime_by_name = {
        "left_arm": left_runtime,
        "right_arm": right_runtime,
    }

    logger.info("Holding current dual-arm pose before VR teleop (freeze infer targets).")
    left_arm.hold_position()
    right_arm.hold_position()
    time.sleep(0.1)

    pre_switch_state = {
        "left_arm": np.asarray(left_arm.get_state(), dtype=np.float64),
        "right_arm": np.asarray(right_arm.get_state(), dtype=np.float64),
    }

    logger.info("Handing off the same ARX5 clients to VR (no CAN reconnect; no protect_mode on arms).")
    left_runtime.bind_client(left_arm)
    right_runtime.bind_client(right_arm)

    vr_cam = not args.vr_no_camera
    if args.use_usb_cams and vr_cam:
        logger.warning("**********VR 相机线程仅支持 RealSense 序列号。USB 模式下已关闭 VR 相机采集。**********")
        vr_cam = False
    vr_camera_serial_dict = {name: camera_specs[name] for name in camera_names if name in camera_specs}

    def _hold_arm_with_vr_gripper_target(
        arm_name: str,
        arm_client: ARX5ArmClient,
        controller: Any | None,
    ) -> None:
        current_state = np.asarray(arm_client.get_state(), dtype=np.float64)
        hold_pose = current_state.copy()
        if hold_pose.size < 7:
            hold_pose = np.pad(hold_pose, (0, max(0, 7 - hold_pose.size)))
        if controller is not None:
            try:
                gripper_config = controller.manipulator_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]
                hold_pose[6] = float(controller.gripper_pos_target[arm_name][joint_name])
            except Exception:
                logger.exception("Failed to read final VR gripper target for %s; falling back to arm state.", arm_name)
        arm_client.send_joint(hold_pose[:7].astype(np.float64, copy=False).tolist())
        time.sleep(0.2)

    class _VRHoldController(ARXX5TeleopController):
        """Same-process VR: no go_home; hold current pose until each controller is activated."""

        def _request_vr_exit_if_needed(self) -> bool:
            if vr_keyboard is None:
                return False
            key = vr_keyboard.get_key()
            if key == "x":
                logger.info("检测到 [X]：正在退出 VR 遥操作并返回推理。")
                stop_event = getattr(self, "_stop_event", None)
                if stop_event is not None:
                    stop_event.set()
                return True
            return False

        def _robot_setup(self):
            self.arm_controllers = {}
            self._hold_joint_targets: dict[str, np.ndarray] = {}
            self._initial_gripper_by_arm: dict[str, float] = {}
            self._vr_gripper_hold_until_trigger = {arm_name: True for arm_name in self.can_ports}
            self._last_vr_images: dict[str, np.ndarray] = {}
            for arm_name in ARM_ORDER:
                if arm_name not in self.can_ports:
                    continue
                logger.info("Binding ARX X5 %s to shared runtime (no second SingleArm).", arm_name)
                arm = SharedARXX5Interface(runtime_by_name[arm_name], dt=self.dt)
                self.arm_controllers[arm_name] = arm

                current_state = pre_switch_state[arm_name]
                current_joint = current_state[:6].astype(np.float64, copy=True)
                self._hold_joint_targets[arm_name] = current_joint.copy()
                arm.set_joint_positions(current_joint)

                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]
                gripper_value = float(current_state[6])
                self.gripper_pos_target[arm_name][joint_name] = gripper_value
                self._initial_gripper_by_arm[arm_name] = gripper_value
                arm.set_catch_pos(gripper_value)
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
            if self._request_vr_exit_if_needed():
                return
            super()._update_gripper_target()
            for arm_name, should_hold in self._vr_gripper_hold_until_trigger.items():
                if not should_hold:
                    continue
                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                trigger_name = gripper_config["gripper_trigger"]
                if self.xr_client.get_key_value_by_name(trigger_name) > 0.12:
                    self._vr_gripper_hold_until_trigger[arm_name] = False

            for arm_name, should_hold in self._vr_gripper_hold_until_trigger.items():
                if not should_hold:
                    continue
                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]
                self.gripper_pos_target[arm_name][joint_name] = self._initial_gripper_by_arm[arm_name]

        def _send_command(self):
            if self._request_vr_exit_if_needed():
                return
            q_cmd_by_arm: dict[str, np.ndarray] = {}
            gripper_by_arm: dict[str, float] = {}
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
                q_cmd_by_arm[arm_name] = np.asarray(q_cmd, dtype=np.float64)

                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]
                gripper_target = float(self.gripper_pos_target[arm_name][joint_name])
                controller.set_catch_pos(gripper_target)
                gripper_by_arm[arm_name] = gripper_target

            if recorder is None:
                return
            if not recorder.recording_active:
                recorder.finish_active_segment()
                return

            should_record_vr_step = _is_any_vr_arm_active(self.active)
            if not should_record_vr_step:
                if recorder.is_segment_active():
                    logger.info("VR 手柄已松开：暂停原始 VR 录制。")
                    recorder.finish_active_segment()
                return

            images: dict[str, np.ndarray] = {}
            if self.camera_interface is not None:
                frames_by_serial = self.camera_interface.get_frames()
                for serial, frame_data in frames_by_serial.items():
                    camera_name = self.camera_serial_to_name.get(serial)
                    if camera_name is None or camera_name not in recorder.camera_names:
                        continue
                    color = frame_data.get("color")
                    if color is not None:
                        images[camera_name] = color
            self._last_vr_images.update(images)
            if not all(camera_name in self._last_vr_images for camera_name in recorder.camera_names):
                return

            if not recorder.is_segment_active():
                logger.info("VR 手柄已握持：开始原始 VR 录制。")
                recorder.start_vr_segment()
                recorder.mark_vr_takeover()

            full_images = {camera_name: self._last_vr_images[camera_name] for camera_name in recorder.camera_names}
            action_vector = _serialize_dual_arm_targets(
                q_cmd_by_arm=q_cmd_by_arm,
                gripper_by_arm=gripper_by_arm,
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
                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]
                controller.set_catch_pos(float(self.gripper_pos_target[arm_name][joint_name]))

    logger.info(
        "正在启动双臂 VR 遥操作。握住对应手柄移动该臂；扳机控制夹爪。"
        "仅在任一手柄握持期间录制原始 VR 数据。在此按 [X] 可结束 VR 并返回推理。"
    )
    controller: _VRHoldController | None = None
    try:
        controller = _VRHoldController(
            robot_urdf_path=DEFAULT_DUAL_ARX_X5_URDF_PATH,
            manipulator_config=DEFAULT_DUAL_ARX_X5_MANIPULATOR_CONFIG,
            can_ports={"left_arm": args.left_can_port, "right_arm": args.right_can_port},
            scale_factor=args.vr_scale_factor,
            enable_camera=vr_cam,
            enable_camera_display=vr_cam and not args.vr_no_camera_display,
            camera_serial_dict=vr_camera_serial_dict,
            camera_width=args.cam_width,
            camera_height=args.cam_height,
            camera_fps=args.fps,
            enable_log_data=False,
            control_rate_hz=args.vr_control_hz,
            visualize_placo=args.vr_visualize_placo,
        )
        controller.run()
    except KeyboardInterrupt:
        logger.warning("**********VR 遥操作收到 Ctrl+C。当前推荐使用 [X] 退出 VR。**********")
    except Exception:
        logger.exception("VR teleop failed.")
    finally:
        if recorder is not None:
            recorder.finish_active_segment()

        if controller is not None:
            try:
                controller.xr_client.close()
                logger.info("Closed XR client after VR teleop.")
            except Exception:
                logger.exception("Failed to close XR client after VR teleop.")

        left_arm_back = left_runtime.take_client() or left_arm
        right_arm_back = right_runtime.take_client() or right_arm
        try:
            _hold_arm_with_vr_gripper_target("left_arm", left_arm_back, controller)
            _hold_arm_with_vr_gripper_target("right_arm", right_arm_back, controller)

            for name in camera_names:
                if name not in cameras_to_reconnect:
                    continue
                logger.info("Reconnecting infer camera '%s' after VR teleop.", name)
                cameras[name].connect()
        finally:
            if vr_keyboard is not None:
                vr_keyboard.restore()
            if keyboard is not None:
                keyboard.resume()
        logger.info("推理键盘已重新生效。仍为停止状态，请按 [R] 运行策略。")
        return left_arm_back, right_arm_back


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot pi05 dual-arm policy on two ARX5 arms.",
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
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help=(
            "Step index to trigger background next-chunk inference. "
            "This does not truncate execution of the current predicted action chunk."
        ),
    )
    parser.add_argument(
        "--RTC",
        action="store_true",
        default=False,
        help="If set, enable LeRobot RTC (Real-Time Chunking) optimization when predicting action chunks.",
    )
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step.")

    parser.add_argument("--left-can-port", type=str, default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", type=str, default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "base:254522071216",
            "left_wrist:150622073629",
            "right_wrist:409122272986",
        ],
        help="Camera specs as name:serial_or_index using policy image slot names.",
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
        help="VR teleop motion scale.",
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
        "--end-traj-dir",
        type=Path,
        default=None,
        help="If set, record and save dual-arm end-effector xyz trajectory as JSON.",
    )
    parser.add_argument(
        "--joint-traj-dir",
        type=Path,
        default=None,
        help="If set, record and save commanded+actual dual-arm joint trajectories as JSON. Press [R] to start and [Space] to stop.",
    )
    parser.add_argument(
        "--raw-train-record-dir",
        type=Path,
        default=None,
        help="If set, enable raw training recording controlled by keyboard: S starts, R resumes policy, D ends, 0/1/2 finalize failure/success/abandon.",
    )
    # TODO: 讨论一下要不要修改
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")

    args = parser.parse_args()

    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("除非使用 --list-cameras，否则必须提供 --task。")
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        parser.error("--execution-horizon 必须为正数。")
    if args.safe_mode and args.no_keyboard:
        parser.error("安全模式需要键盘控制。")

    logger.info("Loading policy config from %s", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    state_feature = None if policy_cfg.input_features is None else policy_cfg.input_features.get(f"{OBS_STR}.state")
    if state_feature is None or state_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects observation.state dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(state_feature, 'shape', None)}."
        )
    action_feature = None if policy_cfg.output_features is None else policy_cfg.output_features.get(ACTION)
    if action_feature is None or action_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects action dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(action_feature, 'shape', None)}."
        )

    image_slot_names = _visual_image_slot_names(policy_cfg)
    required_image_slot_names = _required_image_slot_names(policy_cfg)
    logger.info("Policy image slots: %s", image_slot_names)
    if len(required_image_slot_names) != len(image_slot_names):
        ignored_slots = sorted(set(image_slot_names) - set(required_image_slot_names))
        logger.info("Ignoring padded image slots without hardware cameras: %s", ignored_slots)

    camera_specs = _parse_camera_specs(args.cameras)
    extra_specs = sorted(set(camera_specs) - set(required_image_slot_names))
    if extra_specs:
        logger.warning("Ignoring --cameras entries not consumed by the runtime: %s", extra_specs)

    camera_configs = _make_camera_configs(
        required_image_slot_names=required_image_slot_names,
        camera_specs=camera_specs,
        use_usb_cams=args.use_usb_cams,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
        flipped_cameras=set(args.flip_cameras),
    )
    cameras = make_cameras_from_configs(camera_configs)
    camera_names = list(cameras)

    left_arm = ARX5ArmClient(
        can_port=args.left_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/left_recorded_pose.json"),
    )
    right_arm = ARX5ArmClient(
        can_port=args.right_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/right_recorded_pose.json"),
    )

    policy_cfg, policy, preprocessor, postprocessor, device = _load_policy_bundle(
        policy_path=args.policy_path,
        device_override=args.policy_device,
        stats_path=args.stats_path,
        policy_cfg=policy_cfg,
    )
    dataset_features = _build_dataset_features(camera_configs)
    raw_recorder: TrainRawEpisodeRecorder | None = None
    if args.raw_train_record_dir is not None:
        raw_recorder = TrainRawEpisodeRecorder(
            raw_record_root=args.raw_train_record_dir,
            action_keys=STATE_KEYS,
            camera_names=camera_names,
            task=args.task,
            dt_s=args.duration,
            image_save_size_hw=(args.cam_height, args.cam_width),
        )
    end_traj_recorder: EndEffectorTrajectoryRecorder | None = None
    if args.end_traj_dir is not None:
        end_traj_recorder = EndEffectorTrajectoryRecorder(args.end_traj_dir)
    joint_traj_recorder: JointTrajectoryRecorder | None = None
    if args.joint_traj_dir is not None:
        joint_traj_recorder = JointTrajectoryRecorder(args.joint_traj_dir)

    execution_horizon = args.execution_horizon or int(policy_cfg.n_action_steps)
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    round_index = _discover_next_round_index(args.record_dir) if args.record_dir is not None else 0
    running = True
    camera_failure: Exception | None = None

    try:
        logger.info("Connecting cameras.")
        for name, camera in cameras.items():
            logger.info("Connecting camera '%s' via %s", name, camera)
            camera.connect()
            logger.info("Camera '%s' connected.", name)

        logger.info(
            "Dual-arm runtime ready. Left CAN=%s Right CAN=%s Stub=%s",
            args.left_can_port,
            args.right_can_port,
            args.use_stub,
        )
        if keyboard is not None:
            logger.info("正在将双臂 ARX5 归零。")
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
            _log_keyboard_help(args.safe_mode)
            if raw_recorder is not None:
                logger.info(
                    "原始训练录制：按 [S] 开始录制窗口，再按 [R] 开始推理。按 [D] 结束录制。"
                    "使用 [V] 进入 VR 后，仅在握持任一手柄时录制 VR 数据，按 [X] 退出 VR。"
                    "按 [D] 后请按 [0]（失败）、[1]（成功）或 [2]（放弃）。"
                    "若发生过 VR 接管，[0] 不可用，须选择 [1] 或 [2]。"
                )
            if args.safe_mode:
                logger.info("安全模式已启用。请按 [R] 恢复运行，每段 chunk 按 [I] 继续。")
        else:
            logger.info("正在无键盘控制下持续运行双臂推理循环。")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        async_predictor = _AsyncChunkPredictor(
            dataset_features=dataset_features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=args.task,
            use_amp=policy_cfg.use_amp,
            full_chunk_steps=int(policy_cfg.n_action_steps),
            rtc_enabled=args.RTC,
        )
        prefetched: tuple[list[dict[str, float]], torch.Tensor | None] | None = None
        has_executed_chunk = False
        current_raw_chunk: torch.Tensor | None = None

        while running:
            if keyboard is not None:
                previous_state = state
                key = keyboard.get_key()
                state, request_next_chunk, running, vr_req = _run_keyboard_command(
                    key=key,
                    left_arm=left_arm,
                    right_arm=right_arm,
                    state=state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                    recorder=raw_recorder,
                    end_traj_recorder=end_traj_recorder,
                    joint_traj_recorder=joint_traj_recorder,
                )
                if vr_req:
                    left_arm, right_arm = _run_vr_teleop_session(
                        left_arm=left_arm,
                        right_arm=right_arm,
                        cameras=cameras,
                        camera_names=camera_names,
                        camera_specs=camera_specs,
                        keyboard=keyboard,
                        args=args,
                        recorder=raw_recorder,
                    )
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    prefetched = None
                    current_raw_chunk = None
                    state = LoopState.STOPPED
                    request_next_chunk = False if args.safe_mode else True
                    continue
                if previous_state != LoopState.RUNNING and state == LoopState.RUNNING:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    prefetched = None
                    current_raw_chunk = None
            if not running:
                break
            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if args.safe_mode and not request_next_chunk:
                time.sleep(0.05)
                continue

            used_prefetched = prefetched is not None
            try:
                if used_prefetched:
                    actions, current_raw_chunk = prefetched
                    prefetched = None
                    observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
                else:
                    if has_executed_chunk:
                        logger.warning(
                            "**********prefetched is None：后台预取未就绪，回退到同步推理。**********"
                        )
                    observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
            except (DeviceNotConnectedError, TimeoutError, RuntimeError) as error:
                camera_failure = error
                _home_dual_arms_after_camera_failure(left_arm=left_arm, right_arm=right_arm, error=error)
                state = LoopState.STOPPED
                request_next_chunk = False
                running = False
                break

            if not used_prefetched:
                sync_infer_start = time.perf_counter()
                if args.RTC:
                    actions, current_raw_chunk = _predict_action_chunk_with_kwargs(
                        robot_observation=observation,
                        dataset_features=dataset_features,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        task=args.task,
                        robot_type="arx5_dual",
                        execution_horizon=int(policy_cfg.n_action_steps),
                        use_amp=policy_cfg.use_amp,
                        predict_kwargs={
                            "prev_chunk_left_over": None,
                            "inference_delay": 0,
                            # Keep RTC execution_horizon semantics consistent with later async calls:
                            # execute full current chunk, while `execution_horizon` CLI is only the prefetch trigger.
                            "execution_horizon": int(policy_cfg.n_action_steps),
                        },
                    )
                else:
                    actions = _predict_action_chunk(
                        robot_observation=observation,
                        dataset_features=dataset_features,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        task=args.task,
                        robot_type="arx5_dual",
                        execution_horizon=int(policy_cfg.n_action_steps),
                        use_amp=policy_cfg.use_amp,
                    )
                    current_raw_chunk = None
                sync_infer_elapsed = time.perf_counter() - sync_infer_start
                if has_executed_chunk:
                    logger.warning(
                        "**********同步推理耗时：%.4fs（prefetched 为 None）**********",
                        sync_infer_elapsed,
                    )
            current_state = [float(observation[key]) for key in STATE_KEYS]
            if args.safe_mode:
                actions = _clip_safe_actions(
                    actions,
                    current_state=current_state,
                    max_joint_step=args.max_joint_step,
                )
            if not actions:
                logger.warning("**********策略未返回动作，正在重试。**********")
                time.sleep(0.1)
                continue

            left_grippers = [round(float(action[LEFT_STATE_KEYS[-1]]), 4) for action in actions]
            right_grippers = [round(float(action[RIGHT_STATE_KEYS[-1]]), 4) for action in actions]
            logger.info(
                "Predicted %d actions. Left grippers: %s Right grippers: %s",
                len(actions),
                left_grippers,
                right_grippers,
            )
            _log_predicted_actions(actions)
            request_next_chunk = False

            if args.record_dir is not None:
                round_dir = _save_chunk_io(
                    record_dir=args.record_dir,
                    round_index=round_index,
                    observation=observation,
                    state=current_state,
                    actions=actions,
                    camera_names=camera_names,
                )
                logger.info("已将 chunk %d 保存至 %s", round_index, round_dir)
                round_index += 1
                if args.safe_mode:
                    logger.info("安全模式：请按 [I] 获取下一段 chunk。")
                if raw_recorder is None:
                    continue

            if raw_recorder is not None and raw_recorder.recording_active:
                raw_recorder.start_policy_segment()

            # `execution_horizon` is used as the async prefetch trigger step only.
            # We still execute the full `actions` chunk returned by the model.
            if len(actions) <= 1:
                trigger_step = 1
            else:
                trigger_step = min(max(1, int(execution_horizon)), len(actions) - 1)
            state, request_next_chunk, running = _execute_dual_chunk_async(
                left_arm=left_arm,
                right_arm=right_arm,
                cameras=cameras,
                actions=actions,
                step_duration_s=args.duration,
                keyboard=keyboard,
                safe_mode=args.safe_mode,
                trigger_next_prediction=(
                    (lambda obs: async_predictor.start(observation=obs))
                    if not args.RTC
                    else (
                        lambda obs: async_predictor.start(
                            observation=obs,
                            rtc_prev_left_over=(
                                None
                                if current_raw_chunk is None or trigger_step >= int(current_raw_chunk.shape[0])
                                else current_raw_chunk[trigger_step:].detach()
                            ),
                            rtc_inference_delay=max(int(len(actions) - trigger_step), 0),
                            rtc_execution_horizon=int(len(actions)),
                        )
                    )
                ),
                trigger_step=trigger_step,
                recorder=raw_recorder,
                end_traj_recorder=end_traj_recorder,
                joint_traj_recorder=joint_traj_recorder,
            )
            try:
                ready = async_predictor.consume_ready()
            except Exception:
                logger.exception("Background chunk inference failed; falling back to synchronous prediction.")
                ready = None
            if state == LoopState.RUNNING and running:
                prefetched = ready
            has_executed_chunk = True
            if args.safe_mode and state == LoopState.RUNNING:
                logger.info("安全模式：请按 [I] 获取下一段 chunk。")
    finally:
        if raw_recorder is not None:
            try:
                raw_recorder.finish_active_segment()
            except Exception:
                logger.exception("Failed to flush raw recorder segment on exit.")
            if raw_recorder.recording_active:
                logger.warning("**********退出时原始录制仍在进行。请使用 [D] 正常结束一回合。**********")
            if raw_recorder.waiting_for_success_label:
                logger.warning("**********仍在等待成功/失败标签（请按 [0]、[1] 或 [2]），但进程已退出。**********")
        if keyboard is not None:
            keyboard.restore()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if args.protect_on_disconnect:
            left_arm.protect_mode()
            right_arm.protect_mode()
        if end_traj_recorder is not None:
            try:
                output_path = end_traj_recorder.save()
                logger.info("Saved end-effector trajectory to %s", output_path)
            except Exception:
                logger.exception("Failed to save end-effector trajectory.")
        if joint_traj_recorder is not None and joint_traj_recorder.num_samples > 0:
            try:
                output_path = joint_traj_recorder.save()
                logger.info("Saved joint trajectory to %s", output_path)
            except Exception:
                logger.exception("Failed to save joint trajectory.")
        if camera_failure is not None:
            logger.error("因相机故障终止推理：%s", camera_failure)


if __name__ == "__main__":
    main()
