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

"""Run dual-arm LeRobot Pi0.5 RTC inference directly on two ARX5 arms."""

import argparse
import dataclasses
import logging
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient
from lerobot.robots.arx5_follower.config_arx5_follower import ARX5FollowerConfigBase
from lerobot.scripts.lerobot_arx5_infer import KeyboardListener, LoopState, _list_realsense_cameras, _load_policy_bundle
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "lerobot/pi05_bimanual"
DEFAULT_LEFT_CAN_PORT = "can0"
DEFAULT_RIGHT_CAN_PORT = "can1"
DEFAULT_STATS_PATH = files("lerobot.robots.arx5_follower").joinpath("pi05_arx5_default_stats.json")
DEFAULT_EXECUTION_HORIZON = 30
DEFAULT_PREFETCH_THRESHOLD = 10
DEFAULT_RTC_INFERENCE_DELAY_STEPS = 6
DEFAULT_RTC_MAX_GUIDANCE_WEIGHT = 10.0
ARX5_GRIPPER_FULLY_CLOSED = float(
    next(f.default for f in dataclasses.fields(ARX5FollowerConfigBase) if f.name == "gripper_min")
)
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
LEFT_STATE_KEYS = STATE_KEYS[7:]
RIGHT_STATE_KEYS = STATE_KEYS[:7]


@dataclass
class RTCRequest:
    generation_id: int
    request_id: int
    observation: dict[str, Any]
    task: str
    execution_horizon: int
    inference_delay: int
    prev_chunk_left_over: torch.Tensor | None
    submit_time: float
    submit_global_step: int


@dataclass
class RTCResult:
    generation_id: int
    request_id: int
    infer_latency_s: float
    action_chunk: torch.Tensor
    actions: list[dict[str, float]]
    submit_time: float
    submit_global_step: int
    error: str | None = None


@dataclass
class RTCExecutionState:
    generation_id: int = 0
    request_id: int = 0
    action_queue: list[dict[str, float]] = dataclasses.field(default_factory=list)
    in_flight_request_id: int | None = None
    last_completed_request_id: int | None = None
    global_step: int = 0
    chunk_index: int = 0
    last_infer_latency_s: float | None = None
    last_infer_delay_steps: int = 0


def _visual_image_slot_names(policy_cfg: PreTrainedConfig) -> list[str]:
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
    state = right_state[:7] + left_state[:7]
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
                "**********动作裁剪：已限制双臂关节步长，单关节最大变化=%.4f。**********",
                max_joint_step,
            )
        previous_state = target_state
        safe_actions.append(safe_action)
    return safe_actions


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


def _action_chunk_to_actions(
    *,
    action_chunk: torch.Tensor,
    dataset_features: dict[str, dict[str, Any]],
    execution_horizon: int,
) -> list[dict[str, float]]:
    action_names = dataset_features[ACTION]["names"]
    chunk_cpu = action_chunk.squeeze(0).to("cpu")
    horizon = min(int(execution_horizon), int(chunk_cpu.shape[0]))
    actions: list[dict[str, float]] = []
    for index in range(horizon):
        row = chunk_cpu[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions


def _action_dicts_to_tensor(
    actions: list[dict[str, float]],
    dataset_features: dict[str, dict[str, Any]],
) -> torch.Tensor:
    action_names = dataset_features[ACTION]["names"]
    if not actions:
        return torch.empty((0, len(action_names)), dtype=torch.float32)
    rows = [[float(action[name]) for name in action_names] for action in actions]
    return torch.tensor(rows, dtype=torch.float32)


def _predict_action_chunk_tensor(
    *,
    robot_observation: dict[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    use_amp: bool,
    prev_chunk_left_over: torch.Tensor | None,
    inference_delay: int,
    execution_horizon: int,
) -> torch.Tensor:
    observation_frame = build_dataset_frame(dataset_features, robot_observation, prefix=OBS_STR)
    processed_observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task=task,
        robot_type=robot_type,
    )
    if prev_chunk_left_over is not None:
        prev_chunk_left_over = prev_chunk_left_over.to(device=device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        processed_observation = preprocessor(processed_observation)
        action_chunk_raw = policy.predict_action_chunk(
            processed_observation,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        action_chunk = postprocessor(action_chunk_raw)
    return action_chunk


def _inference_worker_loop(
    *,
    request_queue: queue.Queue[RTCRequest | None],
    result_queue: queue.Queue[RTCResult],
    stop_event: threading.Event,
    policy,
    preprocessor,
    postprocessor,
    dataset_features: dict[str, dict[str, Any]],
    device: torch.device,
    use_amp: bool,
    robot_type: str,
) -> None:
    while not stop_event.is_set():
        try:
            request = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if request is None:
            return

        infer_start = time.perf_counter()
        try:
            action_chunk = _predict_action_chunk_tensor(
                robot_observation=request.observation,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=request.task,
                robot_type=robot_type,
                use_amp=use_amp,
                prev_chunk_left_over=request.prev_chunk_left_over,
                inference_delay=request.inference_delay,
                execution_horizon=request.execution_horizon,
            )
            actions = _action_chunk_to_actions(
                action_chunk=action_chunk,
                dataset_features=dataset_features,
                execution_horizon=request.execution_horizon,
            )
        except Exception:
            logger.exception("RTC worker inference failed for request_id=%d", request.request_id)
            result = RTCResult(
                generation_id=request.generation_id,
                request_id=request.request_id,
                infer_latency_s=time.perf_counter() - infer_start,
                action_chunk=torch.empty((1, 0, len(dataset_features[ACTION]["names"])), dtype=torch.float32),
                actions=[],
                submit_time=request.submit_time,
                submit_global_step=request.submit_global_step,
                error="worker_inference_failed",
            )
            while not stop_event.is_set():
                try:
                    result_queue.put(result, timeout=0.1)
                    break
                except queue.Full:
                    continue
            continue
        infer_latency_s = time.perf_counter() - infer_start

        result = RTCResult(
            generation_id=request.generation_id,
            request_id=request.request_id,
            infer_latency_s=infer_latency_s,
            action_chunk=action_chunk.detach().to("cpu"),
            actions=actions,
            submit_time=request.submit_time,
            submit_global_step=request.submit_global_step,
        )
        while not stop_event.is_set():
            try:
                result_queue.put(result, timeout=0.1)
                break
            except queue.Full:
                continue


def _run_keyboard_command(
    *,
    key: str | None,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    state: LoopState,
) -> tuple[LoopState, bool]:
    if key == " ":
        left_arm.hold_position()
        right_arm.hold_position()
        logger.info("急停：已保持当前姿态。")
        return LoopState.STOPPED, True
    if key == "q":
        logger.info("用户请求退出。")
        return state, False
    if key == "o":
        if state != LoopState.STOPPED:
            logger.warning("**********[O] 已忽略：仅在停止状态下可张开夹爪；请先按 [Space] 急停。**********")
            return state, True
        for arm in (left_arm, right_arm):
            pose = arm.get_state()
            arm.send_joint([float(pose[i]) for i in range(6)] + [ARX5_GRIPPER_FULLY_OPEN])
        logger.info("已张开双臂夹爪（O）。")
        return state, True
    if key == "p":
        if state != LoopState.STOPPED:
            logger.warning("**********[P] 已忽略：仅在停止状态下可闭合夹爪；请先按 [Space] 急停。**********")
            return state, True
        for arm in (left_arm, right_arm):
            pose = arm.get_state()
            arm.send_joint([float(pose[i]) for i in range(6)] + [ARX5_GRIPPER_FULLY_CLOSED])
        logger.info("已闭合双臂夹爪（P）。")
        return state, True

    if state == LoopState.STOPPED:
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
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            logger.info("已开始 RTC 推理。")
            return LoopState.RUNNING, True
    return state, True


def _log_keyboard_help() -> None:
    logger.info(
        "键盘：[Space] 急停 | [H] 回零 | [R] 执行推理 | [Q] 退出 | [O] open grippers | [P] close grippers"
    )


def _reset_execution_state(state: RTCExecutionState) -> None:
    state.generation_id += 1
    state.request_id = 0
    state.action_queue.clear()
    state.in_flight_request_id = None
    state.last_completed_request_id = None
    state.chunk_index = 0
    state.last_infer_latency_s = None
    state.last_infer_delay_steps = 0


def _drain_queue(q: queue.Queue[Any]) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _submit_rtc_request(
    *,
    request_queue: queue.Queue[RTCRequest | None],
    execution_state: RTCExecutionState,
    observation: dict[str, Any],
    task: str,
    execution_horizon: int,
    inference_delay: int,
    dataset_features: dict[str, dict[str, Any]],
) -> bool:
    if execution_state.in_flight_request_id is not None:
        return False
    prev_left_over = _action_dicts_to_tensor(execution_state.action_queue, dataset_features)
    request_id = execution_state.request_id + 1
    request = RTCRequest(
        generation_id=execution_state.generation_id,
        request_id=request_id,
        observation=observation,
        task=task,
        execution_horizon=execution_horizon,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_left_over if prev_left_over.numel() > 0 else None,
        submit_time=time.perf_counter(),
        submit_global_step=execution_state.global_step,
    )
    try:
        request_queue.put_nowait(request)
    except queue.Full:
        return False
    execution_state.request_id = request_id
    execution_state.in_flight_request_id = request_id
    logger.info(
        "RTC request submitted: generation=%d request=%d queue_remaining=%d inference_delay=%d",
        request.generation_id,
        request.request_id,
        len(execution_state.action_queue),
        inference_delay,
    )
    return True


def _consume_ready_rtc_result(
    *,
    result_queue: queue.Queue[RTCResult],
    execution_state: RTCExecutionState,
    clip_actions: bool,
    max_joint_step: float,
    current_state: list[float],
    control_dt_s: float,
) -> None:
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            return

        if result.generation_id != execution_state.generation_id:
            logger.info(
                "Dropping stale RTC result: generation=%d request=%d current_generation=%d",
                result.generation_id,
                result.request_id,
                execution_state.generation_id,
            )
            continue
        if execution_state.in_flight_request_id != result.request_id:
            logger.info(
                "Dropping unmatched RTC result: request=%d in_flight=%s",
                result.request_id,
                execution_state.in_flight_request_id,
            )
            continue

        execution_state.in_flight_request_id = None
        execution_state.last_completed_request_id = result.request_id
        execution_state.last_infer_latency_s = result.infer_latency_s
        delay_by_time = int(round(result.infer_latency_s / max(1e-6, control_dt_s)))
        executed_steps_since_submit = max(0, execution_state.global_step - result.submit_global_step)
        execution_state.last_infer_delay_steps = executed_steps_since_submit

        if result.error is not None:
            logger.warning(
                "RTC result reported worker error: request=%d error=%s",
                result.request_id,
                result.error,
            )
            continue

        actions = result.actions
        if clip_actions:
            actions = _clip_safe_actions(actions, current_state=current_state, max_joint_step=max_joint_step)
        real_delay = min(executed_steps_since_submit, len(actions))
        execution_state.action_queue = actions[real_delay:]
        execution_state.chunk_index += 1
        logger.info(
            "RTC result merged: request=%d latency=%.4fs wall_delay=%d step_delay=%d applied_delay=%d new_queue=%d",
            result.request_id,
            result.infer_latency_s,
            delay_by_time,
            executed_steps_since_submit,
            real_delay,
            len(execution_state.action_queue),
        )


def _execute_dual_action_step(
    *,
    left_arm: ARX5ArmClient,
    right_arm: ARX5ArmClient,
    action: dict[str, float],
) -> None:
    left_joint, right_joint = _split_dual_action(action)
    left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
    right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
    left_thread.start()
    right_thread.start()
    left_thread.join()
    right_thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LeRobot pi05 dual-arm RTC policy on two ARX5 arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=None, help="Natural language task instruction.")
    parser.add_argument("--policy-path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step.")
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=DEFAULT_RTC_INFERENCE_DELAY_STEPS)
    parser.add_argument("--rtc-prefetch-threshold", type=int, default=DEFAULT_PREFETCH_THRESHOLD)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        type=str,
        default=RTCAttentionSchedule.LINEAR.value,
        choices=[member.value for member in RTCAttentionSchedule],
    )
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=DEFAULT_RTC_MAX_GUIDANCE_WEIGHT)

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

    parser.add_argument("--clip-actions", action="store_true", default=True)
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")

    parsed_args = parser.parse_args()
    global args
    args = parsed_args

    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("除非使用 --list-cameras，否则必须提供 --task。")
    if args.execution_horizon <= 0:
        parser.error("--execution-horizon 必须为正数。")
    if args.rtc_prefetch_threshold <= 0:
        parser.error("--rtc-prefetch-threshold 必须为正数。")
    if args.rtc_inference_delay_steps < 0:
        parser.error("--rtc-inference-delay-steps 必须为非负数。")
    if args.rtc_max_guidance_weight <= 0:
        parser.error("--rtc-max-guidance-weight 必须为正数。")

    logger.info("Loading policy config from %s", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    state_feature = None if policy_cfg.input_features is None else policy_cfg.input_features.get(f"{OBS_STR}.state")
    if state_feature is None or state_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm RTC runtime expects observation.state dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(state_feature, 'shape', None)}."
        )
    action_feature = None if policy_cfg.output_features is None else policy_cfg.output_features.get(ACTION)
    if action_feature is None or action_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm RTC runtime expects action dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(action_feature, 'shape', None)}."
        )

    policy_cfg.rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        max_guidance_weight=float(args.rtc_max_guidance_weight),
        execution_horizon=int(args.execution_horizon),
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
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    dataset_features = _build_dataset_features(camera_configs)

    request_queue: queue.Queue[RTCRequest | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[RTCResult] = queue.Queue(maxsize=1)
    worker_stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_inference_worker_loop,
        kwargs={
            "request_queue": request_queue,
            "result_queue": result_queue,
            "stop_event": worker_stop_event,
            "policy": policy,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "dataset_features": dataset_features,
            "device": device,
            "use_amp": policy_cfg.use_amp,
            "robot_type": "arx5_dual",
        },
        daemon=True,
    )

    execution_state = RTCExecutionState()
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    running = True
    camera_failure: Exception | None = None

    try:
        logger.info("Connecting cameras.")
        for name, camera in cameras.items():
            logger.info("Connecting camera '%s' via %s", name, camera)
            camera.connect()
            logger.info("Camera '%s' connected.", name)

        logger.info(
            "Dual-arm RTC runtime ready. Left CAN=%s Right CAN=%s Stub=%s",
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
            _log_keyboard_help()
        else:
            logger.info("正在无键盘控制下持续运行双臂 RTC 推理循环。")

        worker_thread.start()

        while running:
            if keyboard is not None:
                previous_state = state
                key = keyboard.get_key()
                state, running = _run_keyboard_command(
                    key=key,
                    left_arm=left_arm,
                    right_arm=right_arm,
                    state=state,
                )
                if previous_state != LoopState.RUNNING and state == LoopState.RUNNING:
                    _drain_queue(request_queue)
                    _drain_queue(result_queue)
                    _reset_execution_state(execution_state)
                elif previous_state == LoopState.RUNNING and state != LoopState.RUNNING:
                    _drain_queue(request_queue)
                    _drain_queue(result_queue)
                    _reset_execution_state(execution_state)
                    left_arm.hold_position()
                    right_arm.hold_position()
            if not running:
                break
            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue

            current_state = _read_dual_state(left_arm, right_arm)
            _consume_ready_rtc_result(
                result_queue=result_queue,
                execution_state=execution_state,
                clip_actions=args.clip_actions,
                max_joint_step=args.max_joint_step,
                current_state=current_state,
                control_dt_s=args.duration,
            )

            if not execution_state.action_queue and execution_state.in_flight_request_id is None:
                try:
                    observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
                except (DeviceNotConnectedError, TimeoutError, RuntimeError) as error:
                    camera_failure = error
                    _home_dual_arms_after_camera_failure(left_arm=left_arm, right_arm=right_arm, error=error)
                    state = LoopState.STOPPED
                    running = False
                    break
                _submit_rtc_request(
                    request_queue=request_queue,
                    execution_state=execution_state,
                    observation=observation,
                    task=args.task,
                    execution_horizon=args.execution_horizon,
                    inference_delay=args.rtc_inference_delay_steps,
                    dataset_features=dataset_features,
                )

            if execution_state.action_queue:
                step_start = time.perf_counter()
                action = execution_state.action_queue.pop(0)
                _execute_dual_action_step(left_arm=left_arm, right_arm=right_arm, action=action)
                execution_state.global_step += 1
                step_elapsed = time.perf_counter() - step_start
                if step_elapsed > args.duration:
                    logger.warning(
                        "控制步超时：global_step=%d elapsed=%.4fs > duration=%.4fs (overrun=%.4fs)",
                        execution_state.global_step,
                        step_elapsed,
                        args.duration,
                        step_elapsed - args.duration,
                    )
                precise_sleep(max(args.duration - step_elapsed, 0.0))
            else:
                time.sleep(min(args.duration, 0.01))

            if (
                execution_state.in_flight_request_id is None
                and len(execution_state.action_queue) <= args.rtc_prefetch_threshold
            ):
                try:
                    observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
                except (DeviceNotConnectedError, TimeoutError, RuntimeError) as error:
                    camera_failure = error
                    _home_dual_arms_after_camera_failure(left_arm=left_arm, right_arm=right_arm, error=error)
                    state = LoopState.STOPPED
                    running = False
                    break
                _submit_rtc_request(
                    request_queue=request_queue,
                    execution_state=execution_state,
                    observation=observation,
                    task=args.task,
                    execution_horizon=args.execution_horizon,
                    inference_delay=args.rtc_inference_delay_steps,
                    dataset_features=dataset_features,
                )
    finally:
        worker_stop_event.set()
        try:
            request_queue.put_nowait(None)
        except queue.Full:
            pass
        if worker_thread.is_alive():
            worker_thread.join(timeout=5.0)
        if keyboard is not None:
            keyboard.restore()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if args.protect_on_disconnect:
            left_arm.protect_mode()
            right_arm.protect_mode()
        if camera_failure is not None:
            logger.error("因相机故障终止 RTC 推理：%s", camera_failure)


if __name__ == "__main__":
    main()
