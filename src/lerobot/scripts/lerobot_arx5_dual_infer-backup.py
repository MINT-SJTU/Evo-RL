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
import json
import logging
import threading
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
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
STATE_KEYS = tuple(f"state.{index}" for index in range(DUAL_STATE_DIM))
# TODO: action
LEFT_STATE_KEYS = STATE_KEYS[7:]
RIGHT_STATE_KEYS = STATE_KEYS[:7]
# LEFT_STATE_KEYS = STATE_KEYS[7:]
# RIGHT_STATE_KEYS = STATE_KEYS[:7]
ARM_ORDER = ("left_arm", "right_arm")


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
                "SAFE MODE: capped dual-arm joint step, max per-joint delta=%.4f.",
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
        "Keyboard: [Space] stop | [O] open grippers (when stopped) | [H] home | [B] teach | [N] record pose | "
        "[M] goto pose | [V] VR teleop | [S] start raw recording | [R] resume | [D] end raw recording | [Q] quit"
    )
    if safe_mode:
        base_message += " | [I] next chunk"
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
) -> tuple[LoopState, bool, bool, bool]:
    """Returns (state, request_next_chunk, running, vr_teleop_requested)."""
    if recorder is not None and recorder.handle_success_label_hotkey(key or ""):
        return LoopState.STOPPED, False, True, False

    def _warn_raw_recording_not_active(*, action_name: str, key_name: str) -> None:
        if recorder is None or recorder.recording_active or recorder.waiting_for_success_label:
            return
        logger.warning(
            "%s (%s) while raw recording is not active. "
            "Press [S] to start a recording window first if you want this episode to be saved.",
            action_name,
            key_name,
        )

    if key == "s":
        if recorder is None:
            logger.warning("[S] Raw training recording did not start because --raw-train-record-dir was not specified.")
            return state, request_next_chunk, True, False
        if recorder.waiting_for_success_label:
            logger.info("[S] Ignored: waiting for success label; press [0], [1], or [2] first.")
            return state, request_next_chunk, True, False
        if state != LoopState.STOPPED:
            logger.warning("[S] Ignored: stop policy first, then press [S] to start a new recording window.")
            return state, request_next_chunk, True, False
        if recorder.recording_active:
            logger.info("[S] Ignored: raw recording is already active. Press [D] to end the current recording.")
            return state, request_next_chunk, True, False
        if recorder.can_start_new_episode():
            recorder.start_episode()
            logger.info("Started raw training recording window (S). Press [R] to resume policy, then [D] to finish this episode.")
        return state, request_next_chunk, True, False

    if key == "d":
        if recorder is None:
            logger.warning("[D] Raw training recording is not enabled. Re-run with --raw-train-record-dir <dir> to use [S]/[D].")
            return state, request_next_chunk, True, False
        if recorder.recording_active:
            logger.info("Stopping raw training recording (D).")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.05)
            recorder.request_finish_episode()
            return LoopState.STOPPED, False, True, False
        if recorder.waiting_for_success_label:
            logger.info("[D] Ignored: already waiting for success label; press [0], [1], or [2] to finalize.")
            return state, request_next_chunk, True, False
        logger.warning("[D] Ignored: raw recording is not active. Press [S] to start a recording window first.")
        return state, request_next_chunk, True, False

    if key == " ":
        left_arm.hold_position()
        right_arm.hold_position()
        logger.warning("Emergency stop: holding current poses.")
        return LoopState.STOPPED, False, True, False
    if key == "q":
        logger.info("Quit requested.")
        return state, request_next_chunk, False, False

    if key == "o":
        if state != LoopState.STOPPED:
            logger.warning(
                "[O] Ignored: open grippers only while stopped; press [Space] to emergency-stop first."
            )
            return state, request_next_chunk, True, False
        # send_joint -> set_joint_positions(6) + set_catch_pos(gripper). Fully open == gripper_max (not gripper_min).
        for arm in (left_arm, right_arm):
            pose = arm.get_state()
            arm.send_joint([float(pose[i]) for i in range(6)] + [ARX5_GRIPPER_FULLY_OPEN])
        logger.info("Opened both grippers (O).")
        return state, request_next_chunk, True, False

    if state == LoopState.STOPPED:
        if key == "v":
            _warn_raw_recording_not_active(action_name="VR teleop requested", key_name="[V]")
            logger.info("VR teleop requested from STOPPED state.")
            return state, request_next_chunk, True, True
        if key == "h":
            logger.info("Moving both ARX5 arms to the home pose.")
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
                logger.info("Waiting for success label (press '0', '1', or '2'); ignoring [R] resume.")
                return LoopState.STOPPED, False, True, False
            _warn_raw_recording_not_active(action_name="Resuming policy", key_name="[R]")
            left_arm.hold_position()
            right_arm.hold_position()
            time.sleep(0.1)
            if recorder is not None and recorder.recording_active:
                logger.info("Resumed policy control with raw recording active.")
            else:
                logger.info("Resumed policy control.")
            return LoopState.RUNNING, not safe_mode, True, False
        elif key == "b":
            left_arm.enter_teach_mode()
            right_arm.enter_teach_mode()
            logger.info("Teach mode enabled. Drag both arms, then press [N] to save the poses.")
            return LoopState.TEACHING, request_next_chunk, True, False
        elif key == "m":
            if left_arm.has_recorded_pose() and right_arm.has_recorded_pose():
                left_arm.move_to_recorded()
                right_arm.move_to_recorded()
                logger.info("Moved both arms to the recorded poses.")
            else:
                logger.warning("No recorded poses found. Use [B] then [N] first.")
    elif state == LoopState.TEACHING:
        if key == "n":
            left_arm.save_recorded_pose()
            right_arm.save_recorded_pose()
            left_arm.hold_position()
            right_arm.hold_position()
            logger.info("Recorded poses saved for both arms.")
            return LoopState.STOPPED, False, True, False
    elif state == LoopState.RUNNING and safe_mode and key == "i":
        logger.info("SAFE MODE: next chunk requested.")
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
                left_arm=left_arm,
                right_arm=right_arm,
                state=state,
                safe_mode=safe_mode,
                request_next_chunk=request_next_chunk,
                recorder=recorder,
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

        if recorder is not None and recorder.recording_active and observation_for_step is not None:
            recorder.record_policy_step(action_dict=action, observation=observation_for_step)

        precise_sleep(max(step_duration_s - (time.perf_counter() - step_start), 0.0))

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
    no home reset on handoff, and return to infer in STOPPED state after Ctrl+C.
    """
    if args.use_stub:
        logger.error("VR teleop requires real arms; omit --use-stub.")
        return left_arm, right_arm

    _ensure_vr2robot_import()
    from xrobotoolkit_teleop.hardware.arx_x5_teleop_controller import (
        ARXX5TeleopController,
        DEFAULT_DUAL_ARX_X5_MANIPULATOR_CONFIG,
        DEFAULT_DUAL_ARX_X5_URDF_PATH,
    )

    if keyboard is not None:
        keyboard.pause()

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
        logger.warning("VR camera threads only support RealSense serials. Disabling VR camera capture in USB mode.")
        vr_cam = False
    vr_camera_serial_dict = {name: camera_specs[name] for name in camera_names if name in camera_specs}

    class _VRHoldController(ARXX5TeleopController):
        """Same-process VR: no go_home; hold current pose until each controller is activated."""

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
                    logger.info("VR grips released: pausing raw VR recording.")
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
                logger.info("VR grip engaged: starting raw VR recording.")
                recorder.start_vr_segment()
                recorder.mark_vr_takeover()

            full_images = {camera_name: self._last_vr_images[camera_name] for camera_name in recorder.camera_names}
            action_vector: list[float] = []
            for arm_name in ARM_ORDER:
                action_vector.extend(q_cmd_by_arm[arm_name][:6].astype(np.float64, copy=False).tolist())
                action_vector.append(float(gripper_by_arm[arm_name]))
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
        "Starting dual-arm VR teleop. Hold the corresponding grip to move each arm; use trigger for gripper. "
        "Raw VR data is recorded only while either grip is held. Press Ctrl+C here to stop VR and return to infer."
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
        logger.info("VR teleop stopped (Ctrl+C).")
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
            left_arm_back.hold_position()
            right_arm_back.hold_position()

            for name in camera_names:
                if name not in cameras_to_reconnect:
                    continue
                logger.info("Reconnecting infer camera '%s' after VR teleop.", name)
                cameras[name].connect()
        finally:
            if keyboard is not None:
                keyboard.resume()
        logger.info("Infer keyboard active again. Still STOPPED until you press [R] to run policy.")
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
    parser.add_argument("--execution-horizon", type=int, default=None)
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
        parser.error("--task is required unless --list-cameras is used.")
    if args.execution_horizon is not None and args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive.")
    if args.safe_mode and args.no_keyboard:
        parser.error("SAFE MODE requires keyboard control.")

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
        )

    execution_horizon = args.execution_horizon or int(policy_cfg.n_action_steps)
    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    round_index = _discover_next_round_index(args.record_dir) if args.record_dir is not None else 0
    running = True

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
            logger.info("Moving both ARX5 arms to the home pose.")
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
            _log_keyboard_help(args.safe_mode)
            if raw_recorder is not None:
                logger.info(
                    "Raw training recording: press [S] to start a recording window, then press [R] to resume policy. Press [D] to end recording. "
                    "After [V], VR data is recorded only while either grip is held. "
                    "After [D], press [0] (failure), [1] (success), or [2] (abandon). "
                    "If VR takeover happened, [0] is rejected and you must choose [1] or [2]."
                )
            if args.safe_mode:
                logger.info("SAFE MODE is armed. Press [R] to resume, then [I] for each chunk.")
        else:
            logger.info("Starting continuous dual-arm inference loop without keyboard control.")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

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

            observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
            current_state = [float(observation[key]) for key in STATE_KEYS]
            actions = _predict_action_chunk(
                robot_observation=observation,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=args.task,
                robot_type="arx5_dual",
                execution_horizon=execution_horizon,
                use_amp=policy_cfg.use_amp,
            )
            if args.safe_mode:
                actions = _clip_safe_actions(
                    actions,
                    current_state=current_state,
                    max_joint_step=args.max_joint_step,
                )
            if not actions:
                logger.warning("Policy returned no actions. Retrying.")
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
                logger.info("Saved chunk %d to %s", round_index, round_dir)
                round_index += 1
                if args.safe_mode:
                    logger.info("SAFE MODE: press [I] for the next chunk.")
                if raw_recorder is None:
                    continue

            if raw_recorder is not None and raw_recorder.recording_active:
                raw_recorder.start_policy_segment()

            state, request_next_chunk, running = _execute_dual_chunk(
                left_arm=left_arm,
                right_arm=right_arm,
                cameras=cameras,
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
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if args.protect_on_disconnect:
            left_arm.protect_mode()
            right_arm.protect_mode()


if __name__ == "__main__":
    main()
