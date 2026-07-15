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

import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    PIPER_JOINT_NAMES,
    PIPER_ROLE_FOLLOWER,
    get_piper_sdk,
    milli_to_unit,
    parse_piper_log_level,
    set_piper_role,
    unit_to_milli,
    wait_enable_piper,
)

from ..robot import Robot
from .config_piper_follower import PiperFollowerConfig, PiperXFollowerConfig

logger = logging.getLogger(__name__)


class PiperFollower(Robot):
    """Piper follower arm controlled through the Piper SDK (CAN)."""

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig | PiperXFollowerConfig):
        self.robot_type = self.name
        self.id = config.id
        self.config = config
        self._is_connected = False
        self._last_mode_refresh_t = 0.0
        self._teleop_send_only_mode = False

        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=self.config.port,
            judge_flag=self.config.judge_flag,
            can_auto_init=self.config.can_auto_init,
            logger_level=parse_piper_log_level(self.config.log_level),
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (
            self._teleop_send_only_mode or all(cam.is_connected for cam in self.cameras.values())
        )

    def set_teleop_send_only_mode(self, enabled: bool) -> None:
        if self._is_connected:
            raise RuntimeError("teleop send-only mode must be configured before connecting the robot.")
        self._teleop_send_only_mode = enabled

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm.ConnectPort(
            piper_init=not self._teleop_send_only_mode,
            start_thread=not self._teleop_send_only_mode,
        )
        connected_cameras = []
        try:
            if self.config.startup_sleep_s > 0:
                time.sleep(self.config.startup_sleep_s)

            self._is_connected = True
            self.configure()
            if self.config.enable_on_connect:
                if self._teleop_send_only_mode:
                    self.arm.EnablePiper()
                elif not self._wait_enable(self.config.enable_timeout_s):
                    logger.warning("Piper follower did not report enabled state before timeout.")

            if not self._teleop_send_only_mode:
                for cam in self.cameras.values():
                    cam.connect()
                    connected_cameras.append(cam)
        except Exception:
            self.arm.DisconnectPort()
            for cam in connected_cameras:
                cam.disconnect()
            self._is_connected = False
            raise

        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
        self._send_motion_mode()

    def _send_motion_mode(self) -> None:
        mit_mode = 0xAD if self.config.high_follow else 0x00
        self.arm.MotionCtrl_2(0x01, 0x01, self.config.speed_ratio, mit_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_motion_mode_if_needed(self) -> None:
        interval_s = self.config.mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_motion_mode()

    def _wait_enable(self, timeout_s: float) -> bool:
        return wait_enable_piper(self.arm, timeout_s)

    def _read_raw_observation(self) -> RobotObservation:
        joint_msg = self.arm.GetArmJointMsgs()
        joint_state = getattr(joint_msg, "joint_state", None)

        obs: RobotObservation = {}
        for joint_name in PIPER_JOINT_NAMES:
            raw_value = getattr(joint_state, joint_name, 0)
            obs[f"{joint_name}.pos"] = milli_to_unit(raw_value)

        gripper_msg = self.arm.GetArmGripperMsgs()
        gripper_state = getattr(gripper_msg, "gripper_state", None)
        obs["gripper.pos"] = abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))
        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._teleop_send_only_mode:
            raise RuntimeError(
                f"{self} was connected in teleop send-only mode, so follower observations are unavailable."
            )
        obs: dict[str, float] = self._read_raw_observation()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._refresh_motion_mode_if_needed()

        sent_action: dict[str, float] = {}

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in action for key in joint_keys)
        if has_all_joints:
            joint_targets = [action[key] for key in joint_keys]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)
            sent_action.update(
                {key: milli_to_unit(raw) for key, raw in zip(joint_keys, joint_commands, strict=True)}
            )
        elif any(key in action for key in joint_keys):
            logger.debug("Ignoring partial Piper joint action. Need all six joint keys to send command.")

        if self.config.sync_gripper and "gripper.pos" in action:
            gripper_pos_raw = unit_to_milli(action["gripper.pos"])
            self.arm.GripperCtrl(
                gripper_pos_raw,
                self.config.gripper_effort_default,
                self.config.gripper_status_code,
                0x00,
            )
            sent_action["gripper.pos"] = milli_to_unit(gripper_pos_raw)

        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self.arm.DisableArm(7)
        finally:
            self.arm.DisconnectPort()
            for cam in self.cameras.values():
                if cam.is_connected:
                    cam.disconnect()
            self._is_connected = False
            logger.info("%s disconnected.", self)


class PiperXFollower(PiperFollower):
    config_class = PiperXFollowerConfig
    name = "piperx_follower"
