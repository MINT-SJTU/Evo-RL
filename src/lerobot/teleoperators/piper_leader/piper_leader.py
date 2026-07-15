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
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    PIPER_JOINT_NAMES,
    PIPER_ROLE_FOLLOWER,
    PIPER_ROLE_LEADER,
    get_piper_sdk,
    milli_to_unit,
    parse_piper_log_level,
    set_piper_role,
    unit_to_milli,
    wait_enable_piper,
)

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig, PiperXLeaderConfig

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    """Piper leader arm used as a teleoperator through Piper SDK CAN messages."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig | PiperXLeaderConfig):
        self.id = config.id
        self.config = config
        self._is_connected = False
        self._manual_control_enabled: bool | None = None
        self._last_mode_refresh_t = 0.0

        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=self.config.port,
            judge_flag=self.config.judge_flag,
            can_auto_init=self.config.can_auto_init,
            logger_level=parse_piper_log_level(self.config.log_level),
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm.ConnectPort()
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)

        self._is_connected = True
        self._manual_control_enabled = None
        try:
            self.configure()
        except Exception:
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            raise

        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def _send_command_mode(self) -> None:
        mit_mode = 0xAD if self.config.command_high_follow else 0x00
        self.arm.MotionCtrl_2(0x01, 0x01, self.config.command_speed_ratio, mit_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_command_mode_if_needed(self) -> None:
        interval_s = self.config.mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_command_mode()

    def _send_gripper_ctrl(self, gripper_pos_raw: int, enabled: bool) -> None:
        self.arm.GripperCtrl(
            gripper_pos_raw,
            self.config.gripper_effort_default if enabled else 0,
            self.config.gripper_status_code if enabled else 0x00,
            0x00,
        )

    def _set_gripper_enabled(self, enabled: bool) -> None:
        gripper_pos_raw = 0
        try:
            gripper_msg = self.arm.GetArmGripperMsgs()
            gripper_state = getattr(gripper_msg, "gripper_state", None)
            if gripper_state is not None:
                gripper_pos_raw = abs(int(getattr(gripper_state, "grippers_angle", 0)))
        except Exception:
            logger.debug("Could not read current gripper angle before setting enable=%s.", enabled)
        self._send_gripper_ctrl(gripper_pos_raw, enabled)

    @staticmethod
    def _message_timestamp(message: Any) -> float:
        return float(getattr(message, "time_stamp", 0.0) or 0.0)

    def set_manual_control(self, enabled: bool) -> None:
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if enabled == self._manual_control_enabled:
            return

        self._manual_control_enabled = None
        if enabled:
            set_piper_role(self.arm, PIPER_ROLE_LEADER)
        else:
            set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
            self._send_command_mode()
            if not wait_enable_piper(self.arm, self.config.enable_timeout_s):
                raise RuntimeError(
                    f"[{self.config.port}] Piper leader did not enable after switching to follower role."
                )
            if self.config.sync_gripper:
                self._set_gripper_enabled(True)

        self._manual_control_enabled = enabled

    def configure(self) -> None:
        self.set_manual_control(self.config.manual_control)

    def _read_joint_from_ctrl(self) -> dict[str, float] | None:
        joint_ctrl_msg = self.arm.GetArmJointCtrl()
        if getattr(joint_ctrl_msg, "time_stamp", 0.0) <= 0:
            return None
        joint_ctrl = getattr(joint_ctrl_msg, "joint_ctrl", None)
        if joint_ctrl is None:
            return None
        return {
            f"{joint_name}.pos": milli_to_unit(getattr(joint_ctrl, joint_name, 0))
            for joint_name in PIPER_JOINT_NAMES
        }

    def _read_joint_from_feedback(self) -> dict[str, float] | None:
        joint_msg = self.arm.GetArmJointMsgs()
        if self._message_timestamp(joint_msg) <= 0:
            return None
        joint_state = getattr(joint_msg, "joint_state", None)
        if joint_state is None:
            return None
        return {
            f"{joint_name}.pos": milli_to_unit(getattr(joint_state, joint_name, 0))
            for joint_name in PIPER_JOINT_NAMES
        }

    def _read_gripper_from_ctrl(self) -> float | None:
        gripper_ctrl_msg = self.arm.GetArmGripperCtrl()
        if getattr(gripper_ctrl_msg, "time_stamp", 0.0) <= 0:
            return None
        gripper_ctrl = getattr(gripper_ctrl_msg, "gripper_ctrl", None)
        if gripper_ctrl is None:
            return None
        return abs(milli_to_unit(getattr(gripper_ctrl, "grippers_angle", 0)))

    def _read_gripper_from_feedback(self) -> float | None:
        gripper_msg = self.arm.GetArmGripperMsgs()
        if self._message_timestamp(gripper_msg) <= 0:
            return None
        gripper_state = getattr(gripper_msg, "gripper_state", None)
        if gripper_state is None:
            return None
        return abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))

    def _try_read_raw_action(self) -> RobotAction | None:
        if self._manual_control_enabled is True:
            action = self._read_joint_from_ctrl()
            gripper_pos = self._read_gripper_from_ctrl() if self.config.sync_gripper else None
        elif self._manual_control_enabled is False:
            action = self._read_joint_from_feedback()
            gripper_pos = self._read_gripper_from_feedback() if self.config.sync_gripper else None
        else:
            raise RuntimeError(f"[{self.config.port}] Piper leader control mode is unknown.")

        if action is None:
            return None
        if self.config.sync_gripper:
            if gripper_pos is None:
                return None
            action["gripper.pos"] = gripper_pos
        else:
            action["gripper.pos"] = 0.0
        return action

    def _read_raw_action(self) -> RobotAction:
        action = self._try_read_raw_action()
        return action if action is not None else {}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self._read_raw_action()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.set_manual_control(False)
        self._refresh_command_mode_if_needed()

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in feedback for key in joint_keys)
        if has_all_joints:
            joint_targets = [feedback[key] for key in joint_keys]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)

        if self.config.sync_gripper and "gripper.pos" in feedback:
            gripper_pos_raw = unit_to_milli(feedback["gripper.pos"])
            self._send_gripper_ctrl(gripper_pos_raw, enabled=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.set_manual_control(True)
            if self.config.disable_on_disconnect:
                self.arm.DisableArm(7)
        finally:
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            logger.info("%s disconnected.", self)


class PiperXLeader(PiperLeader):
    config_class = PiperXLeaderConfig
    name = "piperx_leader"
