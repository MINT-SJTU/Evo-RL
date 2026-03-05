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

from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    PIPER_JOINT_NAMES,
    get_piper_sdk,
    milli_to_unit,
    parse_piper_log_level,
    should_require_piper_calibration,
    unit_to_milli,
)
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)
PIPER_CALIB_KEYS = list(PIPER_ACTION_KEYS)
PIPER_CALIB_IDS = {key: idx for idx, key in enumerate(PIPER_CALIB_KEYS)}


class PiperLeader(Teleoperator):
    """Piper leader arm used as a teleoperator through Piper SDK CAN messages."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
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
        return {key: float for key in PIPER_ACTION_KEYS}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {key: float for key in PIPER_ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm.ConnectPort()
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)

        self._is_connected = True
        # Recompute control mode on every fresh connection.
        self._manual_control_enabled = None
        try:
            if self.config.set_leader_mode_on_connect:
                # NOTE:
                # - Piper teaching-input mode (0xFA) does not accept external JointCtrl/GripperCtrl commands.
                # - For policy-sync and human-in-the-loop switching, the leader must remain commandable,
                #   so we configure it as motion-output mode (0xFC), same as the follower side.
                # - Mode-role changes (e.g., 0xFA <-> 0xFC) may require a full power-cycle to take effect.
                self.arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
                time.sleep(0.05)
            self.configure()
            if not self.is_calibrated and calibrate and should_require_piper_calibration(self.config.calibration_mode):
                logger.info("No piper-leader calibration file found for '%s'. Running lerobot-calibrate flow.", self.id)
                self.calibrate()
        except Exception:
            self.arm.DisconnectPort()
            self._is_connected = False
            raise

        logger.info("%s connected.", self)

    def _use_uncalibrated_passthrough(self) -> bool:
        return not self.is_calibrated and not should_require_piper_calibration(self.config.calibration_mode)

    @property
    def is_calibrated(self) -> bool:
        if not all(key in self.calibration for key in PIPER_CALIB_KEYS):
            return False
        for key in PIPER_CALIB_KEYS:
            cal = self.calibration[key]
            if cal.range_max <= cal.range_min:
                return False
        return True

    def calibrate(self) -> None:
        if self.calibration and self.is_calibrated:
            user_input = input(
                f"Press ENTER to use existing calibration file for id '{self.id}', "
                "or type 'c' and press ENTER to run a new calibration: "
            )
            if user_input.strip().lower() != "c":
                return

        logger.info("Running calibration for %s", self)
        input("Move piper-leader to your desired neutral/center pose, then press ENTER...")
        neutral = self._read_raw_action()
        print("Move all piper-leader joints through full range. Press ENTER to stop recording...")
        range_mins, range_maxes = self._record_ranges_of_motion()

        self.calibration = {}
        for key in PIPER_CALIB_KEYS:
            min_deg = range_mins[key]
            max_deg = range_maxes[key]
            if max_deg <= min_deg:
                raise ValueError(f"Invalid range for {key}: min={min_deg:.3f}, max={max_deg:.3f}")

            neutral_deg = min(max_deg, max(min_deg, neutral[key]))
            self.calibration[key] = MotorCalibration(
                id=PIPER_CALIB_IDS[key],
                drive_mode=0,
                homing_offset=self._to_calibration_units(neutral_deg),
                range_min=self._to_calibration_units(min_deg),
                range_max=self._to_calibration_units(max_deg),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

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

    def _wait_enable(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if bool(self.arm.EnablePiper()):
                return True
            time.sleep(0.02)
        return False

    def set_manual_control(self, enabled: bool) -> None:
        if not self._is_connected:
            return
        if enabled and self._manual_control_enabled is not True:
            self.arm.DisableArm(7)
            self._manual_control_enabled = True
            return
        if not enabled and self._manual_control_enabled is not False:
            self._send_command_mode()
            if not self._wait_enable(self.config.enable_timeout_s):
                logger.warning("Piper leader did not report enabled state before timeout.")
            self._manual_control_enabled = False

    def configure(self) -> None:
        self.set_manual_control(self.config.manual_control)

    def _read_joint_from_ctrl(self) -> dict[str, float] | None:
        joint_ctrl_msg = self.arm.GetArmJointCtrl()
        if getattr(joint_ctrl_msg, "time_stamp", 0.0) <= 0:
            return None
        joint_ctrl = getattr(joint_ctrl_msg, "joint_ctrl", None)
        return {
            f"{joint_name}.pos": milli_to_unit(getattr(joint_ctrl, joint_name, 0))
            for joint_name in PIPER_JOINT_NAMES
        }

    def _read_joint_from_feedback(self) -> dict[str, float] | None:
        joint_msg = self.arm.GetArmJointMsgs()
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
        return abs(milli_to_unit(getattr(gripper_ctrl, "grippers_angle", 0)))

    def _read_gripper_from_feedback(self) -> float | None:
        gripper_msg = self.arm.GetArmGripperMsgs()
        gripper_state = getattr(gripper_msg, "gripper_state", None)
        if gripper_state is None:
            return None
        return abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))

    def _read_raw_action(self) -> RobotAction:
        used_feedback_for_joints = False
        action: dict[str, float] | None = None
        if self.config.prefer_ctrl_messages:
            action = self._read_joint_from_ctrl()

        if action is None and self.config.fallback_to_feedback:
            action = self._read_joint_from_feedback()
            used_feedback_for_joints = action is not None

        if action is None:
            action = {f"{joint_name}.pos": 0.0 for joint_name in PIPER_JOINT_NAMES}

        use_ctrl_for_gripper = self.config.prefer_ctrl_messages and not used_feedback_for_joints
        gripper_pos = self._read_gripper_from_ctrl() if use_ctrl_for_gripper else None
        if gripper_pos is None and self.config.fallback_to_feedback:
            gripper_pos = self._read_gripper_from_feedback()
        action["gripper.pos"] = 0.0 if gripper_pos is None else gripper_pos
        return action

    def _to_calibration_units(self, angle_deg: float) -> int:
        return int(round(angle_deg * self.config.calibration_scale))

    def _from_calibration_units(self, value: int) -> float:
        return float(value) / float(self.config.calibration_scale)

    def _calibrated_to_offset(self, key: str, raw_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        bounded = min(max_deg, max(min_deg, raw_deg))
        centered = bounded - home_deg
        return -centered if cal.drive_mode else centered

    def _offset_to_calibrated(self, key: str, offset_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        centered = -offset_deg if cal.drive_mode else offset_deg
        target = home_deg + centered
        return min(max_deg, max(min_deg, target))

    def _record_ranges_of_motion(self) -> tuple[dict[str, float], dict[str, float]]:
        current = self._read_raw_action()
        mins = current.copy()
        maxes = current.copy()

        while True:
            current = self._read_raw_action()
            mins = {key: min(mins[key], current[key]) for key in PIPER_CALIB_KEYS}
            maxes = {key: max(maxes[key], current[key]) for key in PIPER_CALIB_KEYS}

            print("\n-----------------------------")
            print("JOINT       |    MIN |    POS |    MAX")
            for key in PIPER_CALIB_KEYS:
                print(f"{key:<11} | {mins[key]:>6.2f} | {current[key]:>6.2f} | {maxes[key]:>6.2f}")

            if enter_pressed():
                break
            move_cursor_up(len(PIPER_CALIB_KEYS) + 3)

        return mins, maxes

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if not self.is_calibrated and not self._use_uncalibrated_passthrough():
            raise RuntimeError(
                f"{self} is not calibrated. Run `lerobot-calibrate --teleop.type=piper_leader --teleop.id={self.id}` first."
            )

        raw_action = self._read_raw_action()
        if self._use_uncalibrated_passthrough():
            action: RobotAction = dict(raw_action)
            if not self.config.sync_gripper:
                action["gripper.pos"] = 0.0
            return action

        action: RobotAction = {
            key: self._calibrated_to_offset(key, raw_action[key]) for key in PIPER_CALIB_KEYS
        }
        if not self.config.sync_gripper:
            action["gripper.pos"] = 0.0
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_calibrated and not self._use_uncalibrated_passthrough():
            raise RuntimeError(
                f"{self} is not calibrated. Run `lerobot-calibrate --teleop.type=piper_leader --teleop.id={self.id}` first."
            )

        self.set_manual_control(False)
        self._refresh_command_mode_if_needed()

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in feedback for key in joint_keys)
        if has_all_joints:
            if self._use_uncalibrated_passthrough():
                joint_targets = [feedback[key] for key in joint_keys]
            else:
                joint_targets = [self._offset_to_calibrated(key, feedback[key]) for key in joint_keys]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)

        if self.config.sync_gripper and "gripper.pos" in feedback:
            if self._use_uncalibrated_passthrough():
                gripper_target = feedback["gripper.pos"]
            else:
                gripper_target = self._offset_to_calibrated("gripper.pos", feedback["gripper.pos"])
            gripper_pos_raw = unit_to_milli(gripper_target)
            self.arm.GripperCtrl(
                gripper_pos_raw,
                self.config.gripper_effort_default,
                self.config.gripper_status_code,
                0x00,
            )

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self.arm.DisableArm(7)
        finally:
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            logger.info("%s disconnected.", self)
