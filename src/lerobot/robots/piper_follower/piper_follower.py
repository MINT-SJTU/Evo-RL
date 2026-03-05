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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
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

from ..robot import Robot
from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)
PIPER_CALIB_KEYS = list(PIPER_ACTION_KEYS)
PIPER_CALIB_IDS = {key: idx for idx, key in enumerate(PIPER_CALIB_KEYS)}


class PiperFollower(Robot):
    """Piper follower arm controlled through the Piper SDK (CAN)."""

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._last_mode_refresh_t = 0.0

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

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**{key: float for key in PIPER_ACTION_KEYS}, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in PIPER_ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm.ConnectPort()
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)

        self._is_connected = True
        connected_cameras = []
        try:
            if self.config.set_follower_mode_on_connect:
                self.arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
                time.sleep(0.05)

            self.configure()
            # lerobot-calibrate calls connect(calibrate=False) before running calibrate().
            # Keep the arm in drag mode for calibration by skipping auto-enable in this path.
            should_enable = self.config.enable_on_connect and calibrate
            if should_enable and not self._wait_enable(self.config.enable_timeout_s):
                logger.warning("Piper follower did not report enabled state before timeout.")
            if not self.is_calibrated and calibrate and should_require_piper_calibration(self.config.calibration_mode):
                logger.info("No piper-follower calibration file found for '%s'. Running lerobot-calibrate flow.", self.id)
                self.calibrate()

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
        # Calibration for Piper follower should be sampled in pure backdrivable mode.
        self.arm.DisableArm(7)
        time.sleep(0.1)
        input("Move piper-follower to your desired neutral/center pose, then press ENTER...")
        neutral = self._read_raw_observation()
        print("Move all piper-follower joints through full range. Press ENTER to stop recording...")
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

    def configure(self) -> None:
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
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if bool(self.arm.EnablePiper()):
                return True
            time.sleep(0.02)
        return False

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

    def _to_calibration_units(self, angle_deg: float) -> int:
        return int(round(angle_deg * self.config.calibration_scale))

    def _from_calibration_units(self, value: int) -> float:
        return float(value) / float(self.config.calibration_scale)

    def _offset_to_target(self, key: str, offset_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        centered = -offset_deg if cal.drive_mode else offset_deg
        target = home_deg + centered
        return min(max_deg, max(min_deg, target))

    def _record_ranges_of_motion(self) -> tuple[dict[str, float], dict[str, float]]:
        current = self._read_raw_observation()
        mins = current.copy()
        maxes = current.copy()

        while True:
            current = self._read_raw_observation()
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
    def get_observation(self) -> RobotObservation:
        obs: dict[str, float] = self._read_raw_observation()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_calibrated and not self._use_uncalibrated_passthrough():
            raise RuntimeError(
                f"{self} is not calibrated. Run `lerobot-calibrate --robot.type=piper_follower --robot.id={self.id}` first."
            )

        self._refresh_motion_mode_if_needed()

        sent_action: dict[str, float] = {}

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in action for key in joint_keys)
        if has_all_joints:
            if self._use_uncalibrated_passthrough():
                joint_targets = [action[key] for key in joint_keys]
            else:
                joint_targets = [self._offset_to_target(key, action[key]) for key in joint_keys]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)
            sent_action.update({key: milli_to_unit(raw) for key, raw in zip(joint_keys, joint_commands, strict=True)})
        elif any(key in action for key in joint_keys):
            logger.debug("Ignoring partial Piper joint action. Need all six joint keys to send command.")

        if self.config.sync_gripper and "gripper.pos" in action:
            if self._use_uncalibrated_passthrough():
                gripper_target = action["gripper.pos"]
            else:
                gripper_target = self._offset_to_target("gripper.pos", action["gripper.pos"])
            gripper_pos_raw = unit_to_milli(gripper_target)
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
                cam.disconnect()
            self._is_connected = False
            logger.info("%s disconnected.", self)
