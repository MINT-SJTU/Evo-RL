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
    resolve_piper_can_port,
    set_piper_role,
    unit_to_milli,
    wait_enable_piper,
)

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig, PiperXLeaderConfig

logger = logging.getLogger(__name__)
PIPER_ACTION_READ_TIMEOUT_S = 5.0
PIPER_ACTION_READ_POLL_S = 0.01
PIPER_FEEDBACK_SEED_TIMEOUT_S = 1.0
PIPER_LEADER_JOINT_FRAME_IDS = (0x155, 0x156, 0x157)
PIPER_LEADER_GRIPPER_FRAME_ID = 0x159
PIPER_LEADER_CAN_DRAIN_LIMIT = 64


class PiperLeader(Teleoperator):
    """PiPER teleoperator with runtime role switching for S-V1.8-9 or newer firmware."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig | PiperXLeaderConfig):
        self.id = config.id
        self.config = config
        self._is_connected = False
        self._manual_control_enabled: bool | None = None
        self._last_mode_refresh_t = 0.0
        self._leader_joint_timestamp_before_switch = 0.0
        self._leader_gripper_timestamp_before_switch = 0.0
        self._leader_frames_ready = False
        self._leader_bus: Any | None = None
        self._raw_leader_action: RobotAction = {}
        self._raw_leader_joint_frames_seen: set[int] = set()
        self._raw_leader_gripper_seen = False
        self.port = resolve_piper_can_port(self.config.port)

        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=self.port,
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
        self.arm.ConnectPort(piper_init=False)
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)

        self._is_connected = True
        self._manual_control_enabled = None
        self._leader_frames_ready = False
        try:
            self.configure()
        except Exception:
            try:
                self.set_manual_control(True)
            except Exception:
                logger.exception("Failed to restore %s to leader role after connect error.", self)
                self._disable_after_role_restore_failure()
            finally:
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

    def _wait_enable(self, timeout_s: float) -> bool:
        return wait_enable_piper(self.arm, timeout_s)

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

    def _safe_message_timestamp(self, getter: Any) -> float:
        try:
            return self._message_timestamp(getter())
        except Exception:
            logger.debug("Could not read PiPER control-frame timestamp before role switch.", exc_info=True)
            return 0.0

    def _disable_after_role_restore_failure(self) -> None:
        try:
            self.arm.DisableArm(7)
        except Exception:
            logger.exception("Failed to disable %s after leader-role restore failure.", self)

    def _enter_manual_role(self) -> None:
        self._reset_raw_leader_action()
        if self.config.seed_manual_action_from_feedback:
            self._seed_raw_leader_action_from_feedback()
        self._leader_joint_timestamp_before_switch = self._safe_message_timestamp(self.arm.GetArmJointCtrl)
        self._leader_gripper_timestamp_before_switch = (
            self._safe_message_timestamp(self.arm.GetArmGripperCtrl) if self.config.sync_gripper else 0.0
        )
        self._leader_frames_ready = False
        self._drain_leader_bus()
        set_piper_role(self.arm, PIPER_ROLE_LEADER)
        self._manual_control_enabled = True

    def set_manual_control(self, enabled: bool) -> None:
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if enabled == self._manual_control_enabled:
            return

        self._manual_control_enabled = None
        if enabled:
            self._enter_manual_role()
        else:
            try:
                set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
                self._send_command_mode()
                if not self._wait_enable(self.config.enable_timeout_s):
                    raise RuntimeError(
                        f"[{self.port}] Piper leader did not enable after switching to follower role."
                    )
                if self.config.sync_gripper:
                    self._set_gripper_enabled(True)
                self._manual_control_enabled = False
            except Exception:
                try:
                    self._enter_manual_role()
                except Exception:
                    self._manual_control_enabled = None
                    logger.exception("Failed to restore %s to leader role after policy-switch error.", self)
                raise

    def configure(self) -> None:
        self.set_manual_control(self.config.manual_control)

    def _read_joint_from_ctrl(self, joint_ctrl_msg: Any | None = None) -> dict[str, float] | None:
        if joint_ctrl_msg is None:
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

    def _read_gripper_from_ctrl(self, gripper_ctrl_msg: Any | None = None) -> float | None:
        if gripper_ctrl_msg is None:
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

    def _reset_raw_leader_action(self) -> None:
        self._raw_leader_action = {}
        self._raw_leader_joint_frames_seen = set()
        self._raw_leader_gripper_seen = False

    def _set_raw_leader_action(self, action: RobotAction) -> None:
        self._raw_leader_action = dict(action)
        self._raw_leader_joint_frames_seen = set(PIPER_LEADER_JOINT_FRAME_IDS)
        self._raw_leader_gripper_seen = "gripper.pos" in action

    def _ensure_leader_bus(self) -> Any:
        if self._leader_bus is None:
            import can

            self._leader_bus = can.interface.Bus(channel=self.port, interface="socketcan", bitrate=1000000)
        return self._leader_bus

    def _drain_leader_bus(self) -> None:
        if self._leader_bus is None:
            return
        for _ in range(PIPER_LEADER_CAN_DRAIN_LIMIT):
            if self._leader_bus.recv(timeout=0.0) is None:
                break

    @staticmethod
    def _decode_int32(data: bytes | bytearray | memoryview, offset: int) -> int:
        return int.from_bytes(data[offset : offset + 4], byteorder="big", signed=True)

    def _update_raw_leader_action(self, arbitration_id: int, data: bytes | bytearray | memoryview) -> None:
        if arbitration_id in PIPER_LEADER_JOINT_FRAME_IDS and len(data) >= 8:
            start_joint_idx = (arbitration_id - PIPER_LEADER_JOINT_FRAME_IDS[0]) * 2
            for offset, joint_name in zip((0, 4), PIPER_JOINT_NAMES[start_joint_idx : start_joint_idx + 2], strict=True):
                self._raw_leader_action[f"{joint_name}.pos"] = milli_to_unit(self._decode_int32(data, offset))
            self._raw_leader_joint_frames_seen.add(arbitration_id)
        elif arbitration_id == PIPER_LEADER_GRIPPER_FRAME_ID and len(data) >= 4:
            self._raw_leader_action["gripper.pos"] = abs(milli_to_unit(self._decode_int32(data, 0)))
            self._raw_leader_gripper_seen = True

    def _try_read_raw_leader_can_action(self) -> RobotAction | None:
        bus = self._ensure_leader_bus()
        for _ in range(PIPER_LEADER_CAN_DRAIN_LIMIT):
            msg = bus.recv(timeout=0.0)
            if msg is None:
                break
            self._update_raw_leader_action(msg.arbitration_id, msg.data)

        has_joints = self._raw_leader_joint_frames_seen.issuperset(PIPER_LEADER_JOINT_FRAME_IDS)
        has_gripper = not self.config.sync_gripper or self._raw_leader_gripper_seen
        if not (has_joints and has_gripper):
            return None
        if not self.config.sync_gripper:
            self._raw_leader_action["gripper.pos"] = 0.0
        return dict(self._raw_leader_action)

    def _seed_raw_leader_action_from_feedback(self) -> None:
        set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
        deadline = time.monotonic() + PIPER_FEEDBACK_SEED_TIMEOUT_S
        while time.monotonic() < deadline:
            action = self._read_joint_from_feedback()
            gripper_pos = self._read_gripper_from_feedback() if self.config.sync_gripper else None
            if action is not None and (not self.config.sync_gripper or gripper_pos is not None):
                action["gripper.pos"] = gripper_pos if gripper_pos is not None else 0.0
                self._set_raw_leader_action(action)
                return
            time.sleep(PIPER_ACTION_READ_POLL_S)
        logger.debug("Could not seed %s manual action from feedback before entering leader role.", self)

    def _try_read_raw_action(self) -> RobotAction | None:
        if self._manual_control_enabled is True:
            joint_msg = self.arm.GetArmJointCtrl()
            joint_timestamp = self._message_timestamp(joint_msg)
            gripper_msg = self.arm.GetArmGripperCtrl() if self.config.sync_gripper else None
            gripper_timestamp = self._message_timestamp(gripper_msg) if gripper_msg is not None else 0.0
            if not self._leader_frames_ready:
                joint_hz = float(getattr(joint_msg, "Hz", 0.0) or 0.0)
                has_fresh_joints = (
                    joint_timestamp > self._leader_joint_timestamp_before_switch and joint_hz > 0.0
                )
                has_fresh_gripper = (
                    not self.config.sync_gripper
                    or gripper_timestamp > self._leader_gripper_timestamp_before_switch
                )
                if not (has_fresh_joints and has_fresh_gripper):
                    return self._try_read_raw_leader_can_action()
                self._leader_frames_ready = True
            action = self._read_joint_from_ctrl(joint_msg)
            gripper_pos = self._read_gripper_from_ctrl(gripper_msg) if gripper_msg is not None else None
            if action is None or (self.config.sync_gripper and gripper_pos is None):
                return self._try_read_raw_leader_can_action()
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
        deadline = time.monotonic() + PIPER_ACTION_READ_TIMEOUT_S
        while True:
            action = self._try_read_raw_action()
            if action is not None:
                return action
            if time.monotonic() >= deadline:
                source = "leader control" if self._manual_control_enabled else "feedback"
                raise RuntimeError(
                    f"[{self.port}] no complete Piper {source} frame received within "
                    f"{PIPER_ACTION_READ_TIMEOUT_S:.1f}s."
                )
            time.sleep(PIPER_ACTION_READ_POLL_S)

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
            try:
                self.set_manual_control(True)
            except Exception:
                self._disable_after_role_restore_failure()
                raise
            else:
                if self.config.disable_on_disconnect:
                    self.arm.DisableArm(7)
        finally:
            if self._leader_bus is not None:
                self._leader_bus.shutdown()
                self._leader_bus = None
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            logger.info("%s disconnected.", self)


class PiperXLeader(PiperLeader):
    config_class = PiperXLeaderConfig
    name = "piperx_leader"
