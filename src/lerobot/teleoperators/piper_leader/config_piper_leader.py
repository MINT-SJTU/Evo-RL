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

from dataclasses import dataclass

from lerobot.utils.piper_sdk import PIPER_CALIBRATION_MODES

from ..config import TeleoperatorConfig


@dataclass
class PiperLeaderConfigBase:
    """Configuration for a Piper leader arm used as teleoperator."""

    # CAN interface name (e.g. "can1")
    port: str

    # Piper SDK connection options
    judge_flag: bool = False
    can_auto_init: bool = True
    log_level: str = "WARNING"
    startup_sleep_s: float = 0.1

    # Optional role command on startup (0xFC = motion output arm)
    set_leader_mode_on_connect: bool = False

    # Manual backdrivable mode for human teleop
    manual_control: bool = True

    # Read control messages from leader first, fallback to feedback state if missing
    prefer_ctrl_messages: bool = True
    fallback_to_feedback: bool = True

    # Gripper handling
    sync_gripper: bool = True
    gripper_effort_default: int = 1000
    gripper_status_code: int = 0x01

    # Command mode for send_feedback
    command_speed_ratio: int = 100
    command_high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    enable_timeout_s: float = 3.0

    # Calibration precision:
    # homing_offset/range_min/range_max are stored as "degree * calibration_scale".
    calibration_scale: int = 1000
    # Calibration policy:
    # - required: always require calibration
    # - off: never require calibration
    calibration_mode: str = "required"

    # Safety behavior on disconnect
    disable_on_disconnect: bool = False


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig, PiperLeaderConfigBase):
    def __post_init__(self):
        if not (0 <= self.command_speed_ratio <= 100):
            raise ValueError("`command_speed_ratio` must be between 0 and 100.")
        if self.mode_refresh_interval_s < 0:
            raise ValueError("`mode_refresh_interval_s` must be >= 0.")
        if self.enable_timeout_s < 0:
            raise ValueError("`enable_timeout_s` must be >= 0.")
        if self.calibration_scale <= 0:
            raise ValueError("`calibration_scale` must be > 0.")
        if self.calibration_mode not in PIPER_CALIBRATION_MODES:
            raise ValueError("`calibration_mode` must be one of: required, off.")
        if self.startup_sleep_s < 0:
            raise ValueError("`startup_sleep_s` must be >= 0.")
        if not (0 <= self.gripper_effort_default <= 5000):
            raise ValueError("`gripper_effort_default` must be between 0 and 5000.")
        if self.gripper_status_code not in {0x00, 0x01, 0x02, 0x03}:
            raise ValueError("`gripper_status_code` must be one of 0x00, 0x01, 0x02, 0x03.")
