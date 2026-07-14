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

    # Initial role: True uses hardware leader drag (0xFA); False accepts policy commands (0xFC).
    manual_control: bool = True

    # Gripper handling
    sync_gripper: bool = True
    gripper_effort_default: int = 1000
    gripper_status_code: int = 0x01
    gripper_teaching_range_per: int = 100
    gripper_max_range_config: int = 100
    gripper_teaching_friction: int = 10

    # Command mode for send_feedback
    command_speed_ratio: int = 100
    command_high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    enable_timeout_s: float = 3.0
    seed_manual_action_from_feedback: bool = False

    # Safety behavior on disconnect
    disable_on_disconnect: bool = False


def _validate_piper_leader_config(config: PiperLeaderConfigBase) -> None:
    if not (0 <= config.command_speed_ratio <= 100):
        raise ValueError("`command_speed_ratio` must be between 0 and 100.")
    if config.mode_refresh_interval_s < 0:
        raise ValueError("`mode_refresh_interval_s` must be >= 0.")
    if config.enable_timeout_s < 0:
        raise ValueError("`enable_timeout_s` must be >= 0.")
    if config.startup_sleep_s < 0:
        raise ValueError("`startup_sleep_s` must be >= 0.")
    if not (0 <= config.gripper_effort_default <= 5000):
        raise ValueError("`gripper_effort_default` must be between 0 and 5000.")
    if config.gripper_status_code not in {0x00, 0x01, 0x02, 0x03}:
        raise ValueError("`gripper_status_code` must be one of 0x00, 0x01, 0x02, 0x03.")
    if not (100 <= config.gripper_teaching_range_per <= 200):
        raise ValueError("`gripper_teaching_range_per` must be between 100 and 200.")
    if config.gripper_max_range_config not in {0, 70, 100}:
        raise ValueError("`gripper_max_range_config` must be one of 0, 70, or 100.")
    if not (1 <= config.gripper_teaching_friction <= 10):
        raise ValueError("`gripper_teaching_friction` must be between 1 and 10.")


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig, PiperLeaderConfigBase):
    def __post_init__(self):
        _validate_piper_leader_config(self)


@dataclass
class PiperXLeaderConfigBase(PiperLeaderConfigBase):
    pass


@TeleoperatorConfig.register_subclass("piperx_leader")
@dataclass
class PiperXLeaderConfig(TeleoperatorConfig, PiperXLeaderConfigBase):
    def __post_init__(self):
        _validate_piper_leader_config(self)
