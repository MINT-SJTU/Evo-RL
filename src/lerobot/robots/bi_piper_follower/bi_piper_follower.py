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

from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.piper_follower import (
    PiperFollower,
    PiperFollowerConfig,
    PiperXFollower,
    PiperXFollowerConfig,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_piper_follower import BiPiperFollowerConfig, BiPiperXFollowerConfig


class BiPiperFollower(Robot):
    """Bimanual PiPER/PiPER-X follower arms."""

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "speed_ratio",
        "high_follow",
        "mode_refresh_interval_s",
        "enable_on_connect",
        "enable_timeout_s",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "cameras",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, arm_config_cls, side_cfg, side: str):
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        return arm_config_cls(**kwargs)

    def __init__(self, config: BiPiperFollowerConfig | BiPiperXFollowerConfig):
        self.robot_type = self.name
        self.id = config.id
        self.config = config

        if config.type == "bi_piperx_follower":
            arm_config_cls = PiperXFollowerConfig
            arm_cls = PiperXFollower
        else:
            arm_config_cls = PiperFollowerConfig
            arm_cls = PiperFollower

        left_arm_config = self._build_arm_config(arm_config_cls, config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(arm_config_cls, config.right_arm_config, "right")

        self.left_arm = arm_cls(left_arm_config)
        self.right_arm = arm_cls(right_arm_config)

        # Only for compatibility with other parts of the codebase that expect `robot.cameras`.
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = self.left_arm._motors_ft
        right_arm_motors_ft = self.right_arm._motors_ft
        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        left_arm_cameras_ft = self.left_arm._cameras_ft
        right_arm_cameras_ft = self.right_arm._cameras_ft
        return {
            **{f"left_{k}": v for k, v in left_arm_cameras_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_cameras_ft.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.left_arm.connect()
        self.right_arm.connect()

    def set_teleop_send_only_mode(self, enabled: bool) -> None:
        self.left_arm.set_teleop_send_only_mode(enabled)
        self.right_arm.set_teleop_send_only_mode(enabled)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action: RobotAction = {}
        right_action: RobotAction = {}
        for key, value in action.items():
            if key.startswith("left_"):
                left_action[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_action[key.removeprefix("right_")] = value

        sent_action_left = self.left_arm.send_action(left_action)
        sent_action_right = self.right_arm.send_action(right_action)

        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}
        return {**prefixed_sent_action_left, **prefixed_sent_action_right}

    @check_if_not_connected
    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()


class BiPiperXFollower(BiPiperFollower):
    config_class = BiPiperXFollowerConfig
    name = "bi_piperx_follower"
