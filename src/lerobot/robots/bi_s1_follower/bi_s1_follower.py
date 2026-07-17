#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.s1_follower import S1Follower, S1FollowerConfig, S1FollowerConfigBase
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_s1_follower import BiS1FollowerConfig

logger = logging.getLogger(__name__)


class BiS1Follower(Robot):
    """双臂 Theseus S1 从臂驱动，组合两个 S1Follower 实例。"""

    config_class = BiS1FollowerConfig
    name = "bi_s1_follower"

    # S1FollowerConfigBase 中需要按臂提取的字段名
    _side_field_names = (
        "dev",
        "end_effector",
        "version",
        "enable_on_connect",
        "disable_on_disconnect",
        "cameras",
        "require_calibration",
    )

    def _build_arm_config(self, side_cfg: S1FollowerConfigBase, side: str) -> S1FollowerConfig:
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        kwargs["calibration_dir"] = self.config.calibration_dir
        return S1FollowerConfig(**kwargs)

    def __init__(self, config: BiS1FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = self._build_arm_config(config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(config.right_arm_config, "right")

        self.left_arm = S1Follower(left_arm_config)
        self.right_arm = S1Follower(right_arm_config)

        # 兼容框架中其他地方对 robot.cameras 的引用
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = dict.fromkeys(self.left_arm._action_keys, float)
        right_arm_motors_ft = dict.fromkeys(self.right_arm._action_keys, float)
        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        left_arm_cameras_ft = self.left_arm._cameras_ft
        right_arm_cameras_ft = self.right_arm._cameras_ft
        overlap = left_arm_cameras_ft.keys() & right_arm_cameras_ft.keys()
        if overlap:
            raise ValueError(
                "BiS1Follower received duplicate camera names from left and right arms: "
                f"{sorted(overlap)}. Rename the cameras explicitly in the arm configs."
            )
        return {**left_arm_cameras_ft, **right_arm_cameras_ft}

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
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}
        left_camera_keys = set(self.left_arm._cameras_ft)
        right_camera_keys = set(self.right_arm._cameras_ft)
        camera_overlap = left_camera_keys & right_camera_keys
        if camera_overlap:
            raise ValueError(
                "BiS1Follower received duplicate observation keys from left and right arms: "
                f"{sorted(camera_overlap)}. Rename the cameras explicitly in the arm configs."
            )

        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()

        obs_dict.update({f"left_{key}": value for key, value in left_obs.items() if key not in left_camera_keys})
        obs_dict.update({key: value for key, value in left_obs.items() if key in left_camera_keys})
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items() if key not in right_camera_keys})
        obs_dict.update({key: value for key, value in right_obs.items() if key in right_camera_keys})
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
