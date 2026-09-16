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
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.s1_leader import S1Leader, S1LeaderConfig, S1LeaderConfigBase
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_s1_leader import BiS1LeaderConfig

logger = logging.getLogger(__name__)


class BiS1Leader(Teleoperator):
    """双臂 Theseus S1 主臂驱动，组合两个 S1Leader 实例。"""

    config_class = BiS1LeaderConfig
    name = "bi_s1_leader"

    # S1LeaderConfigBase 中需要按臂提取的字段名
    _side_field_names = (
        "dev",
        "end_effector",
        "version",
        "require_calibration",
    )

    def _build_arm_config(self, side_cfg: S1LeaderConfigBase, side: str) -> S1LeaderConfig:
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        kwargs["calibration_dir"] = self.config.calibration_dir
        return S1LeaderConfig(**kwargs)

    def __init__(self, config: BiS1LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = self._build_arm_config(config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(config.right_arm_config, "right")

        self.left_arm = S1Leader(left_arm_config)
        self.right_arm = S1Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features
        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.feedback_features
        right_arm_features = self.right_arm.feedback_features
        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

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

    def set_manual_control(self, enabled: bool) -> None:
        """切换双臂的主/被动控制模式。"""
        self.left_arm.set_manual_control(enabled)
        self.right_arm.set_manual_control(enabled)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict: RobotAction = {}
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})
        return action_dict

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        left_feedback: dict[str, Any] = {}
        right_feedback: dict[str, Any] = {}
        for key, value in feedback.items():
            if key.startswith("left_"):
                left_feedback[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_feedback[key.removeprefix("right_")] = value
        self.left_arm.send_feedback(left_feedback)
        self.right_arm.send_feedback(right_feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
