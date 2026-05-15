#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BiS1Direct — 双臂一体式夹爪机器人。"""

import logging
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.s1_direct import S1Direct, S1DirectConfig
from lerobot.robots.s1_direct.config_s1_direct import S1DirectConfigBase
from lerobot.teleoperators.s1_direct_teleop.s1_direct_teleop import (
    is_manual_mode,
    unregister_shared_arm as unregister_single_direct_shared_arm,
)
from lerobot.teleoperators.bi_s1_direct_teleop.bi_s1_direct_teleop import (
    is_manual_mode as is_bi_manual_mode,
    register_shared_arm,
    unregister_shared_arm,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_s1_direct import BiS1DirectConfig

logger = logging.getLogger(__name__)


class BiS1Direct(Robot):
    """双臂 Theseus S1 一体式夹爪驱动，组合两个 S1Direct 实例。"""

    config_class = BiS1DirectConfig
    name = "bi_s1_direct"

    # S1DirectConfigBase 中需要按臂提取的字段名
    _side_field_names = (
        "dev",
        "end_effector",
        "version",
        "enable_on_connect",
        "disable_on_disconnect",
        "cameras",
        "require_calibration",
    )

    def _build_arm_config(self, side_cfg: S1DirectConfigBase, side: str) -> S1DirectConfig:
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        kwargs["calibration_dir"] = self.config.calibration_dir
        return S1DirectConfig(**kwargs)

    def __init__(self, config: BiS1DirectConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = self._build_arm_config(config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(config.right_arm_config, "right")

        self.left_arm = S1Direct(left_arm_config)
        self.right_arm = S1Direct(right_arm_config)

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
                "BiS1Direct received duplicate camera names from left and right arms: "
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

        # 双臂 direct 场景仅使用 bi_s1_direct_teleop 的共享注册表。
        if self.left_arm.config.dev:
            unregister_single_direct_shared_arm(self.left_arm.config.dev)
        if self.right_arm.config.dev:
            unregister_single_direct_shared_arm(self.right_arm.config.dev)

        # 将 arm 实例注册到共享注册中心，供 BiS1DirectTeleop 复用
        if self.left_arm.config.dev:
            register_shared_arm(self.left_arm.config.dev, self.left_arm.arm)
        if self.right_arm.config.dev:
            register_shared_arm(self.right_arm.config.dev, self.right_arm.arm)

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
                "BiS1Direct received duplicate observation keys from left and right arms: "
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
        # 如果处于人工拖拽模式，为了减轻拖动手感，跳过位置指令的下发。
        # 此时 Teleop 端会维持 gravity() 模式，Robot 端只负责记录动作数据。
        # 同时检查单臂和双臂注册表，确保无论何种录制模式都能正确识别 manual 状态
        left_manual = self.left_arm.config.dev and (is_manual_mode(self.left_arm.config.dev) or is_bi_manual_mode(self.left_arm.config.dev))
        right_manual = self.right_arm.config.dev and (is_manual_mode(self.right_arm.config.dev) or is_bi_manual_mode(self.right_arm.config.dev))

        if left_manual or right_manual:
            logger.debug("BiS1Direct: Intervention detected (manual mode), skipping position commands for arms. (L=%s, R=%s)", left_manual, right_manual)

        left_action: RobotAction = {}
        right_action: RobotAction = {}
        for key, value in action.items():
            if key.startswith("left_"):
                left_action[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_action[key.removeprefix("right_")] = value

        # 注意：这里如果 left_manual 为真，我们直接跳过 self.left_arm.send_action 的调用，避免触发其内部的位置控制逻辑。
        # 即使 S1Direct 内部也有 is_manual_mode 检查，此处拦截更加彻底且避免了因注册字典（single vs bi）不一致导致的失效。
        sent_action_left = self.left_arm.send_action(left_action) if not left_manual else left_action
        sent_action_right = self.right_arm.send_action(right_action) if not right_manual else right_action

        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}
        return {**prefixed_sent_action_left, **prefixed_sent_action_right}

    @check_if_not_connected
    def disconnect(self):
        # 先注销共享实例
        if self.left_arm.config.dev:
            unregister_shared_arm(self.left_arm.config.dev)
        if self.right_arm.config.dev:
            unregister_shared_arm(self.right_arm.config.dev)

        self.left_arm.disconnect()
        self.right_arm.disconnect()
