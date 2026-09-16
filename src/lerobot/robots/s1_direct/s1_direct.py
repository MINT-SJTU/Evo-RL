#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
S1Direct — 单臂一体式夹爪机器人

在 connect/disconnect 时维护共享 arm 实例注册，供 S1DirectTeleop 复用。
"""

import logging
import time
import numpy as np
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.teleoperators.s1_direct_teleop.s1_direct_teleop import (
    is_manual_mode,
    register_shared_arm,
    unregister_shared_arm,
)
from lerobot.teleoperators.bi_s1_direct_teleop.bi_s1_direct_teleop import (
    is_manual_mode as is_bi_manual_mode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_s1_direct import S1DirectConfig

logger = logging.getLogger(__name__)

# S1 有 6 个关节（达妙电机, id 1-6）
S1_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
S1_JOINT_ACTION_KEYS = tuple(f"{name}.pos" for name in S1_JOINT_NAMES)
S1_ACTION_KEYS = S1_JOINT_ACTION_KEYS + ("gripper.pos",)
S1_ACTION_KEYS_NO_GRIPPER = S1_JOINT_ACTION_KEYS


class S1Direct(Robot):
    """单臂 Theseus S1 一体式夹爪驱动。

    使用独立名称注册 ``s1_direct``，并在 connect/disconnect 时
    维护共享 arm 实例注册。
    """

    config_class = S1DirectConfig
    name = "s1_direct"

    def __init__(self, config: S1DirectConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.arm = None

        self.cameras = make_cameras_from_configs(config.cameras)
        self._has_gripper = config.end_effector == "mix"
        self._action_keys = S1_ACTION_KEYS if self._has_gripper else S1_ACTION_KEYS_NO_GRIPPER

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**dict.fromkeys(self._action_keys, float), **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._action_keys, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from S1_SDK import S1_arm, control_mode

        logger.info("正在连接 S1 一体式臂: dev=%s, version=%s, end_effector=%s",
                     self.config.dev, self.config.version, self.config.end_effector)

        self.arm = S1_arm(
            mode=control_mode.only_real,
            dev=self.config.dev,
            end_effector=self.config.end_effector,
            # 复合式夹爪 mix 以及其它 S1 末端在 Evo-RL 侧统一关闭碰撞检测。
            check_collision=False,
            arm_version=self.config.version,
        )

        self._is_connected = True
        connected_cameras = []
        try:
            self.configure()

            if self.config.enable_on_connect:
                self.arm.enable()
                logger.info("S1 一体式臂电机已使能")

            for cam in self.cameras.values():
                cam.connect()
                connected_cameras.append(cam)

            # 注册共享 arm 实例
            if self.config.dev:
                register_shared_arm(self.config.dev, self.arm)

        except Exception:
            self._safe_close_arm()
            for cam in connected_cameras:
                cam.disconnect()
            self._is_connected = False
            raise

        logger.info("%s 已连接", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("S1 不需要框架层校准")

    def configure(self) -> None:
        pass

    def _read_observation(self) -> dict[str, float]:
        pos = self.arm.get_pos()
        obs: dict[str, float] = {}
        for i, name in enumerate(S1_JOINT_NAMES):
            obs[f"{name}.pos"] = float(pos[i])
        if self._has_gripper and len(pos) > 6:
            obs["gripper.pos"] = float(pos[6])
        elif self._has_gripper:
            obs["gripper.pos"] = 0.0
        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = self._read_observation()
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent_action: dict[str, float] = {}

        # 如果处于人工拖拽模式，为了减轻拖动手感，跳过位置指令的下发。
        # 此时 Teleop 端会维持 gravity() 模式，Robot 端只负责数据记录。
        # 注意：此处不仅检查单臂注册表，也同时检查双臂注册表。
        if self.config.dev and (is_manual_mode(self.config.dev) or is_bi_manual_mode(self.config.dev)):
            # logger.debug("S1Direct: Skipping send_action (manual mode active for dev=%s)", self.config.dev)
            for key in self._action_keys:
                if key in action:
                    sent_action[key] = action[key]
            return sent_action

        joint_keys = S1_JOINT_ACTION_KEYS
        has_all_joints = all(key in action for key in joint_keys)
        if has_all_joints:
            joint_pos = [action[key] for key in joint_keys]
            self.arm.joint_control(joint_pos)
            # time.sleep(0.1)
            for key in joint_keys:
                sent_action[key] = action[key]
        elif any(key in action for key in joint_keys):
            logger.debug("忽略不完整的关节指令，需要全部 6 个关节的值")

        if self._has_gripper and "gripper.pos" in action:
            gripper_pos = action["gripper.pos"]
            self.arm.control_mix(gripper_pos)
            sent_action["gripper.pos"] = gripper_pos

        return sent_action

    def _safe_close_arm(self) -> None:
        if self.arm is not None:
            try:
                if self.config.disable_on_disconnect:
                    logger.info("S1 一体式臂正在执行平滑归零保护动作...")
                    
                    # 预热读取以避免获取到初始化的 stale 零位数据导致瞬间归零（通常在刚连接即断开时发生）
                    for _ in range(10):
                        pos = self.arm.get_pos()
                        time.sleep(0.02)
                    
                    pos = self.arm.get_pos()
                    current_joints = np.array(pos[:6], dtype=np.float32)
                    target_joints = np.zeros(6, dtype=np.float32)
                    
                    # 准备夹爪轨迹（如果存在且已启用）
                    has_gripper = self._has_gripper and len(pos) > 6
                    if has_gripper:
                        current_gripper = pos[6]
                        target_gripper = 0.0
                        gripper_trajectory = np.linspace(current_gripper, target_gripper, 100)

                    trajectory = np.linspace(current_joints, target_joints, 100)
                    for i, joints in enumerate(trajectory):
                        self.arm.joint_control(joints.tolist())
                        if has_gripper:
                            self.arm.control_mix(float(gripper_trajectory[i]))
                        time.sleep(0.02)
                    self.arm.disable()
                self.arm.close()
            except Exception:
                logger.warning("关闭 S1 一体式臂时出错，已忽略", exc_info=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            # 先注销共享实例
            if self.config.dev:
                unregister_shared_arm(self.config.dev)
            self._safe_close_arm()
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            self._is_connected = False
            self.arm = None
            logger.info("%s 已断开", self)
