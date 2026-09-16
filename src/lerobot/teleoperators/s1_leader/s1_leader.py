#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import time
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_s1_leader import S1LeaderConfig

logger = logging.getLogger(__name__)

# 关节命名必须和从臂完全一致
S1_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
S1_JOINT_ACTION_KEYS = tuple(f"{name}.pos" for name in S1_JOINT_NAMES)
S1_ACTION_KEYS = S1_JOINT_ACTION_KEYS + ("gripper.pos",)


class S1Leader(Teleoperator):
    """Theseus S1 主臂驱动，用作遥操作端，通过 S1_SDK 读取位置。"""

    config_class = S1LeaderConfig
    name = "s1_leader"

    def __init__(self, config: S1LeaderConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.arm = None

        # 主臂 "teach" 模式下，id=8 的电机作为示教输入，映射到夹爪
        self._has_gripper = config.end_effector == "teach"
        self._manual_control = True  # 默认人工拖拽

    @cached_property
    def action_features(self) -> dict[str, type]:
        """主臂输出的动作空间（和从臂一致）"""
        keys = S1_ACTION_KEYS if self._has_gripper else S1_JOINT_ACTION_KEYS
        return dict.fromkeys(keys, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """主臂接收的反馈空间（暂未使用）"""
        keys = S1_ACTION_KEYS if self._has_gripper else S1_JOINT_ACTION_KEYS
        return dict.fromkeys(keys, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """连接主臂：创建 S1_arm 实例 → 使能电机"""
        from S1_SDK import S1_arm, control_mode

        logger.info("正在连接 S1 主臂: dev=%s, version=%s, end_effector=%s",
                     self.config.dev, self.config.version, self.config.end_effector)

        self.arm = S1_arm(
            mode=control_mode.only_real,
            dev=self.config.dev,
            end_effector=self.config.end_effector,
            check_collision=False,
            arm_version=self.config.version,
        )

        self._is_connected = True
        try:
            self.configure()
            # 主臂使能后，示教模式可以提供扭矩反馈
            self.arm.enable()
            logger.info("S1 主臂电机已使能")
        except Exception:
            self._safe_close_arm()
            self._is_connected = False
            raise

        logger.info("%s 已连接", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """S1 主臂不需要框架层校准"""
        logger.info("S1 主臂不需要框架层校准")

    def configure(self) -> None:
        """运行时配置（S1 主臂不需要额外配置）"""
        pass

    def set_manual_control(self, enabled: bool) -> None:
        """切换人工/策略控制模式。
        
        - True:  人工示教模式 — get_action() 调重力补偿，人可自由拖拽
        - False: 策略控制模式 — send_feedback() 下发关节指令让臂跟随策略
        """
        self._manual_control = enabled
        logger.info("%s manual_control = %s", self, enabled)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """读取主臂当前关节位置，作为遥操作的目标动作
        
        人手动移动主臂 → 读取位置 → 传给从臂执行
        """
        pos = self.arm.get_pos()
        
        if self._manual_control:
            # SDK 内部会将重力补偿下发给关节电机
            self.arm.gravity()
            if self._has_gripper:
                self.arm.control_teach_zero_tau()

        action: dict[str, float] = {}
        for i, name in enumerate(S1_JOINT_NAMES):
            action[f"{name}.pos"] = float(pos[i])

        if self._has_gripper and len(pos) > 6:
            # "teach" 模式下 pos[6] 是示教电机位置 → 映射到夹爪
            action["gripper.pos"] = float(pos[6])
        elif self._has_gripper:
            action["gripper.pos"] = 0.0

        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈到主臂
        
        在人机协同录制时，如果由策略控制或需主臂跟随从臂移动，这里接收关节位置，
        主臂执行对应的关节控制走到目标位置，保证主从臂姿态一致。
        """
        if self._manual_control:
            # 人工模式下跳过位置控制，避免干扰 drag 动作
            return

        has_all_joints = all(key in feedback for key in S1_JOINT_ACTION_KEYS)
        if has_all_joints:
            joint_commands = [feedback[key] for key in S1_JOINT_ACTION_KEYS]
            # 这里调用 S1_SDK 的位置控制命令让主臂运动到 feedback 的位置
            self.arm.joint_control(joint_commands)
            
        # 示教模式（teach）下的主臂也支持夹爪位置同步。
        # 这里接收关节反馈位置并下发，保证主从臂姿态一致。
        if self._has_gripper and "gripper.pos" in feedback:
            gripper_pos = feedback["gripper.pos"]
            self.arm.control_teach_pos(gripper_pos)

    def _safe_close_arm(self) -> None:
        """安全关闭主臂，忽略异常"""
        if self.arm is not None:
            try:
                # 结束遥控时，缓慢回到零位，防止突然失去扭矩坠落
                logger.info("S1 主臂正在执行平滑归零保护动作...")
                
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
                        self.arm.control_teach_pos(float(gripper_trajectory[i]))
                    time.sleep(0.02)
                self.arm.disable()
                self.arm.close()
            except Exception:
                logger.warning("关闭 S1 主臂时出错，已忽略", exc_info=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        """断开主臂：失能电机 → 关闭通信"""
        try:
            self._safe_close_arm()
        finally:
            self._is_connected = False
            self.arm = None
            logger.info("%s 已断开", self)
