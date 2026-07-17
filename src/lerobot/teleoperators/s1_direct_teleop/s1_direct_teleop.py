#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
S1DirectTeleop — 单臂一体式夹爪 teleoperator

与双臂版 ``BiS1DirectTeleop`` 同理，此 Teleoperator 不创建新的
S1_arm 实例，而是通过 ``_S1_DIRECT_SHARED_ARMS`` 字典复用
``S1Direct`` robot 已建立好的硬件连接。

set_manual_control(enabled)
---------------------------
- ``True``  → 人工示教模式：get_action() 调 gravity()，人可自由拖拽
- ``False`` → 策略控制模式：send_feedback() 调 joint_control() 及 control_mix()
"""

import logging
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_s1_direct_teleop import S1DirectTeleopConfig

logger = logging.getLogger(__name__)

# 关节命名与 s1_follower / s1_leader 一致
S1_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
S1_JOINT_ACTION_KEYS = tuple(f"{name}.pos" for name in S1_JOINT_NAMES)
S1_ACTION_KEYS = S1_JOINT_ACTION_KEYS + ("gripper.pos",)

# ---------------------  共享 arm 实例注册中心  ---------------------
# key = dev 字符串 (如 "/dev/ttyUSB0"), value = {"arm": S1_arm, "manual_mode": bool}
_S1_DIRECT_SHARED_ARMS: dict[str, dict[str, Any]] = {}


def register_shared_arm(dev: str, arm: Any) -> None:
    """注册一个 S1_arm 实例供 teleop 端共享（由 S1Direct.connect 调用）。"""
    _S1_DIRECT_SHARED_ARMS[dev] = {"arm": arm, "manual_mode": True}
    logger.debug("Shared arm registered (single): dev=%s", dev)


def unregister_shared_arm(dev: str) -> None:
    """注销共享的 S1_arm 实例（由 S1Direct.disconnect 调用）。"""
    _S1_DIRECT_SHARED_ARMS.pop(dev, None)
    logger.debug("Shared arm unregistered (single): dev=%s", dev)


def is_manual_mode(dev: str) -> bool:
    """查询指定端口的臂是否处于人工模式。"""
    state = _S1_DIRECT_SHARED_ARMS.get(dev)
    return state["manual_mode"] if state else False


class S1DirectTeleop(Teleoperator):
    """单臂一体式夹爪 teleoperator —— 复用 S1Direct 的 S1_arm 实例。

    使用方式::

        --teleop.type=s1_direct_teleop
        --teleop.dev=/dev/ttyUSB0
        --teleop.end_effector=mix

    dev 必须与 robot 侧相同，以便查找到已注册的共享 arm。
    """

    config_class = S1DirectTeleopConfig
    name = "s1_direct_teleop"

    def __init__(self, config: S1DirectTeleopConfig):
        super().__init__(config)
        self.config = config
        if config.end_effector != "mix":
            raise ValueError(f"`end_effector` 仅支持 'mix', 当前为 '{config.end_effector}'")
        self._has_gripper = config.end_effector == "mix"
        self._arm: Any | None = None
        self._manual_control = True  # 默认人工拖拽

    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = S1_ACTION_KEYS if self._has_gripper else S1_JOINT_ACTION_KEYS
        return dict.fromkeys(keys, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        state = _S1_DIRECT_SHARED_ARMS.get(self.config.dev)
        if state is None:
            raise RuntimeError(
                f"S1DirectTeleop: 未找到 dev={self.config.dev} 的共享 arm 实例。"
                f"请确保 S1Direct robot 已先于 teleop 连接。"
                f"当前已注册: {list(_S1_DIRECT_SHARED_ARMS.keys())}"
            )
        self._arm = state["arm"]
        logger.info("%s 已连接 (共享模式, dev=%s)", self, self.config.dev)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("S1DirectTeleop 不需要校准 (共享模式)")

    def configure(self) -> None:
        pass

    def set_manual_control(self, enabled: bool) -> None:
        """切换人工/策略控制模式。

        - True:  人工示教模式 — get_action() 调重力补偿，人可自由拖拽
        - False: 策略控制模式 — send_feedback() 下发关节指令让臂跟随策略
        """
        self._manual_control = enabled
        state = _S1_DIRECT_SHARED_ARMS.get(self.config.dev)
        if state:
            state["manual_mode"] = enabled
        logger.info("%s manual_control = %s", self, enabled)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pos = self._arm.get_pos()
        if self._manual_control:
            self._arm.gravity()
            if self._has_gripper:
                self._arm.control_mix_zero_tau()

        action: RobotAction = {}
        # 记录关节位置（弧度）
        if len(pos) >= 6:
            for i, key in enumerate(S1_JOINT_ACTION_KEYS):
                action[key] = float(pos[i])
        
        if self._has_gripper and len(pos) > 6:
            action["gripper.pos"] = float(pos[6])
        elif self._has_gripper:
            action["gripper.pos"] = 0.0
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """direct 模式下不消费 policy feedback。"""
        logger.debug("%s ignores feedback in direct teleop mode", self)
    
    @check_if_not_connected
    def disconnect(self) -> None:
        self._arm = None
        logger.info("%s 已断开 (共享模式)", self)
