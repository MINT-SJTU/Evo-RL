#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
BiS1DirectTeleop — 双臂一体式夹爪 teleoperator

核心思路
--------
一体式夹爪方案中两条从臂同时充当主臂角色。此 Teleoperator **不创建**
任何新的 S1_arm 实例，而是通过 ``_BI_S1_DIRECT_SHARED_ARMS`` 字典
复用同一 robot (``BiS1Direct``) 已经建立好的硬件连接。

共享流程
--------
1. ``BiS1Direct.connect()`` → 创建两条 S1Direct，将 arm 对象注册到
   ``_BI_S1_DIRECT_SHARED_ARMS[dev]``。
2. ``BiS1DirectTeleop.connect()`` → 通过 dev 字符串从
   ``_BI_S1_DIRECT_SHARED_ARMS`` 中查找对应的 arm 对象。
3. ``get_action()``：读取当前位置并调用 ``arm.gravity()`` 提供重力补偿。
4. ``send_feedback()``：在策略控制期间，将关节指令下发让臂跟随策略。

set_manual_control(enabled)
---------------------------
- ``True``  → 人工介入模式：只调 ``gravity()``，人可以自由拖拽
- ``False`` → 策略控制模式：``send_feedback`` 时调 ``joint_control``

由于单臂 S1Direct 也调用 ``joint_control``（在 ``send_action`` 中），
且 ``get_action`` 和 ``send_action`` 在同一个录制循环中交替被调用，
所以不会出现两边同时向底层写控制命令的竞争情况。
"""

import logging
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_s1_direct_teleop import BiS1DirectTeleopConfig

logger = logging.getLogger(__name__)

# 关节命名与 s1_direct / s1_leader 一致
S1_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
S1_JOINT_ACTION_KEYS = tuple(f"{name}.pos" for name in S1_JOINT_NAMES)
S1_ACTION_KEYS = S1_JOINT_ACTION_KEYS + ("gripper.pos",)

# ---------------------  共享 arm 实例注册中心  ---------------------
# key = dev 字符串 (如 "/dev/ttyUSB0"), value = {"arm": S1_arm, "manual_mode": bool}
_BI_S1_DIRECT_SHARED_ARMS: dict[str, dict[str, Any]] = {}


def register_shared_arm(dev: str, arm: Any) -> None:
    """注册一个 S1_arm 实例供 teleop 端共享（由 BiS1Direct.connect 调用）。"""
    _BI_S1_DIRECT_SHARED_ARMS[dev] = {"arm": arm, "manual_mode": True}
    logger.debug("Shared arm registered: dev=%s", dev)


def unregister_shared_arm(dev: str) -> None:
    """注销共享的 S1_arm 实例（由 BiS1Direct.disconnect 调用）。"""
    _BI_S1_DIRECT_SHARED_ARMS.pop(dev, None)
    logger.debug("Shared arm unregistered: dev=%s", dev)


def is_manual_mode(dev: str) -> bool:
    """查询指定端口的臂是否处于人工模式。"""
    state = _BI_S1_DIRECT_SHARED_ARMS.get(dev)
    return state["manual_mode"] if state else False


class _DirectArmProxy:
    """对单臂共享实例的轻量代理，一个 proxy 对应一条臂。"""

    def __init__(self, dev: str, has_gripper: bool):
        self.dev = dev
        self.has_gripper = has_gripper
        self.arm: Any | None = None  # connect 时绑定
        self._manual_control = True  # 默认人工拖拽

    def connect(self) -> None:
        state = _BI_S1_DIRECT_SHARED_ARMS.get(self.dev)
        if state is None:
            raise RuntimeError(
                f"BiS1DirectTeleop: 未找到 dev={self.dev} 的共享 arm 实例。"
                f"请确保 BiS1Direct robot 已先于 teleop 连接。"
                f"当前已注册: {list(_BI_S1_DIRECT_SHARED_ARMS.keys())}"
            )
        self.arm = state["arm"]
        logger.info("Direct teleop proxy 绑定到 dev=%s", self.dev)

    @property
    def is_connected(self) -> bool:
        return self.arm is not None

    def set_manual_control(self, enabled: bool) -> None:
        self._manual_control = enabled

    def get_action(self) -> RobotAction:
        pos = self.arm.get_pos()
        if self._manual_control:
            # SDK 内部会将重力补偿下发给关节电机
            self.arm.gravity()
            if self.has_gripper:
                self.arm.control_mix_zero_tau()

        action: RobotAction = {}
        if len(pos) >= 6:
            for i, key in enumerate(S1_JOINT_ACTION_KEYS):
                action[key] = float(pos[i])
        
        if self.has_gripper and len(pos) > 6:
            action["gripper.pos"] = float(pos[6])
        elif self.has_gripper:
            action["gripper.pos"] = 0.0
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """direct 模式下不消费 policy feedback。"""
        logger.debug("Direct teleop proxy ignores feedback in direct teleop mode (dev=%s)", self.dev)

    def disconnect(self) -> None:
        self.arm = None


class BiS1DirectTeleop(Teleoperator):
    """双臂一体式夹爪 teleoperator —— 复用 BiS1Direct 的 S1Direct 臂。

    使用方式::

        --teleop.type=bi_s1_direct_teleop
        --teleop.left_arm_config.dev=/dev/ttyUSB0
        --teleop.left_arm_config.end_effector=mix
        --teleop.right_arm_config.dev=/dev/ttyUSB1
        --teleop.right_arm_config.end_effector=mix

    dev 必须与 robot 侧相同，以便查找到已注册的共享 arm。
    """

    config_class = BiS1DirectTeleopConfig
    name = "bi_s1_direct_teleop"

    def __init__(self, config: BiS1DirectTeleopConfig):
        super().__init__(config)
        self.config = config
        if config.left_arm_config.end_effector != "mix":
            raise ValueError(
                f"`left_arm_config.end_effector` 仅支持 'mix', 当前为 '{config.left_arm_config.end_effector}'"
            )
        if config.right_arm_config.end_effector != "mix":
            raise ValueError(
                f"`right_arm_config.end_effector` 仅支持 'mix', 当前为 '{config.right_arm_config.end_effector}'"
            )

        left_has_gripper = config.left_arm_config.end_effector == "mix"
        right_has_gripper = config.right_arm_config.end_effector == "mix"

        self._left_proxy = _DirectArmProxy(
            dev=config.left_arm_config.dev,
            has_gripper=left_has_gripper,
        )
        self._right_proxy = _DirectArmProxy(
            dev=config.right_arm_config.dev,
            has_gripper=right_has_gripper,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_keys = S1_ACTION_KEYS if self._left_proxy.has_gripper else S1_JOINT_ACTION_KEYS
        right_keys = S1_ACTION_KEYS if self._right_proxy.has_gripper else S1_JOINT_ACTION_KEYS
        return {
            **{f"left_{k}": float for k in left_keys},
            **{f"right_{k}": float for k in right_keys},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._left_proxy.is_connected and self._right_proxy.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._left_proxy.connect()
        self._right_proxy.connect()
        logger.info("%s 已连接 (共享模式)", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("BiS1DirectTeleop 不需要校准 (共享模式)")

    def configure(self) -> None:
        pass

    def set_manual_control(self, enabled: bool) -> None:
        """切换人工/策略控制模式。

        - True:  人工示教模式 — get_action() 调重力补偿，人可自由拖拽
        - False: 策略控制模式 — send_feedback() 下发关节指令让臂跟随策略
        """
        self._left_proxy.set_manual_control(enabled)
        self._right_proxy.set_manual_control(enabled)

        # 同步到共享状态
        for proxy in (self._left_proxy, self._right_proxy):
            state = _BI_S1_DIRECT_SHARED_ARMS.get(proxy.dev)
            if state:
                state["manual_mode"] = enabled

        logger.info("Direct teleop manual_control = %s", enabled)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict: RobotAction = {}
        left_action = self._left_proxy.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})
        right_action = self._right_proxy.get_action()
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
        self._left_proxy.send_feedback(left_feedback)
        self._right_proxy.send_feedback(right_feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self._left_proxy.disconnect()
        self._right_proxy.disconnect()
        logger.info("%s 已断开 (共享模式)", self)
