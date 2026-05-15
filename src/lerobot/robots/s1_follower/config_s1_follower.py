#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


def _validate_s1_follower_config(config: "S1FollowerConfigBase") -> None:
    if config.end_effector not in ("None", "gripper", "teach", "mix"):
        raise ValueError(
            f"`end_effector` 必须是 'None', 'gripper', 'teach', 'mix' 之一, 当前为 '{config.end_effector}'"
        )
    if config.version not in ("V1", "V2"):
        raise ValueError(f"`version` 必须是 'V1' 或 'V2', 当前为 '{config.version}'")


@dataclass
class S1FollowerConfigBase:
    """Theseus S1 从臂配置（基类，不注册到 RobotConfig，供嵌套使用）"""

    # S1_SDK 连接参数 —— 直接传递给 S1_arm()
    # 设备路径：CAN 接口名 (如 "can0") 或串口设备 (如 "/dev/ttyUSB3")
    dev: str | None = None

    # 末端执行器类型: "None"(无), "gripper"(夹爪), "teach"(示教)
    end_effector: str = "gripper"

    # 通信版本: "V1" 为 CAN 通信, "V2" 为 UART 串口通信
    version: str = "V2"

    # 连接后是否自动使能电机
    enable_on_connect: bool = True

    # 断开时是否自动失能电机（安全措施：机械臂会掉力）
    disable_on_disconnect: bool = True

    # 可选挂载的相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # S1 自带零位机制，不需要框架层校准
    require_calibration: bool = False


@RobotConfig.register_subclass("s1_follower")
@dataclass
class S1FollowerConfig(RobotConfig, S1FollowerConfigBase):
    """Theseus S1 从臂配置"""

    def __post_init__(self):
        super().__post_init__()
        _validate_s1_follower_config(self)
