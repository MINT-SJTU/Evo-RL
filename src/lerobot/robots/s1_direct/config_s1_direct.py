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


def _validate_s1_direct_config(config: "S1DirectConfigBase") -> None:
    if config.end_effector != "mix":
        raise ValueError(f"`end_effector` 仅支持 'mix', 当前为 '{config.end_effector}'")
    if config.version not in ("V1", "V2"):
        raise ValueError(f"`version` 必须是 'V1' 或 'V2', 当前为 '{config.version}'")


@dataclass
class S1DirectConfigBase:
    """Theseus S1 direct 配置基类（供嵌套 arm 配置复用）。"""

    # S1_SDK 连接参数 —— 直接传递给 S1_arm()
    dev: str | None = None

    # 末端执行器类型: "mix"(复合式夹爪)
    end_effector: str = "mix"

    # 通信版本: "V1" 为 CAN 通信, "V2" 为 UART 串口通信
    version: str = "V2"

    # 连接后是否自动使能电机
    enable_on_connect: bool = True

    # 断开时是否自动失能电机
    disable_on_disconnect: bool = True

    # 可选挂载的相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # S1 自带零位机制，不需要框架层校准
    require_calibration: bool = False


@RobotConfig.register_subclass("s1_direct")
@dataclass
class S1DirectConfig(RobotConfig, S1DirectConfigBase):
    """单臂 Theseus S1 一体式夹爪配置。"""

    def __post_init__(self):
        super().__post_init__()
        _validate_s1_direct_config(self)
