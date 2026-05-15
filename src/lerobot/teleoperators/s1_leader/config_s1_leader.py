#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from ..config import TeleoperatorConfig


def _validate_s1_leader_config(config: "S1LeaderConfigBase") -> None:
    if config.end_effector not in ("None", "gripper", "teach"):
        raise ValueError(
            f"`end_effector` 必须是 'None', 'gripper', 'teach' 之一, 当前为 '{config.end_effector}'"
        )
    if config.version not in ("V1", "V2"):
        raise ValueError(f"`version` 必须是 'V1' 或 'V2', 当前为 '{config.version}'")


@dataclass
class S1LeaderConfigBase:
    """Theseus S1 主臂（遥操作端）配置（基类，不注册到 TeleoperatorConfig，供嵌套使用）"""

    # S1_SDK 连接参数 —— 直接传递给 S1_arm()
    # 主臂使用 CAN 接口 (如 "can0")
    dev: str | None = None

    # 末端执行器类型：主臂通常为 "teach"（示教模式）
    end_effector: str = "teach"

    # 通信版本: "V1" 为 CAN 通信（主臂）, "V2" 为 UART 串口通信（从臂）
    version: str = "V1"

    # S1 不需要框架层校准
    require_calibration: bool = False


@TeleoperatorConfig.register_subclass("s1_leader")
@dataclass
class S1LeaderConfig(TeleoperatorConfig, S1LeaderConfigBase):
    """Theseus S1 主臂（遥操作端）配置"""

    def __post_init__(self):
        _validate_s1_leader_config(self)
