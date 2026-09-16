#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from lerobot.robots.s1_direct.config_s1_direct import S1DirectConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_s1_direct_teleop")
@dataclass
class BiS1DirectTeleopConfig(TeleoperatorConfig):
    """双臂一体式夹爪 teleop 配置 —— 复用 direct 臂作为示教输入。

    实际硬件与 BiS1DirectConfig 相同（同一对从臂），这里仅作为
    Teleoperator 端的 draccus 配置入口。真正的硬件实例在 connect()
    时从已连接的 BiS1Direct robot 中借用。
    """

    left_arm_config: S1DirectConfigBase = None
    right_arm_config: S1DirectConfigBase = None

    # S1 不需要框架层校准
    require_calibration: bool = False
