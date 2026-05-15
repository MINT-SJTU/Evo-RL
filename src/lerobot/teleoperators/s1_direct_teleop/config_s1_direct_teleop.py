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


@TeleoperatorConfig.register_subclass("s1_direct_teleop")
@dataclass
class S1DirectTeleopConfig(TeleoperatorConfig):
    """单臂一体式夹爪 teleop 配置 —— 复用 follower 臂作为示教输入。

    实际硬件与 S1DirectConfig 相同（同一条臂），这里仅作为
    Teleoperator 端的 draccus 配置入口。
    """

    # 设备路径：必须与 robot 侧一致
    dev: str | None = None

    # 末端执行器类型（direct 仅支持 mix）
    end_effector: str = "mix"

    # S1 不需要框架层校准
    require_calibration: bool = False
