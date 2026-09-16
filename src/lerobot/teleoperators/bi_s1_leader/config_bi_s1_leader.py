#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from lerobot.teleoperators.s1_leader import S1LeaderConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_s1_leader")
@dataclass
class BiS1LeaderConfig(TeleoperatorConfig):
    """双臂 Theseus S1 主臂（遥操作端）配置"""

    left_arm_config: S1LeaderConfigBase = None
    right_arm_config: S1LeaderConfigBase = None

    # S1 不需要框架层校准
    require_calibration: bool = False
