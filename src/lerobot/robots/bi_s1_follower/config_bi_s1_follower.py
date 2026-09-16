#!/usr/bin/env python

# Copyright 2026 Theseus S1 integration for Evo-RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from lerobot.robots.s1_follower import S1FollowerConfigBase

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_s1_follower")
@dataclass
class BiS1FollowerConfig(RobotConfig):
    """双臂 Theseus S1 从臂配置"""

    left_arm_config: S1FollowerConfigBase = None
    right_arm_config: S1FollowerConfigBase = None

    # S1 不需要框架层校准
    require_calibration: bool = False
