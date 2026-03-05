#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import lru_cache
from typing import Any

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)
PIPER_CALIBRATION_MODES = {"required", "off"}


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `piper_sdk`. Install Evo-RL dependencies first (for example: `pip install -e .`)."
        ) from exc


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc


def should_require_piper_calibration(calibration_mode: str) -> bool:
    if calibration_mode == "required":
        return True
    if calibration_mode == "off":
        return False
    raise ValueError(
        f"Invalid Piper calibration mode '{calibration_mode}'. "
        "Expected one of: required, off."
    )
