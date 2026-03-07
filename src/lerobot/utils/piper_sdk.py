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

import re
import time
from dataclasses import dataclass
from enum import Enum
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
PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06
_CTRL_MODE_HEX_RE = re.compile(r"0x([0-9a-fA-F]+)")


@dataclass(frozen=True)
class PiperCtrlMode:
    code: int | None
    text: str


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


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    while time.monotonic() < deadline:
        if bool(arm.EnablePiper()):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(interval_s, remaining_s))
    return False


def parse_piper_ctrl_mode(value: Any) -> PiperCtrlMode:
    if value is None:
        return PiperCtrlMode(code=None, text="None")

    if isinstance(value, bool):
        code = int(value)
        return PiperCtrlMode(code=code, text=f"0x{code:02X}")

    if isinstance(value, int):
        return PiperCtrlMode(code=value, text=f"0x{value:02X}")

    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, int):
            return PiperCtrlMode(code=enum_value, text=str(value))
        if isinstance(enum_value, str):
            m = _CTRL_MODE_HEX_RE.search(enum_value)
            if m:
                return PiperCtrlMode(code=int(m.group(1), 16), text=str(value))
            if enum_value.isdigit():
                return PiperCtrlMode(code=int(enum_value), text=str(value))
        return PiperCtrlMode(code=None, text=str(value))

    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, int):
        return PiperCtrlMode(code=raw_value, text=str(value))

    if isinstance(value, str):
        m = _CTRL_MODE_HEX_RE.search(value)
        if m:
            return PiperCtrlMode(code=int(m.group(1), 16), text=value)
        if value.isdigit():
            code = int(value)
            return PiperCtrlMode(code=code, text=f"0x{code:02X}")
        return PiperCtrlMode(code=None, text=value)

    return PiperCtrlMode(code=None, text=repr(value))


def read_piper_ctrl_mode(arm: Any, timeout_s: float = 1.0, poll_s: float = 0.02) -> PiperCtrlMode | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        status_msg = arm.GetArmStatus()
        if getattr(status_msg, "time_stamp", 0.0) > 0.0:
            arm_status = getattr(status_msg, "arm_status", None)
            if arm_status is not None:
                mode = parse_piper_ctrl_mode(getattr(arm_status, "ctrl_mode", None))
                if mode.code is not None:
                    return mode
        time.sleep(max(0.005, poll_s))
    return None


def guard_piper_ctrl_mode_on_connect(
    arm: Any,
    *,
    interface_name: str,
    timeout_s: float = 0.5,
    poll_s: float = 0.02,
    settle_s: float = 0.05,
) -> None:
    mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
    if mode is None:
        raise RuntimeError(
            f"[{interface_name}] could not read arm ctrl_mode within {timeout_s:.2f}s. "
            "Check CAN wiring/power and rerun."
        )

    if mode.code is None:
        raise RuntimeError(
            f"[{interface_name}] got unreadable arm ctrl_mode value: {mode.text}. "
            "Check SDK version/firmware and rerun."
        )

    if mode.code in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
        arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        if settle_s > 0:
            time.sleep(settle_s)
        raise RuntimeError(
            f"[{interface_name}] arm is in master/teaching role (ctrl_mode={mode.text}). "
            "Follower role command (0xFC) has been sent. Power-cycle this arm, then retry."
        )
