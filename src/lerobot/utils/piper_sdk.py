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

import subprocess
import time
from functools import lru_cache
from pathlib import Path
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
PIPER_ROLE_LEADER = 0xFA
PIPER_ROLE_FOLLOWER = 0xFC
PIPER_ROLE_SWITCH_SETTLE_S = 0.2
SYS_CLASS_NET = Path("/sys/class/net")


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


def _read_udev_properties(interface: str) -> dict[str, str]:
    interface_path = SYS_CLASS_NET / interface
    try:
        result = subprocess.run(
            ["udevadm", "info", "-q", "property", "-p", str(interface_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if result.returncode != 0:
        return {}

    properties = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            properties[key] = value
    return properties


def list_piper_can_interfaces() -> dict[str, str]:
    """Return available socketcan interfaces keyed by stable USB-CAN identifiers."""
    if not SYS_CLASS_NET.exists():
        return {}

    interfaces: dict[str, str] = {}
    for interface_path in sorted(SYS_CLASS_NET.glob("can*")):
        interface = interface_path.name
        properties = _read_udev_properties(interface)
        for key in ("ID_SERIAL_SHORT", "ID_SERIAL"):
            value = properties.get(key)
            if value:
                interfaces[value] = interface
    return interfaces


def resolve_piper_can_port(port: str) -> str:
    """Resolve a socketcan interface name or USB-CAN serial into the current canN name."""
    port = port.strip()
    if not port:
        raise ValueError("Piper CAN port must not be empty.")

    if (SYS_CLASS_NET / port).exists():
        return port

    interfaces_by_serial = list_piper_can_interfaces()
    if port in interfaces_by_serial:
        return interfaces_by_serial[port]

    suffix_matches = {
        interface for serial, interface in interfaces_by_serial.items() if serial.endswith(port)
    }
    if len(suffix_matches) == 1:
        return suffix_matches.pop()

    if not interfaces_by_serial or port.startswith("can"):
        return port

    available = ", ".join(
        f"{serial}->{interface}" for serial, interface in sorted(interfaces_by_serial.items())
    )
    raise ValueError(
        f"Could not resolve Piper CAN port or USB-CAN serial '{port}'. Available CAN serials: {available}"
    )


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


def set_piper_role(arm: Any, role: int) -> None:
    arm.MasterSlaveConfig(role, 0x00, 0x00, 0x00)
    if PIPER_ROLE_SWITCH_SETTLE_S > 0:
        time.sleep(PIPER_ROLE_SWITCH_SETTLE_S)
