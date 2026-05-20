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

import ctypes
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ARX_SDK_ROOT = Path(
    os.path.expanduser(os.environ.get("ARX_SDK_ROOT", "~/workspace/ARX_X5/py/arx_x5_python"))
)


def _setup_sdk_search_path(sdk_root: Path) -> None:
    if sdk_root.is_dir() and str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))


def _preload_sdk_shared_objects(sdk_root: Path) -> None:
    if not sdk_root.is_dir():
        return

    shared_object_dirs = [
        sdk_root / "bimanual" / "api" / "arx_x5_src",
        sdk_root / "bimanual" / "api",
    ]
    for directory in shared_object_dirs:
        if not directory.is_dir():
            continue
        for shared_object in sorted(directory.glob("*.so")):
            if shared_object.name.endswith("-arm64.so"):
                continue
            try:
                ctypes.cdll.LoadLibrary(str(shared_object))
            except OSError as error:
                logger.debug("Skipping optional ARX5 shared object %s: %s", shared_object, error)


_setup_sdk_search_path(DEFAULT_ARX_SDK_ROOT)
_preload_sdk_shared_objects(DEFAULT_ARX_SDK_ROOT)

try:
    SingleArm = importlib.import_module("bimanual").SingleArm
    HAS_ARX_SDK = True
except ImportError:
    SingleArm = None
    HAS_ARX_SDK = False
    logger.warning("ARX5 SDK not found under %s. The follower will run in stub mode.", DEFAULT_ARX_SDK_ROOT)


class _StubArm:
    """Fallback arm used when the vendor SDK or hardware is unavailable."""

    def __init__(self, config: dict[str, Any]):
        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._ee_pose = np.zeros(7, dtype=np.float64)
        logger.info("Using ARX5 stub arm with config=%s", config)

    def go_home(self) -> None:
        self._joint_positions[:] = 0.0
        self._ee_pose[:] = 0.0

    def protect_mode(self) -> None:
        return

    def gravity_compensation(self) -> None:
        return

    def set_ee_pose_xyzrpy(self, xyzrpy: list[float]) -> None:
        self._ee_pose[:6] = np.asarray(xyzrpy, dtype=np.float64)

    def set_joint_positions(self, joints: list[float]) -> None:
        joints_array = np.asarray(joints, dtype=np.float64)
        self._joint_positions[: min(6, joints_array.size)] = joints_array[:6]

    def set_catch_pos(self, value: float) -> None:
        self._ee_pose[6] = float(value)
        self._joint_positions[6] = float(value)

    def get_joint_positions(self) -> list[float]:
        return self._joint_positions.tolist()

    def get_ee_pose_xyzrpy(self) -> list[float]:
        return self._ee_pose[:6].tolist()


class ARX5ArmClient:
    """Thin wrapper around the ARX5 vendor SDK with a deterministic stub fallback."""

    def __init__(
        self,
        *,
        can_port: str,
        arm_type: int,
        use_stub: bool,
        recorded_pose_path: Path,
    ):
        self.recorded_pose_path = recorded_pose_path
        self.recorded_pose_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_sent_pose = [0.0] * 7

        if use_stub or not HAS_ARX_SDK:
            self.arm = _StubArm({"can_port": can_port, "type": arm_type})
        else:
            self.arm = SingleArm({"can_port": can_port, "type": arm_type})

        self._last_sent_pose = self.get_state()

    def _read_gripper_position(self) -> float:
        if hasattr(self.arm, "get_catch_pos"):
            value = self.arm.get_catch_pos()
            if isinstance(value, (list, tuple, np.ndarray)):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)

        if hasattr(self, "_last_sent_pose") and len(self._last_sent_pose) >= 7:
            return float(self._last_sent_pose[6])
        return 0.0

    def get_state(self) -> list[float]:
        joint_positions = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        gripper = self._read_gripper_position()
        state = np.zeros(7, dtype=np.float64)
        state[: min(6, joint_positions.size)] = joint_positions[:6]
        state[6] = gripper
        return state.tolist()

    def get_joint_positions(self) -> list[float]:
        joint_positions = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        gripper = self._read_gripper_position()
        state = np.zeros(7, dtype=np.float64)
        state[: min(6, joint_positions.size)] = joint_positions[:6]
        state[6] = gripper
        return state.tolist()

    def get_joint_velocities(self) -> list[float]:
        """Read 6-DoF joint velocities from the ARX5 SDK.

        Returns a zero vector for the stub arm or when the SDK omits the API
        (e.g. older firmware). Gripper velocity is not exposed by the SDK.
        """
        getter = getattr(self.arm, "get_joint_velocities", None)
        if getter is None:
            return [0.0] * 6
        try:
            velocities = np.asarray(getter(), dtype=np.float64).reshape(-1)
        except Exception:
            return [0.0] * 6
        out = np.zeros(6, dtype=np.float64)
        out[: min(6, velocities.size)] = velocities[:6]
        return out.tolist()

    def send_joint(self, joint: list[float]) -> list[float]:
        full_joint = list(joint[:7])
        if len(full_joint) < 7:
            full_joint.extend([0.0] * (7 - len(full_joint)))

        if hasattr(self.arm, "set_joint_positions"):
            self.arm.set_joint_positions(full_joint[:6])
        elif hasattr(self.arm, "set_joint_pos"):
            self.arm.set_joint_pos(full_joint[:6])
        elif hasattr(self.arm, "set_joints"):
            self.arm.set_joints(full_joint[:6])
        else:
            raise RuntimeError("ARX5 SDK arm object does not expose a supported joint write API.")

        self.arm.set_catch_pos(float(full_joint[6]))
        self._last_sent_pose = full_joint
        return full_joint

    def hold_position(self) -> None:
        self.send_joint(self.get_state())
        time.sleep(0.2)

    def go_home(self) -> None:
        self.arm.go_home()

    def protect_mode(self) -> None:
        self.arm.protect_mode()

    def enter_teach_mode(self) -> None:
        self.arm.gravity_compensation()

    def has_recorded_pose(self) -> bool:
        return self.recorded_pose_path.is_file()

    def save_recorded_pose(self, pose: list[float] | None = None) -> list[float]:
        recorded_pose = self.get_state() if pose is None else list(pose)
        payload = {"recorded_pose": recorded_pose}
        self.recorded_pose_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return recorded_pose

    def load_recorded_pose(self) -> list[float]:
        payload = json.loads(self.recorded_pose_path.read_text(encoding="utf-8"))
        recorded_pose = payload.get("recorded_pose")
        if not isinstance(recorded_pose, list) or len(recorded_pose) < 7:
            raise ValueError(f"Invalid recorded pose file: {self.recorded_pose_path}")
        return [float(value) for value in recorded_pose[:7]]

    def move_to_recorded(self, *, num_steps: int = 20, step_interval_s: float = 0.1) -> list[float]:
        start = np.asarray(self.get_state(), dtype=np.float64)
        target = np.asarray(self.load_recorded_pose(), dtype=np.float64)
        for index in range(1, num_steps + 1):
            interpolation = index / num_steps
            pose = start + (target - start) * interpolation
            self.send_joint(pose.tolist())
            if index < num_steps and step_interval_s > 0:
                time.sleep(step_interval_s)
        return target.tolist()
