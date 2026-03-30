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

"""Shared ARX5 runtime: one `ARX5ArmClient` / SDK arm for inference and VR teleop.

`Arx5ControlMode` enforces a single logical writer (infer vs VR). Reads are allowed in any
mode once connected. Use :class:`SharedARXX5Interface` in Evo-RL code (subclass of vr2robot
:class:`ARXX5Interface`) so VR stacks can reuse the same underlying arm without editing
``ARXX5TeleopController`` in vr2robot.
"""

from __future__ import annotations

__all__ = [
    "Arx5ControlMode",
    "Arx5Runtime",
    "SharedARXX5Interface",
]

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient

# Keep in sync with `ARX5_REAL_STATE_KEYS` in `arx5_follower.py` (avoid import cycles).
_ARX5_REAL_STATE_KEYS = (
    "joint.1",
    "joint.2",
    "joint.3",
    "joint.4",
    "joint.5",
    "joint.6",
    "gripper.pos",
)


class Arx5ControlMode(str, Enum):
    """Who may call :meth:`Arx5Runtime.write_targets` / :meth:`Arx5Runtime.send_joint`."""

    IDLE = "idle"
    INFER = "infer"
    VR = "vr"


_registry: dict[tuple[str, int], Arx5Runtime] = {}
_registry_lock = threading.Lock()


class Arx5Runtime:
    """Owns a single :class:`ARX5ArmClient` for one CAN port + arm type.

    Thread-safety: all public methods use an internal re-entrant lock. Only the active
    :class:`Arx5ControlMode` may issue joint writes; other writers get :class:`RuntimeError`.

    Typical wiring:

    * Inference: ``runtime.connect(...)`` then ``with runtime.exclusive(Arx5ControlMode.INFER): ...``
    * VR: subclass :class:`SharedARXX5Interface` and pass ``runtime``; in teleop loop use
      ``with runtime.exclusive(Arx5ControlMode.VR):`` around sends, or call
      ``acquire_mode`` / ``release_mode`` once per session.
    """

    def __init__(
        self,
        *,
        can_port: str,
        arm_type: int = 0,
        dt: float = 0.01,
        use_stub: bool = False,
        recorded_pose_path: Path | None = None,
    ) -> None:
        self.can_port = can_port
        self.arm_type = arm_type
        self.dt = dt
        self._use_stub = use_stub
        self._recorded_pose_path = recorded_pose_path
        self._client: ARX5ArmClient | None = None
        self._lock = threading.RLock()
        self._mode = Arx5ControlMode.IDLE

    @classmethod
    def get_or_create(
        cls,
        *,
        can_port: str,
        arm_type: int = 0,
        dt: float = 0.01,
        use_stub: bool = False,
        recorded_pose_path: Path | None = None,
    ) -> Arx5Runtime:
        """Return the process-wide runtime for ``(can_port, arm_type)``."""
        key = (can_port, arm_type)
        with _registry_lock:
            existing = _registry.get(key)
            if existing is not None:
                return existing
            inst = cls(
                can_port=can_port,
                arm_type=arm_type,
                dt=dt,
                use_stub=use_stub,
                recorded_pose_path=recorded_pose_path,
            )
            _registry[key] = inst
            return inst

    @classmethod
    def clear_registry(cls) -> None:
        """Test helper: drop cached instances (does not disconnect hardware)."""
        with _registry_lock:
            _registry.clear()

    @property
    def mode(self) -> Arx5ControlMode:
        with self._lock:
            return self._mode

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._client is not None

    @property
    def client(self) -> ARX5ArmClient:
        with self._lock:
            if self._client is None:
                raise RuntimeError("Arx5Runtime is not connected; call connect() first.")
            return self._client

    @property
    def sdk_arm(self) -> Any:
        """Underlying SDK object (``SingleArm`` or stub), same reference as ``client.arm``."""
        return self.client.arm

    def connect(self, *, recorded_pose_path: Path | None = None) -> None:
        """Construct :class:`ARX5ArmClient` once. Idempotent if already connected."""
        with self._lock:
            if self._client is not None:
                return
            path = recorded_pose_path or self._recorded_pose_path
            if path is None:
                raise ValueError("recorded_pose_path is required on first connect().")
            self._client = ARX5ArmClient(
                can_port=self.can_port,
                arm_type=self.arm_type,
                use_stub=self._use_stub,
                recorded_pose_path=path,
            )
            self._recorded_pose_path = path

    def bind_client(self, client: ARX5ArmClient) -> None:
        """Take ownership of an existing client (infer→VR handoff without re-opening CAN)."""
        with self._lock:
            self._client = client
            self._recorded_pose_path = client.recorded_pose_path

    def take_client(self) -> ARX5ArmClient | None:
        """Remove and return the client so :class:`~lerobot.robots.arx5_follower.arx5_follower.ARX5Follower` can ``connect(reuse_arm_client=...)``."""
        with self._lock:
            client = self._client
            self._client = None
            self._mode = Arx5ControlMode.IDLE
            return client

    def disconnect(self) -> None:
        """Drop client reference (no ``protect_mode``; match :class:`ARX5Follower` policy at call site)."""
        with self._lock:
            self._client = None
            self._mode = Arx5ControlMode.IDLE

    def acquire_mode(self, mode: Arx5ControlMode, *, force: bool = False) -> None:
        """Reserve write mode. Fails if another non-idle mode is active unless ``force``."""
        if mode == Arx5ControlMode.IDLE:
            raise ValueError("Use release_mode() to return to IDLE.")
        with self._lock:
            if self._mode == mode:
                return
            if self._mode != Arx5ControlMode.IDLE and not force:
                raise RuntimeError(
                    f"Arx5Runtime mode is {self._mode!r}; cannot acquire {mode!r} without force=True"
                )
            self._mode = mode

    def release_mode(self) -> None:
        """Clear write mode to :data:`Arx5ControlMode.IDLE`."""
        with self._lock:
            self._mode = Arx5ControlMode.IDLE

    @contextmanager
    def exclusive(self, mode: Arx5ControlMode) -> Iterator[None]:
        """Temporarily acquire ``mode`` and release to IDLE on exit."""
        self.acquire_mode(mode)
        try:
            yield
        finally:
            self.release_mode()

    def read_state(self) -> list[float]:
        """7-float state: 6 joints + gripper (same as :meth:`ARX5ArmClient.get_state`)."""
        with self._lock:
            return self.client.get_state()

    def send_joint(self, joint: list[float] | np.ndarray, *, mode: Arx5ControlMode) -> list[float]:
        """Send joint targets; ``mode`` must match :attr:`Arx5Runtime.mode`."""
        joint_list = np.asarray(joint, dtype=np.float64).ravel().tolist()
        with self._lock:
            if self._mode != mode:
                raise RuntimeError(
                    f"send_joint requires mode {mode!r}, current is {self._mode!r}. "
                    "Call acquire_mode() or use exclusive()."
                )
            return self.client.send_joint(joint_list)

    def write_targets(
        self,
        targets: Mapping[str, float] | np.ndarray | list[float],
        *,
        mode: Arx5ControlMode,
    ) -> list[float]:
        """Write from a 7-vector or a dict keyed like ``joint.1`` … ``gripper.pos``."""
        if isinstance(targets, Mapping):
            joint = [float(targets.get(k, 0.0)) for k in _ARX5_REAL_STATE_KEYS]
        else:
            joint = np.asarray(targets, dtype=np.float64).ravel().tolist()
        return self.send_joint(joint, mode=mode)

    def hold_position(self) -> None:
        with self._lock:
            self.client.hold_position()

    def go_home(self) -> None:
        with self._lock:
            self.client.go_home()

    def protect_mode(self) -> None:
        with self._lock:
            self.client.protect_mode()

    def enter_teach_mode(self) -> None:
        with self._lock:
            self.client.enter_teach_mode()


try:
    from xrobotoolkit_teleop.hardware.interface.arx_x5 import ARXX5Interface as _ARXX5Interface

    class SharedARXX5Interface(_ARXX5Interface):
        """Subclass of vr2robot :class:`ARXX5Interface` bound to :class:`Arx5Runtime`.

        Does not open a second ``SingleArm``; uses ``runtime.sdk_arm`` after
        ``runtime.connect()``. Use this from Evo-RL VR adapters instead of editing
        ``ARXX5TeleopController`` in vr2robot.
        """

        def __init__(self, runtime: Arx5Runtime, *, dt: float | None = None) -> None:
            self.dt = dt if dt is not None else runtime.dt
            self.arm_type = runtime.arm_type
            self.num_joints = 6
            if not runtime.is_connected:
                raise RuntimeError("SharedARXX5Interface requires Arx5Runtime.connect() first.")
            self.arm = runtime.sdk_arm

except ImportError:

    class SharedARXX5Interface:  # type: ignore[no-redef]
        """Placeholder when vr2robot / xrobotoolkit_teleop is not importable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                "SharedARXX5Interface requires vr2robot (xrobotoolkit_teleop). "
                "Install or extend PYTHONPATH."
            )
