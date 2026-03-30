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

import logging
import time
from functools import cached_property
from pathlib import Path

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .arx5_client import ARX5ArmClient
from .config_arx5_follower import ARX5FollowerConfig
from .arx5_runtime import Arx5Runtime

logger = logging.getLogger(__name__)

ARX5_REAL_STATE_KEYS = (
    "joint.1",
    "joint.2",
    "joint.3",
    "joint.4",
    "joint.5",
    "joint.6",
    "gripper.pos",
)
ARX5_STATE_KEYS = ARX5_REAL_STATE_KEYS
ARX5_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class ARX5Follower(Robot):
    """ARX5 follower arm with direct joint-space control for Pi0.5-style policies."""

    config_class = ARX5FollowerConfig
    name = "arx5_follower"

    def __init__(self, config: ARX5FollowerConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._arm_client: ARX5ArmClient | None = None
        self._last_joint = np.zeros(7, dtype=np.float64)
        self.cameras = make_cameras_from_configs(config.cameras)
        default_recorded_pose = self.calibration_dir / f"{self.id or 'default'}_recorded_pose.json"
        self.recorded_pose_path = config.recorded_pose_path or default_recorded_pose

    @property
    def _state_features(self) -> dict[str, type]:
        return dict.fromkeys(ARX5_STATE_KEYS, float)

    @property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (self.config.cameras[name].height, self.config.cameras[name].width, 3)
            for name in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._state_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_already_connected
    def connect(self, calibrate: bool = True, *, reuse_arm_client: ARX5ArmClient | None = None) -> None:
        del calibrate
        connected_cameras = []
        try:
            for name, camera in self.cameras.items():
                logger.info("Connecting camera '%s' via %s", name, camera)
                camera.connect()
                connected_cameras.append(camera)
                logger.info("Camera '%s' connected.", name)

            if reuse_arm_client is not None:
                logger.info("Reusing existing ARX5 arm client (infer↔VR handoff).")
                self._arm_client = reuse_arm_client
            else:
                logger.info("Connecting ARX5 arm client (stub=%s).", self.config.use_stub)
                self._arm_client = ARX5ArmClient(
                    can_port=self.config.port,
                    arm_type=self.config.arm_type,
                    use_stub=self.config.use_stub,
                    recorded_pose_path=self.recorded_pose_path,
                )
            if self.config.startup_sleep_s > 0:
                time.sleep(self.config.startup_sleep_s)
            self._last_joint = np.asarray(self._arm_client.get_state(), dtype=np.float64)
            self._is_connected = True
        except Exception:
            self._arm_client = None
            for camera in reversed(connected_cameras):
                camera.disconnect()
            raise

        logger.info("%s connected.", self)

    def _require_arm_client(self) -> ARX5ArmClient:
        if self._arm_client is None:
            raise RuntimeError("ARX5 arm client is not connected.")
        return self._arm_client

    def _state_vector_to_dict(self, vector: list[float]) -> dict[str, float]:
        return {key: float(value) for key, value in zip(ARX5_REAL_STATE_KEYS, vector[:7], strict=True)}

    def get_state_vector(self) -> list[float]:
        return self._require_arm_client().get_state()

    def get_joint_vector(self) -> list[float]:
        return self._require_arm_client().get_joint_positions()

    def _extract_joint_from_action(self, action: RobotAction) -> list[float]:
        values = [float(action.get(key, 0.0)) for key in ARX5_REAL_STATE_KEYS]
        values[-1] = float(np.clip(values[-1], self.config.gripper_min, self.config.gripper_max))

        max_translation_step_m = self.config.max_translation_step_m
        if max_translation_step_m is not None:
            current_joint = np.asarray(self._require_arm_client().get_state(), dtype=np.float64)
            target_joint = np.asarray(values, dtype=np.float64)
            delta = target_joint[:3] - current_joint[:3]
            distance = float(np.linalg.norm(delta))
            if distance > max_translation_step_m and distance > 0:
                target_joint[:3] = current_joint[:3] + delta * (max_translation_step_m / distance)
                values = target_joint.tolist()
                logger.warning(
                    "Clamped ARX5 first-3-joint step from %.4f to %.4f.",
                    distance,
                    max_translation_step_m,
                )
        return values

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._state_vector_to_dict(self.get_state_vector())
        for name, camera in self.cameras.items():
            state[name] = camera.async_read()
        return state

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        joint = self._extract_joint_from_action(action)
        sent_joint = self._require_arm_client().send_joint(joint)
        self._last_joint = np.asarray(sent_joint, dtype=np.float64)
        return self._state_vector_to_dict(sent_joint)

    def hold_position(self) -> None:
        self._require_arm_client().hold_position()
        self._last_joint = np.asarray(self._require_arm_client().get_state(), dtype=np.float64)

    def go_home(self) -> None:
        self._require_arm_client().go_home()
        self._last_joint = np.asarray(self._require_arm_client().get_state(), dtype=np.float64)

    def protect_mode(self) -> None:
        self._require_arm_client().protect_mode()

    def enter_teach_mode(self) -> None:
        self._require_arm_client().enter_teach_mode()

    def exit_teach_mode_and_record(self) -> Path:
        arm_client = self._require_arm_client()
        arm_client.save_recorded_pose()
        arm_client.hold_position()
        self._last_joint = np.asarray(arm_client.get_state(), dtype=np.float64)
        return arm_client.recorded_pose_path

    def has_recorded_pose(self) -> bool:
        return self._require_arm_client().has_recorded_pose()

    def move_to_recorded(self, *, num_steps: int = 20, step_interval_s: float = 0.1) -> RobotAction:
        joint = self._require_arm_client().move_to_recorded(
            num_steps=num_steps,
            step_interval_s=step_interval_s,
        )
        self._last_joint = np.asarray(joint, dtype=np.float64)
        return self._state_vector_to_dict(joint)

    @check_if_not_connected
    def disconnect(self, *, protect: bool | None = None, release_arm_to: Arx5Runtime | None = None) -> None:
        """
        Disconnect cameras and arm client.

        Args:
            protect: Override whether to call protect_mode before disconnecting.
                - None: follow config.protect_on_disconnect (default behavior)
                - True: force protect_mode
                - False: skip protect_mode (useful for infer->VR CAN handoff)
            release_arm_to: If set, move the arm client to this :class:`~.arx5_runtime.Arx5Runtime`
                instead of dropping it (same CAN session for VR). ``protect_mode`` is not applied
                to the arm in this path so torque is not dropped before handoff.
        """
        try:
            if release_arm_to is not None:
                if self._arm_client is not None:
                    release_arm_to.bind_client(self._arm_client)
            else:
                do_protect = self.config.protect_on_disconnect if protect is None else protect
                if do_protect and self._arm_client is not None:
                    self._arm_client.protect_mode()
        finally:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self._arm_client = None
            self._is_connected = False
            logger.info("%s disconnected.", self)
