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
import multiprocessing as mp
import traceback
from contextlib import suppress
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.piper_leader import (
    PiperLeader,
    PiperLeaderConfig,
    PiperXLeader,
    PiperXLeaderConfig,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

from ..teleoperator import Teleoperator
from .config_bi_piper_leader import BiPiperLeaderConfig, BiPiperXLeaderConfig

logger = logging.getLogger(__name__)


def _bi_piper_leader_worker(conn, arm_cls, arm_config) -> None:
    arm = arm_cls(arm_config)
    try:
        while True:
            request = conn.recv()
            command = request["command"]
            if command == "__close__":
                break

            try:
                args = request.get("args", ())
                kwargs = request.get("kwargs", {})
                result = getattr(arm, command)(*args, **kwargs)
                conn.send({"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                conn.send(
                    {
                        "ok": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        try:
            if arm.is_connected:
                arm.disconnect()
        except Exception:  # noqa: BLE001
            pass
        conn.close()


class _PiperLeaderProcessProxy:
    def __init__(self, arm_cls, arm_config):
        self._arm_cls = arm_cls
        self._arm_config = arm_config
        self._ctx = mp.get_context("spawn")
        self._parent_conn = None
        self._process = None
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_bi_piper_leader_worker,
            args=(child_conn, self._arm_cls, self._arm_config),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._parent_conn = parent_conn
        self._process = process

    def _call(self, command: str, *args, **kwargs):
        self._ensure_process()
        assert self._parent_conn is not None
        self._parent_conn.send({"command": command, "args": args, "kwargs": kwargs})
        response = self._parent_conn.recv()
        if response["ok"]:
            return response.get("result")
        raise RuntimeError(
            f"bi_piper leader worker command '{command}' failed: {response['error']}\n"
            f"{response['traceback']}"
        )

    def connect(self) -> None:
        try:
            self._call("connect")
            self._is_connected = True
        except Exception:
            self.disconnect()
            raise

    def configure(self) -> None:
        self._call("configure")

    def setup_motors(self) -> None:
        self._call("setup_motors")

    def set_manual_control(self, enabled: bool) -> None:
        self._call("set_manual_control", enabled)

    def get_action(self) -> RobotAction:
        return self._call("get_action")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self._call("send_feedback", feedback)

    def disconnect(self) -> None:
        if self._process is None:
            self._is_connected = False
            return

        disconnect_error: Exception | None = None
        if self._parent_conn is not None:
            try:
                if self._is_connected:
                    self._call("disconnect")
            except Exception as exc:
                disconnect_error = exc
            with suppress(Exception):
                self._parent_conn.send({"command": "__close__"})
            with suppress(Exception):
                self._parent_conn.close()

        if self._process.is_alive():
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

        self._parent_conn = None
        self._process = None
        self._is_connected = False
        if disconnect_error is not None:
            raise disconnect_error


class BiPiperLeader(Teleoperator):
    """Bimanual PiPER/PiPER-X leader arms."""

    config_class = BiPiperLeaderConfig
    name = "bi_piper_leader"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "manual_control",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "command_speed_ratio",
        "command_high_follow",
        "mode_refresh_interval_s",
        "enable_timeout_s",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, arm_config_cls, side_cfg, side: str):
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        return arm_config_cls(**kwargs)

    def __init__(self, config: BiPiperLeaderConfig | BiPiperXLeaderConfig):
        self.id = config.id
        self.config = config
        self._use_process_isolation = config.process_isolation

        if config.type == "bi_piperx_leader":
            arm_config_cls = PiperXLeaderConfig
            arm_cls = PiperXLeader
        else:
            arm_config_cls = PiperLeaderConfig
            arm_cls = PiperLeader

        left_arm_config = self._build_arm_config(arm_config_cls, config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(arm_config_cls, config.right_arm_config, "right")

        if self._use_process_isolation:
            self.left_arm = _PiperLeaderProcessProxy(arm_cls, left_arm_config)
            self.right_arm = _PiperLeaderProcessProxy(arm_cls, right_arm_config)
        else:
            self.left_arm = arm_cls(left_arm_config)
            self.right_arm = arm_cls(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features
        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.left_arm.connect()
        try:
            self.right_arm.connect()
        except Exception:
            try:
                self.left_arm.disconnect()
            except Exception:
                logger.exception("Failed to disconnect left PiPER leader after right connect error.")
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def set_manual_control(self, enabled: bool) -> None:
        try:
            self.left_arm.set_manual_control(enabled)
            self.right_arm.set_manual_control(enabled)
        except Exception:
            for arm in (self.left_arm, self.right_arm):
                try:
                    arm.set_manual_control(True)
                except Exception:
                    logger.exception("Failed to restore a PiPER leader arm after role-switch error.")
            raise

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict: RobotAction = {}
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})
        return action_dict

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        left_feedback: dict[str, Any] = {}
        right_feedback: dict[str, Any] = {}
        for key, value in feedback.items():
            if key.startswith("left_"):
                left_feedback[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_feedback[key.removeprefix("right_")] = value
        try:
            self.left_arm.send_feedback(left_feedback)
            self.right_arm.send_feedback(right_feedback)
        except Exception:
            for arm in (self.left_arm, self.right_arm):
                try:
                    arm.set_manual_control(True)
                except Exception:
                    logger.exception("Failed to restore a PiPER leader arm after feedback error.")
            raise

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.left_arm.disconnect()
        finally:
            self.right_arm.disconnect()


class BiPiperXLeader(BiPiperLeader):
    config_class = BiPiperXLeaderConfig
    name = "bi_piperx_leader"
