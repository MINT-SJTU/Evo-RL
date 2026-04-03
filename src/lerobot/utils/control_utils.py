# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

########################################################################################
# Utilities
########################################################################################


import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS

# Minimum interval (seconds) between consecutive intervention toggle presses.
INTERVENTION_TOGGLE_COOLDOWN_S = 0.5


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception as e:
        logging.info("pynput unavailable; using headless controls instead: %s", e)
        return True


class TTYKeyboardListener:
    """Read control keys directly from the current TTY for SSH/headless sessions."""

    def __init__(
        self,
        events: dict[str, Any],
        intervention_toggle_key: str,
        critical_phase_toggle_key: str | None,
        episode_success_key: str | None,
        episode_failure_key: str | None,
    ):
        self.events = events
        self.intervention_toggle_key = intervention_toggle_key.lower()
        self.critical_phase_toggle_key = (
            critical_phase_toggle_key.lower() if critical_phase_toggle_key else None
        )
        self.episode_success_key = episode_success_key.lower() if episode_success_key else None
        self.episode_failure_key = episode_failure_key.lower() if episode_failure_key else None
        self._fd = sys.stdin.fileno()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_attrs = None
        self._last_intervention_time: float = 0.0
        self._last_critical_phase_time: float = 0.0

    def start(self):
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="tty-keyboard-listener", daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _run(self):
        while not self._stop_event.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue

            try:
                key = self._read_key()
                if key is not None:
                    self._handle_key(key)
            except Exception as e:
                logging.warning("TTY keyboard listener stopped after read error: %s", e)
                self._stop_event.set()

    def _read_key(self) -> str | None:
        chunk = os.read(self._fd, 1)
        if not chunk:
            return None

        if chunk == b"\x1b":
            sequence = bytearray(chunk)
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0.01)
                if not ready:
                    break
                sequence.extend(os.read(self._fd, 1))
                last_byte = bytes(sequence[-1:])
                if len(sequence) >= 3 and last_byte in {b"A", b"B", b"C", b"D", b"~"}:
                    break

            sequence_bytes = bytes(sequence)
            if sequence_bytes in {b"\x1b[C", b"\x1bOC"}:
                return "RIGHT"
            if sequence_bytes in {b"\x1b[D", b"\x1bOD"}:
                return "LEFT"
            if sequence_bytes == b"\x1b":
                return "ESC"
            return None

        try:
            return chunk.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _handle_key(self, key: str):
        normalized = key.lower() if len(key) == 1 else key
        if normalized == "RIGHT":
            print("Right arrow key pressed. Exiting loop...")
            self.events["exit_early"] = True
        elif normalized == "LEFT":
            print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif normalized == "ESC":
            print("Escape key pressed. Stopping data recording...")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
        elif normalized == self.intervention_toggle_key:
            now = time.monotonic()
            if now - self._last_intervention_time < INTERVENTION_TOGGLE_COOLDOWN_S:
                return
            self._last_intervention_time = now
            print(f"'{self.intervention_toggle_key}' key pressed. Toggling intervention mode...")
            self.events["toggle_intervention"] = True
        elif self.critical_phase_toggle_key and normalized == self.critical_phase_toggle_key:
            now = time.monotonic()
            if now - self._last_critical_phase_time < INTERVENTION_TOGGLE_COOLDOWN_S:
                return
            self._last_critical_phase_time = now
            print(f"'{self.critical_phase_toggle_key}' key pressed. Toggling critical phase...")
            self.events["toggle_critical_phase"] = True
        elif self.episode_success_key and normalized == self.episode_success_key:
            print(f"'{self.episode_success_key}' key pressed. Marking episode as success and exiting loop...")
            self.events["episode_outcome"] = EPISODE_SUCCESS
            self.events["exit_early"] = True
        elif self.episode_failure_key and normalized == self.episode_failure_key:
            print(f"'{self.episode_failure_key}' key pressed. Marking episode as failure and exiting loop...")
            self.events["episode_outcome"] = EPISODE_FAILURE
            self.events["exit_early"] = True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener(
    intervention_toggle_key: str = "i",
    critical_phase_toggle_key: str | None = None,
    episode_success_key: str | None = None,
    episode_failure_key: str | None = None,
):
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape, intervention
    toggle key, and optional episode success/failure keys) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["toggle_intervention"] = False
    if critical_phase_toggle_key is not None:
        events["toggle_critical_phase"] = False
    events["episode_outcome"] = None

    listener = None
    if not is_headless():
        # Only import pynput if not in a headless environment
        from pynput import keyboard

        last_intervention_time = [0.0]
        last_critical_phase_time = [0.0]

        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Exiting loop...")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                elif hasattr(key, "char") and key.char and key.char.lower() == intervention_toggle_key.lower():
                    now = time.monotonic()
                    if now - last_intervention_time[0] < INTERVENTION_TOGGLE_COOLDOWN_S:
                        return
                    last_intervention_time[0] = now
                    print(f"'{intervention_toggle_key}' key pressed. Toggling intervention mode...")
                    events["toggle_intervention"] = True
                elif (
                    critical_phase_toggle_key
                    and hasattr(key, "char")
                    and key.char
                    and key.char.lower() == critical_phase_toggle_key.lower()
                ):
                    now = time.monotonic()
                    if now - last_critical_phase_time[0] < INTERVENTION_TOGGLE_COOLDOWN_S:
                        return
                    last_critical_phase_time[0] = now
                    print(f"'{critical_phase_toggle_key}' key pressed. Toggling critical phase...")
                    events["toggle_critical_phase"] = True
                elif (
                    episode_success_key
                    and hasattr(key, "char")
                    and key.char
                    and key.char.lower() == episode_success_key.lower()
                ):
                    print(f"'{episode_success_key}' key pressed. Marking episode as success and exiting loop...")
                    events["episode_outcome"] = EPISODE_SUCCESS
                    events["exit_early"] = True
                elif (
                    episode_failure_key
                    and hasattr(key, "char")
                    and key.char
                    and key.char.lower() == episode_failure_key.lower()
                ):
                    print(f"'{episode_failure_key}' key pressed. Marking episode as failure and exiting loop...")
                    events["episode_outcome"] = EPISODE_FAILURE
                    events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener, events

    if sys.stdin.isatty():
        listener = TTYKeyboardListener(
            events=events,
            intervention_toggle_key=intervention_toggle_key,
            critical_phase_toggle_key=critical_phase_toggle_key,
            episode_success_key=episode_success_key,
            episode_failure_key=episode_failure_key,
        )
        listener.start()
        logging.warning(
            "Headless environment detected. Using terminal keyboard controls over the current TTY; on-screen camera display remains unavailable."
        )
        return listener, events

    logging.warning(
        "Headless environment detected without an interactive TTY. On-screen cameras display and keyboard inputs will not be available."
    )

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


def sanity_check_bimanual_piper_pair(robot_cfg, teleop_cfg) -> None:
    """Ensure bimanual PiPER configs are not mixed between PiPER and PiPER-X variants."""
    if teleop_cfg is None:
        return

    robot_type = getattr(robot_cfg, "type", None)
    teleop_type = getattr(teleop_cfg, "type", None)
    expected_teleop_by_robot = {
        "bi_piper_follower": "bi_piper_leader",
        "bi_piperx_follower": "bi_piperx_leader",
    }
    expected_robot_by_teleop = {teleop: robot for robot, teleop in expected_teleop_by_robot.items()}

    if robot_type in expected_teleop_by_robot and teleop_type != expected_teleop_by_robot[robot_type]:
        expected = expected_teleop_by_robot[robot_type]
        raise ValueError(
            f"In bimanual PiPER mode, '{robot_type}' must be paired with '{expected}', got '{teleop_type}'."
        )
    if teleop_type in expected_robot_by_teleop and robot_type != expected_robot_by_teleop[teleop_type]:
        expected = expected_robot_by_teleop[teleop_type]
        raise ValueError(
            f"In bimanual PiPER mode, '{teleop_type}' must be paired with '{expected}', got '{robot_type}'."
        )
