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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import math
import os
import select
import sys
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    IdentityProcessorStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.bi_piper_follower import BiPiperFollowerConfig, BiPiperXFollowerConfig
from lerobot.robots.piper_follower import PiperFollowerConfigBase
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_piper_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    piper_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.bi_piper_leader import BiPiperLeaderConfig, BiPiperXLeaderConfig
from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.piper_sdk import get_piper_sdk, parse_piper_log_level
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

LOOP_STATUS_INTERVAL_S = 0.5
FAKE_INFERENCE_BASE_JOINT_SUFFIX = "joint_1.pos"


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Press this key during teleoperation to toggle a simple fake policy/debug action.
    fake_inference_key: str = ""
    # Total base-joint sweep around the pose captured when fake inference starts.
    fake_inference_total_swing_deg: float = 120.0
    # Seconds per full back-and-forth base-joint cycle.
    fake_inference_period_s: float = 10.0
    # Also command the leader arms as temporary followers during fake inference.
    fake_inference_control_leaders: bool = False
    # Speed ratio used when leader arms are temporarily controlled as followers.
    fake_inference_leader_speed_ratio: int = 10


class _NonBlockingKeyReader:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._fd: int | None = None
        self._old_term_settings = None
        self._termios = None
        self._msvcrt = None

    def __enter__(self):
        if not self.enabled:
            return self
        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt
            return self
        if not sys.stdin.isatty():
            self.enabled = False
            return self

        import termios
        import tty

        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        if self._termios is not None and self._fd is not None and self._old_term_settings is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_term_settings)

    def get_key(self) -> str | None:
        if not self.enabled:
            return None
        if self._msvcrt is not None:
            if self._msvcrt.kbhit():
                key = self._msvcrt.getwch()
                if key == "\x03":
                    raise KeyboardInterrupt
                return key.lower()
            return None
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        key = os.read(self._fd, 1).decode(errors="ignore")
        if key == "\x03":
            raise KeyboardInterrupt
        return key.lower()


def _is_fake_inference_base_joint_key(key: str) -> bool:
    return key == FAKE_INFERENCE_BASE_JOINT_SUFFIX or key.endswith(f"_{FAKE_INFERENCE_BASE_JOINT_SUFFIX}")


def _make_fake_inference_action(
    hold_action: RobotAction,
    *,
    elapsed_s: float,
    total_swing_deg: float,
    period_s: float,
) -> RobotAction:
    action = dict(hold_action)
    safe_period_s = max(period_s, 1e-6)
    amplitude_deg = abs(total_swing_deg) / 2.0
    offset_deg = amplitude_deg * math.sin(2.0 * math.pi * elapsed_s / safe_period_s)
    for key, value in hold_action.items():
        if _is_fake_inference_base_joint_key(key):
            action[key] = value + offset_deg
    return action


def _fake_inference_required_hold_keys(robot: Robot) -> list[str]:
    return [
        key
        for key in robot.action_features
        if key.endswith(".pos") and "joint_" in key
    ]


def _piper_follower_config_from_leader_config(
    leader_cfg,
    *,
    speed_ratio: int,
) -> PiperFollowerConfigBase:
    return PiperFollowerConfigBase(
        port=leader_cfg.port,
        judge_flag=leader_cfg.judge_flag,
        can_auto_init=leader_cfg.can_auto_init,
        log_level=leader_cfg.log_level,
        startup_sleep_s=leader_cfg.startup_sleep_s,
        speed_ratio=speed_ratio,
        high_follow=leader_cfg.command_high_follow,
        mode_refresh_interval_s=leader_cfg.mode_refresh_interval_s,
        enable_on_connect=True,
        enable_timeout_s=leader_cfg.enable_timeout_s,
        calibration_scale=leader_cfg.calibration_scale,
        require_calibration=False,
        sync_gripper=leader_cfg.sync_gripper,
        gripper_effort_default=leader_cfg.gripper_effort_default,
        gripper_status_code=leader_cfg.gripper_status_code,
        cameras={},
        disable_on_disconnect=False,
    )


def _make_fake_inference_leader_follower_config(
    teleop_cfg: TeleoperatorConfig,
    *,
    speed_ratio: int,
) -> RobotConfig | None:
    if isinstance(teleop_cfg, BiPiperXLeaderConfig):
        robot_config_cls = BiPiperXFollowerConfig
    elif isinstance(teleop_cfg, BiPiperLeaderConfig):
        robot_config_cls = BiPiperFollowerConfig
    else:
        return None

    return robot_config_cls(
        id=f"{teleop_cfg.id}_fake_inference_followers" if teleop_cfg.id else None,
        calibration_dir=teleop_cfg.calibration_dir,
        left_arm_config=_piper_follower_config_from_leader_config(
            teleop_cfg.left_arm_config,
            speed_ratio=speed_ratio,
        ),
        right_arm_config=_piper_follower_config_from_leader_config(
            teleop_cfg.right_arm_config,
            speed_ratio=speed_ratio,
        ),
    )


def _switch_piper_role(port_cfg, role: int, *, settle_s: float = 0.2) -> None:
    interface_cls, _ = get_piper_sdk()
    arm = interface_cls(
        can_name=port_cfg.port,
        judge_flag=port_cfg.judge_flag,
        can_auto_init=port_cfg.can_auto_init,
        logger_level=parse_piper_log_level(port_cfg.log_level),
    )
    arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
    try:
        time.sleep(0.05)
        arm.MasterSlaveConfig(role, 0x00, 0x00, 0x00)
        if settle_s > 0:
            time.sleep(settle_s)
    finally:
        arm.DisconnectPort()


class _FakeInferenceLeaderFollowerController:
    def __init__(self, teleop_cfg: TeleoperatorConfig, robot_cfg: RobotConfig):
        self.teleop_cfg = teleop_cfg
        self.robot = make_robot_from_config(robot_cfg)
        self.robot.set_teleop_send_only_mode(True)
        self._connected = False

    def _switch_role(self, role: int) -> None:
        _switch_piper_role(self.teleop_cfg.left_arm_config, role)
        _switch_piper_role(self.teleop_cfg.right_arm_config, role)

    def connect(self) -> None:
        if self._connected:
            return
        self._switch_role(0xFC)
        self.robot.connect(calibrate=False)
        self._connected = True

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            return {}
        return self.robot.send_action(action)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self.robot.disconnect()
        finally:
            self._connected = False
            self._switch_role(0xFA)


def _make_fake_inference_leader_follower_controller(
    teleop_cfg: TeleoperatorConfig,
    *,
    enabled: bool,
    speed_ratio: int,
) -> _FakeInferenceLeaderFollowerController | None:
    if not enabled:
        return None
    robot_cfg = _make_fake_inference_leader_follower_config(teleop_cfg, speed_ratio=speed_ratio)
    if robot_cfg is None:
        return None
    return _FakeInferenceLeaderFollowerController(teleop_cfg, robot_cfg)


def _processor_pipeline_needs_observation(
    processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> bool:
    return not all(isinstance(step, IdentityProcessorStep) for step in processor.steps)


def _teleop_needs_robot_observation(
    display_data: bool,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
) -> bool:
    return (
        display_data
        or _processor_pipeline_needs_observation(teleop_action_processor)
        or _processor_pipeline_needs_observation(robot_action_processor)
    )


def _configure_robot_for_lightweight_teleop(
    robot: Robot,
    should_fetch_obs: bool,
) -> None:
    if should_fetch_obs:
        return
    robot.set_teleop_send_only_mode(True)


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    fake_inference_key: str = "",
    fake_inference_total_swing_deg: float = 120.0,
    fake_inference_period_s: float = 10.0,
    fake_inference_leader_follower_controller: _FakeInferenceLeaderFollowerController | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    last_loop_status_t = 0.0
    fake_key = fake_inference_key.strip().lower()[:1]
    fake_inference_active = False
    fake_inference_pending = False
    fake_inference_start_t = 0.0
    fake_inference_hold_action: RobotAction | None = None
    last_sent_action: RobotAction = {}
    required_fake_hold_keys = _fake_inference_required_hold_keys(robot)
    should_fetch_obs = _teleop_needs_robot_observation(
        display_data, teleop_action_processor, robot_action_processor
    )

    def _missing_fake_hold_keys() -> list[str]:
        return [hold_key for hold_key in required_fake_hold_keys if hold_key not in last_sent_action]

    def _start_fake_inference_if_ready() -> bool:
        nonlocal fake_inference_active, fake_inference_pending, fake_inference_hold_action, fake_inference_start_t

        missing_hold_keys = _missing_fake_hold_keys()
        if missing_hold_keys:
            return False

        if fake_inference_leader_follower_controller is not None:
            fake_inference_leader_follower_controller.connect()
        fake_inference_hold_action = dict(last_sent_action)
        fake_inference_start_t = time.perf_counter()
        fake_inference_active = True
        fake_inference_pending = False
        leader_mode = (
            "leaders also controlled as followers"
            if fake_inference_leader_follower_controller is not None
            else "followers only"
        )
        print(
            "\nFake inference ON: holding all joints except base sweep "
            f"(total {fake_inference_total_swing_deg:.0f} deg, {leader_mode}). "
            f"Press {fake_key} again to resume."
        )
        return True

    def _stop_fake_inference() -> None:
        nonlocal fake_inference_active, fake_inference_pending

        fake_inference_pending = False
        if fake_inference_active and fake_inference_leader_follower_controller is not None:
            fake_inference_leader_follower_controller.disconnect()
        fake_inference_active = False

    with _NonBlockingKeyReader(enabled=bool(fake_key)) as key_reader:
        try:
            while True:
                loop_start = time.perf_counter()

                # Get robot observation
                # Not really needed for now other than for visualization
                # teleop_action_processor can take None as an observation
                # given that it is the identity processor as default
                obs = robot.get_observation() if should_fetch_obs else {}

                key = key_reader.get_key()
                if key == fake_key:
                    if fake_inference_active:
                        _stop_fake_inference()
                        print("\nFake inference OFF: leaders restored to drag mode; resumed leader teleoperation.")
                    elif fake_inference_pending:
                        fake_inference_pending = False
                        print("\nFake inference request canceled; resumed leader teleoperation.")
                    else:
                        if not _start_fake_inference_if_ready():
                            fake_inference_pending = True
                            missing_hold_keys = _missing_fake_hold_keys()
                            print(
                                "\nFake inference is waiting for "
                                f"{', '.join(missing_hold_keys)}. Move the missing leader arm(s) once; "
                                f"it will start automatically. Press {fake_key} again to cancel."
                            )

                if fake_inference_active and fake_inference_hold_action is not None:
                    robot_action_to_send = _make_fake_inference_action(
                        fake_inference_hold_action,
                        elapsed_s=time.perf_counter() - fake_inference_start_t,
                        total_swing_deg=fake_inference_total_swing_deg,
                        period_s=fake_inference_period_s,
                    )
                    teleop_action = robot_action_to_send
                else:
                    # Get teleop action
                    raw_action = teleop.get_action()

                    # Process teleop action through pipeline
                    teleop_action = teleop_action_processor((raw_action, obs))

                    # Process action for robot through pipeline
                    robot_action_to_send = robot_action_processor((teleop_action, obs))

                # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
                sent_action = robot.send_action(robot_action_to_send)
                last_sent_action.update(sent_action)
                if fake_inference_pending and not fake_inference_active:
                    _start_fake_inference_if_ready()
                if fake_inference_active and fake_inference_leader_follower_controller is not None:
                    fake_inference_leader_follower_controller.send_action(robot_action_to_send)

                if display_data:
                    # Process robot observation through pipeline
                    obs_transition = robot_observation_processor(obs)

                    log_rerun_data(
                        observation=obs_transition,
                        action=teleop_action,
                        compress_images=display_compressed_images,
                    )

                    print("\n" + "-" * (display_len + 10))
                    print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                    for motor, value in robot_action_to_send.items():
                        print(f"{motor:<{display_len}} | {value:>7.2f}")
                    if sys.stdout.isatty():
                        move_cursor_up(len(robot_action_to_send) + 3)

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1 / fps - dt_s, 0.0))
                loop_s = time.perf_counter() - loop_start
                now = time.monotonic()
                if now - last_loop_status_t >= LOOP_STATUS_INTERVAL_S:
                    print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                    if sys.stdout.isatty():
                        move_cursor_up(1)
                    last_loop_status_t = now

                if duration is not None and time.perf_counter() - start >= duration:
                    return
        finally:
            _stop_fake_inference()


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    sanity_check_bimanual_piper_pair(cfg.robot, cfg.teleop)
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    should_fetch_obs = _teleop_needs_robot_observation(
        cfg.display_data, teleop_action_processor, robot_action_processor
    )
    _configure_robot_for_lightweight_teleop(robot, should_fetch_obs)
    fake_inference_leader_follower_controller = _make_fake_inference_leader_follower_controller(
        cfg.teleop,
        enabled=cfg.fake_inference_control_leaders,
        speed_ratio=cfg.fake_inference_leader_speed_ratio,
    )

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            fake_inference_key=cfg.fake_inference_key,
            fake_inference_total_swing_deg=cfg.fake_inference_total_swing_deg,
            fake_inference_period_s=cfg.fake_inference_period_s,
            fake_inference_leader_follower_controller=fake_inference_leader_follower_controller,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
