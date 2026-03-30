#!/usr/bin/env python3
"""Deploy RLT policy on a real robot.

Usage:
    python scripts/deploy_rlt.py \
        --vla-model /path/to/pi05_base \
        --rl-token-ckpt /path/to/demo_adapt_checkpoint.pt \
        --ac-ckpt /path/to/rl_checkpoint_best.pt \
        --task "screw the bolt" \
        --phase-mode always_rl

Controls:
    q - quit
    r - reset episode
    v - switch to VLA phase (manual mode)
    c - switch to RL/critical phase (manual mode)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from lerobot.rlt.deploy import load_rlt_deploy_policy
from lerobot.rlt.deploy_config import DeployConfig

log = logging.getLogger(__name__)

# SO101 bilateral: 6 joints per arm × 2 = 12 DOF
_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_PROPRIO_KEYS = [f"left_{j}.pos" for j in _JOINT_NAMES] + [f"right_{j}.pos" for j in _JOINT_NAMES]
DEFAULT_ACTION_KEYS = DEFAULT_PROPRIO_KEYS  # same joint names for action
DEFAULT_CAMERA_KEYS = ["left_wrist", "right_wrist", "right_front"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy RLT policy on robot")
    p.add_argument("--vla-model", type=str, default="lerobot/pi05_base")
    p.add_argument("--rl-token-ckpt", type=str, default="")
    p.add_argument("--ac-ckpt", type=str, default="")
    p.add_argument("--task", type=str, default="Insert the copper screw into the black sleeve.")
    p.add_argument("--phase-mode", type=str, default="always_rl",
                    choices=["always_rl", "always_vla", "manual"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--token-pool-size", type=int, default=64)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", action="store_true", default=False)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=3000,
                    help="Max steps per episode before auto-reset")
    p.add_argument("--log-level", type=str, default="INFO")
    # Robot hardware
    p.add_argument("--left-port", type=str, default="/dev/ttyACM3")
    p.add_argument("--right-port", type=str, default="/dev/ttyACM2")
    p.add_argument("--left-cam", type=str,
                    default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0")
    p.add_argument("--right-cam", type=str,
                    default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4:1.0-video-index0")
    p.add_argument("--front-cam-serial", type=str, default="152122079296",
                    help="Intel RealSense serial number for front camera")
    return p.parse_args()


def _try_read_key() -> str | None:
    """Non-blocking single character read from stdin."""
    import select
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def run_control_loop(policy, robot, args: argparse.Namespace) -> None:
    """Main control loop: obs -> policy -> action at control_hz."""
    dt_target = 1.0 / args.control_hz
    episode = 0
    step = 0
    total_steps = 0

    log.info("Starting control loop at %.1f Hz (dt=%.1f ms)", args.control_hz, dt_target * 1000)
    log.info("Controls: q=quit, r=reset, v=vla_phase, c=critical_phase")

    policy.reset()

    while True:
        t_loop_start = time.monotonic()

        # Check keyboard input
        key = _try_read_key()
        if key == "q":
            log.info("Quit requested")
            break
        if key == "r":
            log.info("Episode reset requested")
            policy.reset()
            episode += 1
            step = 0
            continue
        if key == "v":
            policy.phase_controller.trigger_vla()
            log.info("Switched to VLA phase")
        if key == "c":
            policy.phase_controller.trigger_critical()
            log.info("Switched to critical/RL phase")

        # Get observation
        obs_dict = robot.get_observation()

        # Compute action
        t_action_start = time.monotonic()
        action = policy.select_action(obs_dict)
        action_time_ms = (time.monotonic() - t_action_start) * 1000

        # Convert tensor to named action dict and send
        from lerobot.rlt.obs_bridge import rlt_action_to_robot_action
        robot_action = rlt_action_to_robot_action(action, DEFAULT_ACTION_KEYS)
        robot.send_action(robot_action)

        step += 1
        total_steps += 1

        # Log timing periodically
        if step % 100 == 0:
            timing = policy.timing
            chunk_ms = timing.get("last_chunk_compute_ms", 0.0)
            log.info(
                "ep=%d step=%d total=%d | action=%.1fms chunk=%.1fms",
                episode, step, total_steps, action_time_ms, chunk_ms,
            )

        # Auto-reset if max steps reached
        if step >= args.max_steps:
            log.info("Max steps %d reached, resetting episode", args.max_steps)
            policy.reset()
            episode += 1
            step = 0

        # Rate limiting
        elapsed = time.monotonic() - t_loop_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    deterministic = not args.stochastic

    config = DeployConfig(
        vla_model_path=args.vla_model,
        rl_token_checkpoint=args.rl_token_ckpt,
        ac_checkpoint=args.ac_ckpt,
        camera_keys=DEFAULT_CAMERA_KEYS,
        proprio_keys=DEFAULT_PROPRIO_KEYS,
        action_keys=DEFAULT_ACTION_KEYS,
        phase_mode=args.phase_mode,
        deterministic=deterministic,
        device=args.device,
        task_instruction=args.task,
        token_pool_size=args.token_pool_size,
        chunk_length=args.chunk_length,
    )

    log.info("Building RLT deploy policy...")
    policy = load_rlt_deploy_policy(config)

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollowerConfig

    left_cfg = SOFollowerConfig(
        port=args.left_port,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.left_cam,
                width=640, height=480, fps=30, fourcc="MJPG",
            ),
        },
    )
    right_cfg = SOFollowerConfig(
        port=args.right_port,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.right_cam,
                width=640, height=480, fps=30, fourcc="MJPG",
            ),
            "front": RealSenseCameraConfig(
                serial_number_or_name=args.front_cam_serial,
                width=640, height=480, fps=30, warmup_s=2,
            ),
        },
    )
    robot_cfg = BiSOFollowerConfig(
        left_arm_config=left_cfg,
        right_arm_config=right_cfg,
        id="bi_so101_follower",
    )
    robot = BiSOFollower(config=robot_cfg)

    log.info("Connecting to robot...")
    robot.connect()

    run_control_loop(policy, robot, args)

    robot.disconnect()
    log.info("Done. Total timing stats: %s", policy.timing)


if __name__ == "__main__":
    main()
