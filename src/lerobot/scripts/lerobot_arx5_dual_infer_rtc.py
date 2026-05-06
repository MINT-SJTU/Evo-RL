#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Dual-arm Pi0.5 on two ARX5 arms with asynchronous Real-Time Chunking (RTC).

Reuses setup from :mod:`lerobot.scripts.lerobot_arx5_dual_infer`; control reads from
:class:`~lerobot.policies.rtc.action_queue.ActionQueue` while a thread runs RTC inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.scripts.lerobot_arx5_infer import (
    KeyboardListener,
    LoopState,
    TrainRawEpisodeRecorder,
    _discover_next_round_index,
    _list_realsense_cameras,
    _load_policy_bundle,
)
from lerobot.scripts.lerobot_arx5_dual_infer import (
    DEFAULT_LEFT_CAN_PORT,
    DEFAULT_POLICY_PATH,
    DEFAULT_RIGHT_CAN_PORT,
    DUAL_STATE_DIM,
    EndEffectorTrajectoryRecorder,
    JointTrajectoryRecorder,
    JointVelocityRecorder,
    STATE_KEYS,
    _build_dataset_features,
    _build_dual_observation,
    _clip_safe_actions,
    _home_dual_arms_after_camera_failure,
    _log_keyboard_help,
    _make_camera_configs,
    _parse_camera_specs,
    _required_image_slot_names,
    _run_keyboard_command,
    _run_vr_teleop_session,
    _split_dual_action,
    _visual_image_slot_names,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


class _RTCSync:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.observation: dict[str, Any] | None = None
        self.inference_ready = threading.Condition(self.lock)


def apply_pi05_rtc_for_inference(
    policy: PI05Policy,
    *,
    max_guidance_weight: float | None,
    rtc_debug: bool,
) -> None:
    weight = float(max_guidance_weight if max_guidance_weight is not None else 10.0)
    cfg = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=weight,
        execution_horizon=10,
        debug=bool(rtc_debug),
        debug_maxlen=100,
    )
    policy.config.rtc_config = cfg
    policy.init_rtc_processor()


def _predict_action_chunk_tensor_rtc(
    *,
    robot_observation: dict[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    policy: PI05Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    use_amp: bool,
    inference_delay: int,
    prev_chunk_left_over: Tensor | None,
    execution_horizon: int,
) -> Tensor:
    observation_frame = build_dataset_frame(dataset_features, robot_observation, prefix=OBS_STR)
    processed_observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task=task,
        robot_type=robot_type,
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        processed_observation = preprocessor(processed_observation)
        action_chunk_raw = policy.predict_action_chunk(
            processed_observation,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )
        return postprocessor(action_chunk_raw)


def _tensor_row_to_action_dict(row: Tensor, dataset_features: dict[str, dict[str, Any]]) -> dict[str, float]:
    action_names = dataset_features[ACTION]["names"]
    return {name: float(row[offset]) for offset, name in enumerate(action_names)}


def _clamp_d_used_for_chunk(d_candidate: int, time_steps: int) -> tuple[int, bool]:
    if time_steps <= 0:
        return d_candidate, False
    max_delay = time_steps - 1
    if d_candidate > max_delay:
        return max_delay, True
    return d_candidate, False


def _execution_horizon_kwarg(*, horizon: int, s_steps: int, d_used: int) -> int:
    H = horizon
    raw_end = H - int(s_steps)
    end_kw = max(int(d_used), min(raw_end, H))
    end_kw = max(0, min(end_kw, H))
    return int(end_kw)


def _inference_thread_fn(
    *,
    sync: _RTCSync,
    action_queue: ActionQueue,
    policy: PI05Policy,
    dataset_features: dict[str, dict[str, Any]],
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    use_amp: bool,
    policy_cfg: PreTrainedConfig,
    delay_min: int,
    delay_hist: deque[int],
    s_min: int,
    stop_event: threading.Event,
    inference_pause: threading.Event,
) -> None:
    horizon = int(policy_cfg.chunk_size)
    robot_type = "arx5_dual"

    while not stop_event.is_set():
        if inference_pause.is_set():
            time.sleep(0.02)
            continue

        with sync.lock:
            while action_queue.get_action_index() < s_min and not stop_event.is_set():
                sync.inference_ready.wait(timeout=0.05)
            if stop_event.is_set():
                break
            idx_start = action_queue.get_action_index()
            prev_left = action_queue.get_left_over()

            with sync.obs_lock:
                obs = sync.observation
            if obs is None:
                sync.inference_ready.wait(timeout=0.05)
                continue

        hist_max = max(delay_hist) if len(delay_hist) > 0 else 0
        d_raw = max(int(delay_min), int(hist_max))
        d_used, clamped_d = _clamp_d_used_for_chunk(d_raw, horizon)
        if clamped_d:
            logger.warning(
                "RTC: clamped inference_delay/real_delay from %d to %d (chunk_len=%d) to keep merge non-empty.",
                d_raw,
                d_used,
                horizon,
            )

        end_kw = _execution_horizon_kwarg(horizon=horizon, s_steps=idx_start, d_used=d_used)

        prev_arg: Tensor | None = None
        if prev_left is not None and prev_left.numel() > 0 and prev_left.shape[0] > 0:
            prev_arg = prev_left.unsqueeze(0).to(device=device, dtype=torch.float32)

        t_infer0 = time.perf_counter()
        orig = _predict_action_chunk_tensor_rtc(
            robot_observation=obs,
            dataset_features=dataset_features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=task,
            robot_type=robot_type,
            use_amp=use_amp,
            inference_delay=int(d_used),
            prev_chunk_left_over=prev_arg,
            execution_horizon=int(end_kw),
        )
        proc = orig

        with sync.lock:
            idx_end = action_queue.get_action_index()
            d_meas = int(idx_end - idx_start)
            delay_hist.append(d_meas)

            action_queue.merge(
                original_actions=orig.squeeze(0).detach().cpu(),
                processed_actions=proc.squeeze(0).detach().cpu(),
                real_delay=int(d_used),
                action_index_before_inference=idx_start,
            )
            remaining = action_queue.qsize()
            if remaining <= 0:
                raise RuntimeError(
                    "RTC: merge produced an empty ActionQueue; check delay settings and chunk length."
                )
            sync.inference_ready.notify_all()

        t_infer1 = time.perf_counter()
        logger.info(
            "RTC inference: d_used=%d d_meas=%d idx %d→%d exec_horizon_kw=%d wall=%.4fs qsize=%d",
            d_used,
            d_meas,
            idx_start,
            idx_end,
            end_kw,
            t_infer1 - t_infer0,
            remaining,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-arm Pi0.5 + asynchronous RTC on two ARX5 arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--left-can-port", type=str, default=DEFAULT_LEFT_CAN_PORT)
    parser.add_argument("--right-can-port", type=str, default=DEFAULT_RIGHT_CAN_PORT)
    parser.add_argument("--arm-type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use-stub", action="store_true")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "base:254522071216",
            "left_wrist:150622073629",
            "right_wrist:409122272986",
        ],
    )
    parser.add_argument("--use-usb-cams", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-cameras", nargs="*", default=[])
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--safe-mode", action="store_true", default=False)
    parser.add_argument("--max-joint-step", type=float, default=0.02)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument(
        "--vr-scale-factor",
        type=float,
        default=1.5,
        help="VR teleop motion scale (when using [V]).",
    )
    parser.add_argument(
        "--vr-no-camera",
        action="store_true",
        help="VR teleop without RealSense threads (arm-only IK).",
    )
    parser.add_argument(
        "--vr-no-camera-display",
        action="store_true",
        help="In VR teleop, do not open OpenCV camera preview windows.",
    )
    parser.add_argument("--vr-control-hz", type=int, default=50, help="VR IK / control loop rate.")
    parser.add_argument(
        "--vr-visualize-placo",
        action="store_true",
        help="Open Placo / MeshCat during VR teleop.",
    )
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument("--end-traj-dir", type=Path, default=None)
    parser.add_argument("--joint-traj-dir", type=Path, default=None)
    parser.add_argument("--joint-vel-dir", type=Path, default=None)
    parser.add_argument("--raw-train-record-dir", type=Path, default=None)
    parser.add_argument("--protect-on-disconnect", action="store_true", default=True)
    parser.add_argument("--no-protect-on-disconnect", dest="protect_on_disconnect", action="store_false")

    parser.add_argument(
        "--rtc-s-min",
        type=int,
        default=10,
        help="Min steps consumed since merge before the next inference may run.",
    )
    parser.add_argument(
        "--rtc-inference-delay-min",
        type=int,
        default=1,
        help="Lower bound: d_used = max(this, max(delay history)).",
    )
    parser.add_argument("--rtc-delay-history-maxlen", type=int, default=64)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
    parser.add_argument("--rtc-debug", action="store_true")

    args = parser.parse_args()
    if args.list_cameras:
        _list_realsense_cameras()
        return
    if not args.task:
        parser.error("除非使用 --list-cameras，否则必须提供 --task。")
    if args.safe_mode and args.no_keyboard:
        parser.error("安全模式需要键盘控制。")
    if args.rtc_s_min < 1:
        parser.error("--rtc-s-min 必须 >= 1。")
    if args.rtc_inference_delay_min < 0:
        parser.error("--rtc-inference-delay-min 必须为非负数。")
    if args.rtc_delay_history_maxlen < 1:
        parser.error("--rtc-delay-history-maxlen 必须 >= 1。")

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    state_feature = None if policy_cfg.input_features is None else policy_cfg.input_features.get(f"{OBS_STR}.state")
    if state_feature is None or state_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects observation.state dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(state_feature, 'shape', None)}."
        )
    action_feature = None if policy_cfg.output_features is None else policy_cfg.output_features.get(ACTION)
    if action_feature is None or action_feature.shape[0] != DUAL_STATE_DIM:
        raise ValueError(
            f"Dual-arm runtime expects action dim={DUAL_STATE_DIM}, "
            f"but checkpoint declares {getattr(action_feature, 'shape', None)}."
        )

    required_image_slot_names = _required_image_slot_names(policy_cfg)
    camera_specs = _parse_camera_specs(args.cameras)
    extra_specs = sorted(set(camera_specs) - set(required_image_slot_names))
    if extra_specs:
        logger.warning("Ignoring --cameras entries not consumed by the runtime: %s", extra_specs)

    camera_configs = _make_camera_configs(
        required_image_slot_names=required_image_slot_names,
        camera_specs=camera_specs,
        use_usb_cams=args.use_usb_cams,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
        flipped_cameras=set(args.flip_cameras),
    )
    cameras = make_cameras_from_configs(camera_configs)
    camera_names = list(cameras)

    from lerobot.robots.arx5_follower.arx5_client import ARX5ArmClient

    left_arm = ARX5ArmClient(
        can_port=args.left_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/left_recorded_pose.json"),
    )
    right_arm = ARX5ArmClient(
        can_port=args.right_can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
        recorded_pose_path=Path("checkpoints/right_recorded_pose.json"),
    )

    policy_cfg, policy, preprocessor, postprocessor, device = _load_policy_bundle(
        policy_path=args.policy_path,
        device_override=args.policy_device,
        stats_path=args.stats_path,
        policy_cfg=policy_cfg,
    )
    if not isinstance(policy, PI05Policy):
        raise ValueError("This script requires a Pi0.5 (pi05) policy checkpoint.")

    apply_pi05_rtc_for_inference(
        policy,
        max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_debug=bool(args.rtc_debug),
    )

    dataset_features = _build_dataset_features(camera_configs)
    horizon = int(policy_cfg.chunk_size)

    action_queue = ActionQueue(policy.config.rtc_config)

    raw_recorder: TrainRawEpisodeRecorder | None = None
    if args.raw_train_record_dir is not None:
        raw_recorder = TrainRawEpisodeRecorder(
            raw_record_root=args.raw_train_record_dir,
            action_keys=STATE_KEYS,
            camera_names=camera_names,
            task=args.task,
            dt_s=args.duration,
            image_save_size_hw=(args.cam_height, args.cam_width),
        )
    end_traj_recorder: EndEffectorTrajectoryRecorder | None = None
    if args.end_traj_dir is not None:
        end_traj_recorder = EndEffectorTrajectoryRecorder(args.end_traj_dir)
    joint_traj_recorder: JointTrajectoryRecorder | None = None
    if args.joint_traj_dir is not None:
        joint_traj_recorder = JointTrajectoryRecorder(args.joint_traj_dir)
    joint_vel_recorder: JointVelocityRecorder | None = None
    if args.joint_vel_dir is not None:
        joint_vel_recorder = JointVelocityRecorder(args.joint_vel_dir)

    keyboard = None if args.no_keyboard else KeyboardListener()
    state = LoopState.RUNNING if keyboard is None else LoopState.STOPPED
    request_next_chunk = keyboard is None or not args.safe_mode
    inference_pause = threading.Event()

    _ = _discover_next_round_index(args.record_dir) if args.record_dir is not None else 0
    running = True
    camera_failure: Exception | None = None
    segment_step_count = 0
    safe_segment_len = int(policy_cfg.n_action_steps)

    sync = _RTCSync()
    stop_event = threading.Event()
    delay_hist: deque[int] = deque(maxlen=int(args.rtc_delay_history_maxlen))
    delay_hist.append(0)

    try:
        for name, camera in cameras.items():
            logger.info("Connecting camera '%s'", name)
            camera.connect()
            logger.info("Camera '%s' connected.", name)

        if keyboard is not None:
            left_arm.go_home()
            right_arm.go_home()
            time.sleep(2.0)
            left_arm.hold_position()
            right_arm.hold_position()
            _log_keyboard_help(args.safe_mode)

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        observation0 = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
        with sync.obs_lock:
            sync.observation = observation0

        bootstrap_orig = _predict_action_chunk_tensor_rtc(
            robot_observation=observation0,
            dataset_features=dataset_features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=args.task,
            robot_type="arx5_dual",
            use_amp=bool(policy_cfg.use_amp),
            inference_delay=0,
            prev_chunk_left_over=None,
            execution_horizon=_execution_horizon_kwarg(horizon=horizon, s_steps=0, d_used=0),
        )
        action_queue.merge(
            original_actions=bootstrap_orig.squeeze(0).detach().cpu(),
            processed_actions=bootstrap_orig.squeeze(0).detach().cpu(),
            real_delay=0,
            action_index_before_inference=0,
        )
        if action_queue.qsize() <= 0:
            raise RuntimeError("RTC bootstrap produced an empty queue.")
        logger.info("RTC bootstrap merge OK, qsize=%d", action_queue.qsize())

        worker = threading.Thread(
            target=_inference_thread_fn,
            kwargs={
                "sync": sync,
                "action_queue": action_queue,
                "policy": policy,
                "dataset_features": dataset_features,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "device": device,
                "task": args.task,
                "use_amp": bool(policy_cfg.use_amp),
                "policy_cfg": policy_cfg,
                "delay_min": int(args.rtc_inference_delay_min),
                "delay_hist": delay_hist,
                "s_min": int(args.rtc_s_min),
                "stop_event": stop_event,
                "inference_pause": inference_pause,
            },
            daemon=True,
            name="rtc-inference",
        )
        worker.start()

        while running:
            if keyboard is not None:
                previous_state = state
                key = keyboard.get_key()
                state, request_next_chunk, running, vr_req = _run_keyboard_command(
                    key=key,
                    left_arm=left_arm,
                    right_arm=right_arm,
                    state=state,
                    safe_mode=args.safe_mode,
                    request_next_chunk=request_next_chunk,
                    recorder=raw_recorder,
                    end_traj_recorder=end_traj_recorder,
                    joint_traj_recorder=joint_traj_recorder,
                    joint_vel_recorder=joint_vel_recorder,
                )
                if vr_req:
                    stop_event.set()
                    worker.join(timeout=3.0)
                    left_arm, right_arm = _run_vr_teleop_session(
                        left_arm=left_arm,
                        right_arm=right_arm,
                        cameras=cameras,
                        camera_names=camera_names,
                        camera_specs=camera_specs,
                        keyboard=keyboard,
                        args=args,
                        recorder=raw_recorder,
                    )
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    state = LoopState.STOPPED
                    request_next_chunk = False if args.safe_mode else True
                    stop_event.clear()
                    worker = threading.Thread(
                        target=_inference_thread_fn,
                        kwargs={
                            "sync": sync,
                            "action_queue": action_queue,
                            "policy": policy,
                            "dataset_features": dataset_features,
                            "preprocessor": preprocessor,
                            "postprocessor": postprocessor,
                            "device": device,
                            "task": args.task,
                            "use_amp": bool(policy_cfg.use_amp),
                            "policy_cfg": policy_cfg,
                            "delay_min": int(args.rtc_inference_delay_min),
                            "delay_hist": delay_hist,
                            "s_min": int(args.rtc_s_min),
                            "stop_event": stop_event,
                            "inference_pause": inference_pause,
                        },
                        daemon=True,
                        name="rtc-inference",
                    )
                    worker.start()
                    continue
                if previous_state != LoopState.RUNNING and state == LoopState.RUNNING:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
            if not running:
                break
            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if args.safe_mode and not request_next_chunk:
                inference_pause.set()
                time.sleep(0.05)
                continue

            inference_pause.clear()

            try:
                observation = _build_dual_observation(left_arm=left_arm, right_arm=right_arm, cameras=cameras)
            except (DeviceNotConnectedError, TimeoutError, RuntimeError) as error:
                camera_failure = error
                _home_dual_arms_after_camera_failure(left_arm=left_arm, right_arm=right_arm, error=error)
                state = LoopState.STOPPED
                running = False
                break

            with sync.obs_lock:
                sync.observation = observation

            with sync.lock:
                sync.inference_ready.notify_all()

            step_start = time.perf_counter()
            row = action_queue.get()
            if row is None:
                raise RuntimeError(
                    "RTC: action queue is empty on control step; violates non-starvation assumptions."
                )

            action_dict = _tensor_row_to_action_dict(row, dataset_features)
            current_state = [float(observation[key]) for key in STATE_KEYS]
            if args.safe_mode:
                clipped = _clip_safe_actions(
                    [action_dict],
                    current_state=current_state,
                    max_joint_step=args.max_joint_step,
                )
                action_dict = clipped[0]

            left_joint, right_joint = _split_dual_action(action_dict)
            left_thread = threading.Thread(target=left_arm.send_joint, args=(left_joint,))
            right_thread = threading.Thread(target=right_arm.send_joint, args=(right_joint,))
            left_thread.start()
            right_thread.start()
            left_thread.join()
            right_thread.join()

            segment_step_count += 1
            if args.safe_mode and segment_step_count >= safe_segment_len:
                request_next_chunk = False
                segment_step_count = 0

            step_elapsed = time.perf_counter() - step_start
            precise_sleep(max(args.duration - step_elapsed, 0.0))

        stop_event.set()
        with sync.lock:
            sync.inference_ready.notify_all()
        worker.join(timeout=5.0)

    finally:
        if raw_recorder is not None:
            raw_recorder.finish_active_segment()
        if keyboard is not None:
            keyboard.restore()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if args.protect_on_disconnect:
            left_arm.protect_mode()
            right_arm.protect_mode()

    if camera_failure is not None:
        logger.error("因相机故障终止：%s", camera_failure)
        sys.exit(1)


if __name__ == "__main__":
    main()
