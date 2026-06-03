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

"""PI05 RTC + async inference helpers for ARX5 dual-arm RTC scripts."""

from __future__ import annotations

import collections
import threading
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STR


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


def _clamp_d_used_for_chunk(d_candidate: int, time_steps: int) -> tuple[int, bool]:
    if time_steps <= 0:
        return d_candidate, False
    max_delay = time_steps - 1
    if d_candidate > max_delay:
        return max_delay, True
    return d_candidate, False


def _execution_horizon_kwarg(*, horizon: int, s_steps: int, d_used: int) -> int:
    raw_end = int(horizon) - int(s_steps)
    end_kw = max(int(d_used), min(raw_end, horizon))
    return int(max(0, min(end_kw, horizon)))


def _predict_pi05_action_chunk_rtc(
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
    prev_chunk_left_over: torch.Tensor | None,
    execution_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (postprocessed chunk for the robot, raw model output before postprocessor)."""
    observation_frame = build_dataset_frame(dataset_features, robot_observation, prefix=OBS_STR)
    processed_observation = prepare_observation_for_inference(
        dict(observation_frame),
        device,
        task=task,
        robot_type=robot_type,
    )
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        processed_observation = preprocessor(processed_observation)
        action_chunk_raw = policy.predict_action_chunk(
            processed_observation,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )
        return postprocessor(action_chunk_raw), action_chunk_raw.detach()


def _prev_chunk_left_over_from_raw_tail(
    tail_cpu: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tail_cpu is None or tail_cpu.numel() == 0 or tail_cpu.shape[0] == 0:
        return None
    return tail_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)


def _chunk_tensor_to_action_dict_list(
    chunk: torch.Tensor,
    dataset_features: dict[str, dict[str, Any]],
) -> list[dict[str, float]]:
    action_names = dataset_features[ACTION]["names"]
    action_chunk = chunk.squeeze(0).to("cpu")
    horizon = int(action_chunk.shape[0])
    actions: list[dict[str, float]] = []
    for index in range(horizon):
        row = action_chunk[index]
        actions.append({name: float(row[offset]) for offset, name in enumerate(action_names)})
    return actions


def _copy_observation_for_predict(observation: dict[str, Any]) -> dict[str, Any]:
    """Copy numpy image arrays so the worker can GPU-predict without racing the main-thread live obs."""
    out: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            out[key] = np.array(value)
        else:
            out[key] = value
    return out


class _AsyncInferCtl:
    def __init__(self, s_min: int, rtc_delay_buffer_b: int = 8, rtc_d_init: int = 0) -> None:
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.s_min = s_min
        self.actions_cur: list[dict[str, float]] = []
        self.chunk_cur_raw: torch.Tensor | None = None
        self.idx = 0
        self.obs_cur: dict[str, Any] = {}
        self.chunk_serial = 1
        self.async_rtc_fatal: str | None = None
        self.rtc_d_init = rtc_d_init
        self.rtc_delay_q: collections.deque[int] = collections.deque(maxlen=rtc_delay_buffer_b)
