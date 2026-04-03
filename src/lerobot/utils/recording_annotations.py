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

"""Helpers for recording-time annotations and per-step policy-source tracing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"
VALID_EPISODE_SUCCESS_LABELS = {EPISODE_SUCCESS, EPISODE_FAILURE}

# RLT phase constants
PHASE_PREFIX = 0.0
PHASE_CRITICAL = 1.0

# RLT source type constants
SOURCE_VLA = 0.0
SOURCE_RL = 1.0
SOURCE_HUMAN = 2.0


def normalize_episode_success_label(label: str | None) -> str | None:
    """Normalize a user-provided episode label to canonical lowercase values."""
    if label is None:
        return None
    normalized = label.strip().lower()
    if normalized not in VALID_EPISODE_SUCCESS_LABELS:
        raise ValueError(
            f"`episode_success` must be one of {sorted(VALID_EPISODE_SUCCESS_LABELS)}, got '{label}'."
        )
    return normalized


def resolve_episode_success_label(
    explicit_label: str | None,
    default_label: str | None = None,
    require_label: bool = False,
) -> str | None:
    """Resolve the final episode-success label from explicit and default values."""
    explicit = normalize_episode_success_label(explicit_label)
    if explicit is not None:
        return explicit

    default = normalize_episode_success_label(default_label)
    if default is not None:
        return default

    if require_label:
        raise ValueError(
            "Missing `episode_success` label. Use success/failure hotkeys or set `default_episode_success`."
        )
    return None


def infer_collector_policy_id(policy_cfg: Any | None) -> str:
    """Infer a stable policy identifier for frame-level provenance."""
    if policy_cfg is None:
        return "human"

    pretrained_path = getattr(policy_cfg, "pretrained_path", None)
    if pretrained_path:
        return str(pretrained_path)

    policy_type = getattr(policy_cfg, "type", None)
    if policy_type:
        return str(policy_type)

    return "policy"


def infer_collector_policy_version(policy_cfg: Any | None) -> str:
    """Infer a compact policy-version string suitable for metadata columns."""
    if policy_cfg is None:
        return "human"

    pretrained_path = getattr(policy_cfg, "pretrained_path", None)
    if pretrained_path:
        return Path(str(pretrained_path)).name

    policy_type = getattr(policy_cfg, "type", None)
    if policy_type:
        return str(policy_type)

    return "policy"


def resolve_collector_policy_id(
    *,
    intervention_enabled: bool,
    is_intervention: bool,
    selected_from_policy: bool,
    policy_id: str,
    human_id: str,
) -> str:
    """Resolve frame-level `collector_policy_id` from control mode and source."""
    if intervention_enabled:
        return human_id if is_intervention else policy_id
    return policy_id if selected_from_policy else human_id
