#!/usr/bin/env python

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.dataset as pa_ds

logger = logging.getLogger(__name__)

EPISODE_INDEX_FIELD = "episode_index"
IS_INTERVENTION_FIELD = "complementary_info.is_intervention"
COLLECTOR_POLICY_ID_FIELD = "complementary_info.collector_policy_id"

TRAJECTORY_AUTONOMOUS = "autonomous"
TRAJECTORY_INTERVENTION = "intervention"
TRAJECTORY_EXPERT = "expert"

_POLICY_TOKENS = ("policy", "model", "autonomous")
_HUMAN_TOKENS = ("human", "expert", "vr", "teleop")


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        return _to_float(value[0]) if value else 0.0
    return float(value)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple)):
        return _to_text(value[0]) if value else ""
    return str(value)


def _batch_column(record_batch, name: str):
    return record_batch.column(record_batch.schema.names.index(name))


def build_episode_trajectory_types(root: str | Path) -> dict[int, str]:
    """Infer lightweight episode-level trajectory types from local parquet data.

    Expert episodes are detected as episodes collected entirely by a human-like collector and without any
    policy/model/autonomous collector frames. Mixed episodes with intervention frames are marked as
    intervention. Everything else is autonomous/model data.
    """
    root = Path(root)
    data_dir = root / "data"
    if not data_dir.exists():
        logger.warning("Trajectory type scan skipped; data directory not found: %s", data_dir)
        return {}

    data_dataset = pa_ds.dataset(data_dir, format="parquet")
    schema_names = set(data_dataset.schema.names)
    if EPISODE_INDEX_FIELD not in schema_names:
        logger.warning("Trajectory type scan skipped for %s; missing column: %s", root, EPISODE_INDEX_FIELD)
        return {}

    columns = [EPISODE_INDEX_FIELD]
    has_intervention = IS_INTERVENTION_FIELD in schema_names
    has_collector = COLLECTOR_POLICY_ID_FIELD in schema_names
    if has_intervention:
        columns.append(IS_INTERVENTION_FIELD)
    if has_collector:
        columns.append(COLLECTOR_POLICY_ID_FIELD)

    episode_counts: dict[int, Counter[str]] = defaultdict(Counter)
    scanner = data_dataset.scanner(columns=columns, batch_size=65536)
    for record_batch in scanner.to_batches():
        episode_indices = _batch_column(record_batch, EPISODE_INDEX_FIELD).to_pylist()
        interventions = (
            _batch_column(record_batch, IS_INTERVENTION_FIELD).to_pylist()
            if has_intervention
            else [0.0] * len(episode_indices)
        )
        collectors = (
            _batch_column(record_batch, COLLECTOR_POLICY_ID_FIELD).to_pylist()
            if has_collector
            else [""] * len(episode_indices)
        )

        for episode_raw, intervention_raw, collector_raw in zip(
            episode_indices,
            interventions,
            collectors,
            strict=True,
        ):
            episode_index = int(episode_raw)
            counts = episode_counts[episode_index]
            counts["frames"] += 1
            if _to_float(intervention_raw) > 0.5:
                counts["intervention"] += 1

            collector = _to_text(collector_raw).strip().lower()
            if collector:
                if any(token in collector for token in _POLICY_TOKENS):
                    counts["policy_collector"] += 1
                if any(token in collector for token in _HUMAN_TOKENS):
                    counts["human_collector"] += 1

    trajectory_types: dict[int, str] = {}
    for episode_index, counts in episode_counts.items():
        if counts["human_collector"] > 0 and counts["policy_collector"] == 0:
            trajectory_type = TRAJECTORY_EXPERT
        elif counts["intervention"] > 0:
            trajectory_type = TRAJECTORY_INTERVENTION
        else:
            trajectory_type = TRAJECTORY_AUTONOMOUS
        trajectory_types[int(episode_index)] = trajectory_type

    return trajectory_types

