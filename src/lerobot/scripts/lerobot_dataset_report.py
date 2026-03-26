#!/usr/bin/env python

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
Generate a structured report for a LeRobot dataset.

Examples:

```bash
lerobot-dataset-report --dataset local/eval_twl2_10_hil_auto
lerobot-dataset-report --dataset ~/.cache/huggingface/lerobot/local/eval_twl2_10_hil_auto --json
```
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as pa_ds

from lerobot.datasets.utils import load_episodes, load_info, load_tasks
from lerobot.utils.constants import HF_LEROBOT_HOME


def resolve_dataset_root(dataset: str, root: Path | None) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base_root = root.expanduser().resolve() if root is not None else HF_LEROBOT_HOME.resolve()
    candidates = [base_root / dataset]
    if "/" not in dataset.strip("/"):
        candidates.append(base_root / "local" / dataset)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Dataset path not found. Searched:\n{searched}")


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return _to_float(value[0]) if len(value) > 0 else 0.0
    if value is None:
        return 0.0
    return float(value)


def _normalize_label(value: Any) -> str:
    if value is None:
        return "unlabeled"
    label = str(value).strip().lower()
    if label == "":
        return "unlabeled"
    return label


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _flatten_episode_tasks(task_value: Any) -> list[str]:
    if hasattr(task_value, "tolist") and not isinstance(task_value, (str, bytes)):
        return _flatten_episode_tasks(task_value.tolist())
    if isinstance(task_value, list):
        values: list[str] = []
        for item in task_value:
            values.extend(_flatten_episode_tasks(item))
        return values
    if task_value is None:
        return []
    return [str(task_value)]


def _build_episode_length_histogram(lengths: list[float], bins: int = 20) -> list[dict[str, float | int]]:
    if bins <= 0:
        raise ValueError(f"bins must be > 0, got {bins}")
    if not lengths:
        return [{"start": 0.0, "end": 0.0, "count": 0} for _ in range(bins)]

    min_len = float(min(lengths))
    max_len = float(max(lengths))
    span = max_len - min_len

    histogram = []
    if span == 0.0:
        histogram.append({"start": min_len, "end": max_len, "count": len(lengths)})
        histogram.extend({"start": min_len, "end": max_len, "count": 0} for _ in range(bins - 1))
        return histogram

    bin_width = span / bins
    for i in range(bins):
        start = min_len + i * bin_width
        end = min_len + (i + 1) * bin_width if i < bins - 1 else max_len
        histogram.append({"start": start, "end": end, "count": 0})

    for length in lengths:
        idx = min(int((float(length) - min_len) / bin_width), bins - 1)
        histogram[idx]["count"] = int(histogram[idx]["count"]) + 1

    return histogram


def _format_ascii_histogram(histogram: list[dict[str, float | int]], bar_width: int = 40) -> list[str]:
    max_count = max((bin_info["count"] for bin_info in histogram), default=0)
    lines: list[str] = []
    for bin_info in histogram:
        count = int(bin_info["count"])
        start = float(bin_info["start"])
        end = float(bin_info["end"])
        filled = 0 if max_count == 0 else int(round((count / max_count) * bar_width))
        bar = "#" * filled
        lines.append(f"- [{start:>7.2f}s, {end:>7.2f}s] | {count:>5} | {bar}")
    return lines


def build_report(dataset_root: Path) -> dict[str, Any]:
    info = load_info(dataset_root)
    episodes_ds = load_episodes(dataset_root)
    episodes_df = episodes_ds.to_pandas()

    actual_episode_count = int(len(episodes_df))
    episode_success_labels = (
        episodes_df["episode_success"].tolist() if "episode_success" in episodes_df else []
    )
    normalized_labels = [_normalize_label(v) for v in episode_success_labels]

    success_count = sum(1 for v in normalized_labels if v == "success")
    failure_count = sum(1 for v in normalized_labels if v == "failure")
    unlabeled_count = sum(1 for v in normalized_labels if v not in {"success", "failure"})
    labeled_count = success_count + failure_count

    data_dataset = pa_ds.dataset(dataset_root / "data", format="parquet")
    data_schema_names = set(data_dataset.schema.names)
    actual_frame_count = int(data_dataset.count_rows())

    intervention_col = "complementary_info.is_intervention"
    episode_index_col = "episode_index"
    intervention_frames = 0
    intervention_episode_count = 0

    if (
        intervention_col in data_schema_names
        and episode_index_col in data_schema_names
        and actual_frame_count > 0
    ):
        table = data_dataset.to_table(columns=[episode_index_col, intervention_col])
        episode_indices = table[episode_index_col].to_pylist()
        intervention_values = table[intervention_col].to_pylist()

        intervention_episode_ids = set()
        for ep_idx, raw_value in zip(episode_indices, intervention_values, strict=True):
            if _to_float(raw_value) > 0.0:
                intervention_frames += 1
                intervention_episode_ids.add(int(ep_idx))
        intervention_episode_count = len(intervention_episode_ids)

    tasks_df = load_tasks(dataset_root)
    unique_tasks: list[str] = []
    seen_tasks: set[str] = set()
    if "tasks" in episodes_df.columns:
        for row_tasks in episodes_df["tasks"].tolist():
            for task in _flatten_episode_tasks(row_tasks):
                if task not in seen_tasks:
                    seen_tasks.add(task)
                    unique_tasks.append(task)

    feature_rows = []
    for key in sorted(info["features"].keys()):
        feature = info["features"][key]
        feature_rows.append(
            {
                "name": key,
                "dtype": feature.get("dtype"),
                "shape": list(feature.get("shape", ())),
                "names_count": len(feature["names"]) if feature.get("names") else 0,
                "is_video": feature.get("dtype") == "video",
            }
        )

    fps = float(info.get("fps", 0) or 0)
    episode_lengths_frames = (
        episodes_df["length"].astype(int).tolist()
        if "length" in episodes_df and actual_episode_count > 0
        else []
    )
    episode_lengths_seconds = [frames / fps for frames in episode_lengths_frames] if fps > 0 else []
    length_mean = (
        float(sum(episode_lengths_seconds) / len(episode_lengths_seconds)) if episode_lengths_seconds else 0.0
    )
    length_min = float(min(episode_lengths_seconds)) if episode_lengths_seconds else 0.0
    length_max = float(max(episode_lengths_seconds)) if episode_lengths_seconds else 0.0
    length_histogram = _build_episode_length_histogram(episode_lengths_seconds, bins=20)

    return {
        "dataset_root": str(dataset_root),
        "meta": {
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "codebase_version": info.get("codebase_version"),
            "splits": info.get("splits", {}),
            "declared_total_episodes": int(info.get("total_episodes", 0)),
            "declared_total_frames": int(info.get("total_frames", 0)),
            "declared_total_tasks": int(info.get("total_tasks", 0)),
        },
        "structure": {
            "features": feature_rows,
            "task_count": int(len(unique_tasks)),
            "tasks": unique_tasks,
            "tasks_parquet_columns": list(tasks_df.columns),
            "tasks_parquet_row_count": int(len(tasks_df)),
        },
        "quality": {
            "actual_episode_count": actual_episode_count,
            "actual_frame_count": actual_frame_count,
            "episode_length": {
                "unit": "seconds",
                "mean": length_mean,
                "min": length_min,
                "max": length_max,
                "histogram_bins": length_histogram,
            },
        },
        "success_metrics": {
            "success_count": success_count,
            "failure_count": failure_count,
            "unlabeled_count": unlabeled_count,
            "success_ratio_all_episodes": _safe_ratio(success_count, actual_episode_count),
            "success_ratio_labeled_episodes": _safe_ratio(success_count, labeled_count),
        },
        "intervention_metrics": {
            "intervention_frames": intervention_frames,
            "intervention_frame_ratio": _safe_ratio(intervention_frames, actual_frame_count),
            "episodes_with_intervention": intervention_episode_count,
            "episode_intervention_ratio": _safe_ratio(intervention_episode_count, actual_episode_count),
            "has_intervention_column": intervention_col in data_schema_names,
        },
    }


def format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("=== Dataset Report ===")
    lines.append(f"Root: {report['dataset_root']}")
    lines.append("")

    meta = report["meta"]
    lines.append("[Meta]")
    lines.append(f"- robot_type: {meta['robot_type']}")
    lines.append(f"- fps: {meta['fps']}")
    lines.append(f"- codebase_version: {meta['codebase_version']}")
    lines.append(f"- splits: {meta['splits']}")
    lines.append(
        f"- declared totals: episodes={meta['declared_total_episodes']}, "
        f"frames={meta['declared_total_frames']}, tasks={meta['declared_total_tasks']}"
    )
    lines.append("")

    quality = report["quality"]
    lines.append("[Actual Data Stats]")
    lines.append(
        f"- actual totals: episodes={quality['actual_episode_count']}, frames={quality['actual_frame_count']}"
    )
    lines.append(
        f"- episode length ({quality['episode_length']['unit']}): mean={quality['episode_length']['mean']:.2f}, "
        f"min={quality['episode_length']['min']:.2f}, max={quality['episode_length']['max']:.2f}"
    )
    lines.append("- episode length histogram (20 bins, terminal view):")
    lines.extend(_format_ascii_histogram(quality["episode_length"]["histogram_bins"]))
    lines.append("")

    success = report["success_metrics"]
    lines.append("[Success Metrics]")
    lines.append(
        f"- counts: success={success['success_count']}, failure={success['failure_count']}, "
        f"unlabeled={success['unlabeled_count']}"
    )
    lines.append(f"- success ratio (all episodes): {success['success_ratio_all_episodes']:.4f}")
    lines.append(f"- success ratio (labeled episodes): {success['success_ratio_labeled_episodes']:.4f}")
    lines.append("")

    intervention = report["intervention_metrics"]
    lines.append("[Intervention Metrics]")
    lines.append(f"- has intervention field: {intervention['has_intervention_column']}")
    lines.append(
        f"- frame-level: intervention_frames={intervention['intervention_frames']}, "
        f"ratio={intervention['intervention_frame_ratio']:.4f}"
    )
    lines.append(
        f"- episode-level: episodes_with_intervention={intervention['episodes_with_intervention']}, "
        f"ratio={intervention['episode_intervention_ratio']:.4f}"
    )
    lines.append("")

    structure = report["structure"]
    lines.append("[Task List]")
    lines.append(f"- task_count: {structure['task_count']}")
    for idx, task in enumerate(structure["tasks"]):
        lines.append(f"- task[{idx}]: {task}")
    lines.append("")

    lines.append("[Feature Schema]")
    for feature in structure["features"]:
        lines.append(
            f"- {feature['name']}: dtype={feature['dtype']}, shape={feature['shape']}, "
            f"names_count={feature['names_count']}, is_video={feature['is_video']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize LeRobot dataset schema and quality metrics.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id (e.g. local/eval_xxx) or absolute/local filesystem path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory containing datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    report = build_report(dataset_root)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_text_report(report))

    print(
        "\nTip: To visually inspect episode boundaries, run:\n"
        f"  lerobot-export-boundary-frames --dataset {args.dataset} --episodes all --output-dir <output_dir>"
    )


if __name__ == "__main__":
    main()
