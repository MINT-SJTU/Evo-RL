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

"""
Export a LeRobot v3.0 dataset into a legacy v2.1-style layout that Evo1 can train on.

This script is intentionally pragmatic: it restores the files that Evo1's current dataset
loader actually consumes.

It recreates:
- meta/tasks.jsonl
- meta/episodes.jsonl
- meta/episodes_stats.jsonl
- meta/stats.json
- data/chunk-xxx/episode_xxxxxx.parquet
- videos/chunk-xxx/<camera>/episode_xxxxxx.mp4

It can also optionally rewrite task strings using ACP indicators so the exported dataset
preserves Evo-RL's prompt-conditioning semantics without changing Evo1's training code.
"""

import argparse
import copy
import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_info,
    load_stats,
    load_tasks,
    serialize_dict,
    unflatten_dict,
    write_info,
    write_stats,
)
from lerobot.datasets.dataset_tools import _keep_episodes_from_video_with_av
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(make_json_ready(record), ensure_ascii=False))
            f.write("\n")


def make_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_ready(subvalue) for key, subvalue in value.items()}
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if converted is not value:
            return make_json_ready(converted)
    if isinstance(value, list):
        return [make_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def legacy_chunk_dir(episode_index: int) -> str:
    return f"chunk-{episode_index // DEFAULT_CHUNK_SIZE:03d}"


def legacy_data_file_path(root: Path, episode_index: int) -> Path:
    return root / "data" / legacy_chunk_dir(episode_index) / f"episode_{episode_index:06d}.parquet"


def legacy_video_file_path(root: Path, video_key: str, episode_index: int) -> Path:
    return root / "videos" / legacy_chunk_dir(episode_index) / video_key / f"episode_{episode_index:06d}.mp4"


def resolve_source_root(
    src_root: str | None,
    repo_id: str | None,
    root: str | None,
    episodes: list[int] | None,
) -> Path:
    if src_root is not None:
        resolved = Path(src_root).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Source dataset root does not exist: {resolved}")
        return resolved

    if repo_id is None:
        raise ValueError("Provide either --src-root or --repo-id.")

    base_root = Path(root).expanduser().resolve() if root is not None else HF_LEROBOT_HOME.resolve()
    dataset_root = base_root / repo_id
    LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=episodes, revision=V30, download_videos=True)
    return dataset_root.resolve()


def build_exported_task_maps(
    tasks_df: pd.DataFrame, acp_indicator_field: str | None
) -> tuple[dict[int, str], dict[tuple[int, int], int] | None]:
    source_tasks: list[tuple[int, str]] = []
    ordered_tasks = tasks_df.sort_values("task_index")
    for task_text, row in ordered_tasks.iterrows():
        source_tasks.append((int(row["task_index"]), "" if task_text is None else str(task_text)))

    if acp_indicator_field is None:
        task_by_index = {task_index: task_text for task_index, task_text in source_tasks}
        return task_by_index, None

    task_by_index: dict[int, str] = {}
    pair_to_exported: dict[tuple[int, int], int] = {}
    next_task_index = 0
    for task_index, task_text in source_tasks:
        for indicator in (1, 0):
            pair_to_exported[(task_index, indicator)] = next_task_index
            task_by_index[next_task_index] = build_acp_tagged_task(task_text, is_positive=bool(indicator))
            next_task_index += 1

    return task_by_index, pair_to_exported


def ensure_selected_episodes(episodes_df: pd.DataFrame, episodes: list[int] | None) -> pd.DataFrame:
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    if episodes is None:
        return episodes_df

    selected = set(episodes)
    filtered = episodes_df[episodes_df["episode_index"].isin(selected)].copy()
    found = {int(ep) for ep in filtered["episode_index"].tolist()}
    missing = sorted(selected - found)
    if missing:
        raise ValueError(f"Requested episode indices not found in metadata: {missing}")
    return filtered.sort_values("episode_index").reset_index(drop=True)


def load_episodes_dataframe(src_root: Path) -> pd.DataFrame:
    episode_paths = sorted((src_root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode metadata parquet files found under {src_root / 'meta' / 'episodes'}")

    episode_frames = [pd.read_parquet(path) for path in episode_paths]
    return pd.concat(episode_frames, ignore_index=True)


def normalize_indicator_series(series: pd.Series, field_name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"ACP indicator field '{field_name}' contains null values.")
    invalid = series[~series.isin([0, 1])]
    if not invalid.empty:
        values = sorted({repr(value) for value in invalid.tolist()[:8]})
        raise ValueError(f"ACP indicator field '{field_name}' must contain only 0/1 values, got {values}")
    return series.astype(int)


def export_data_files(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episodes: list[int] | None,
    task_by_index: dict[int, str],
    acp_indicator_field: str | None,
    acp_task_remap: dict[tuple[int, int], int] | None,
) -> tuple[dict[int, list[str]], set[int]]:
    selected = set(episodes) if episodes is not None else None
    visual_feature_columns = {
        key for key, feature in info["features"].items() if feature["dtype"] in {"image", "video"}
    }

    episode_tasks: dict[int, list[str]] = {}
    used_task_indices: set[int] = set()

    for parquet_path in sorted((src_root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)

        if selected is not None:
            df = df[df["episode_index"].isin(selected)]
        if df.empty:
            continue

        drop_columns = [column for column in visual_feature_columns if column in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)

        if "task_index" not in df.columns:
            raise KeyError(f"Expected 'task_index' column in {parquet_path}")

        if acp_indicator_field is not None:
            if acp_task_remap is None:
                raise ValueError("ACP task remap is missing while ACP export is enabled.")
            if acp_indicator_field not in df.columns:
                raise KeyError(f"ACP indicator field '{acp_indicator_field}' not found in {parquet_path}")
            indicators = normalize_indicator_series(df[acp_indicator_field], acp_indicator_field)
            if df["task_index"].isna().any():
                raise ValueError(f"'task_index' contains null values in {parquet_path}")
            remapped_indices = [
                acp_task_remap[(int(task_index), int(indicator))]
                for task_index, indicator in zip(df["task_index"].tolist(), indicators.tolist(), strict=False)
            ]
            df = df.copy()
            df["task_index"] = remapped_indices

        for episode_index, episode_df in df.groupby("episode_index", sort=True):
            episode_index = int(episode_index)
            output_path = legacy_data_file_path(dst_root, episode_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            episode_df.to_parquet(output_path, index=False)

            task_indices = [int(task_index) for task_index in episode_df["task_index"].tolist()]
            used_task_indices.update(task_indices)
            episode_tasks[episode_index] = list(
                dict.fromkeys(task_by_index[task_index] for task_index in task_indices if task_index in task_by_index)
            )

    return episode_tasks, used_task_indices


def export_tasks_jsonl(dst_root: Path, task_by_index: dict[int, str], used_task_indices: set[int]) -> None:
    records = [
        {"task_index": int(task_index), "task": task_by_index[task_index]}
        for task_index in sorted(used_task_indices)
    ]
    write_jsonl(records, dst_root / LEGACY_TASKS_PATH)


def flatten_episode_tasks(task_value: Any) -> list[str]:
    if hasattr(task_value, "tolist") and not isinstance(task_value, (str, bytes)):
        return flatten_episode_tasks(task_value.tolist())
    if isinstance(task_value, list):
        flattened: list[str] = []
        for value in task_value:
            flattened.extend(flatten_episode_tasks(value))
        return flattened
    if task_value is None:
        return []
    return [str(task_value)]


def export_episodes_jsonl(dst_root: Path, episodes_df: pd.DataFrame, episode_tasks: dict[int, list[str]]) -> None:
    records: list[dict[str, Any]] = []
    for _, row in episodes_df.iterrows():
        episode_index = int(row["episode_index"])
        tasks = episode_tasks.get(episode_index)
        if tasks is None:
            tasks = flatten_episode_tasks(row.get("tasks"))
        records.append(
            {
                "episode_index": episode_index,
                "tasks": tasks,
                "length": int(row["length"]),
            }
        )
    write_jsonl(records, dst_root / LEGACY_EPISODES_PATH)


def export_episode_stats_jsonl(src_root: Path, dst_root: Path, episodes: list[int] | None) -> None:
    selected = set(episodes) if episodes is not None else None
    records: list[dict[str, Any]] = []

    for parquet_path in sorted((src_root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        if selected is not None:
            df = df[df["episode_index"].isin(selected)]
        if df.empty:
            continue

        stats_columns = [column for column in df.columns if column.startswith("stats/")]
        if not stats_columns:
            continue

        for _, row in df.sort_values("episode_index").iterrows():
            flat_stats = {column.removeprefix("stats/"): row[column] for column in stats_columns}
            records.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "stats": serialize_dict(unflatten_dict(flat_stats)),
                }
            )

    if records:
        write_jsonl(records, dst_root / LEGACY_EPISODES_STATS_PATH)


def export_videos(
    src_root: Path,
    dst_root: Path,
    info: dict[str, Any],
    episodes_df: pd.DataFrame,
    vcodec: str,
    pix_fmt: str,
) -> None:
    video_keys = sorted([key for key, feature in info["features"].items() if feature["dtype"] == "video"])
    if not video_keys:
        logging.warning("Source dataset has no video features; skipping video export.")
        return

    if info["video_path"] is None:
        raise ValueError("Source dataset declares video features but info['video_path'] is None.")

    fps = float(info["fps"])
    file_usage: dict[str, Counter[tuple[int, int]]] = {video_key: Counter() for video_key in video_keys}
    for _, row in episodes_df.iterrows():
        for video_key in video_keys:
            file_usage[video_key][
                (int(row[f"videos/{video_key}/chunk_index"]), int(row[f"videos/{video_key}/file_index"]))
            ] += 1

    for video_key in video_keys:
        logging.info(f"Exporting legacy videos for {video_key}")
        for _, row in episodes_df.iterrows():
            episode_index = int(row["episode_index"])
            chunk_index = int(row[f"videos/{video_key}/chunk_index"])
            file_index = int(row[f"videos/{video_key}/file_index"])
            from_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
            to_timestamp = float(row[f"videos/{video_key}/to_timestamp"])

            src_video_rel = info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            src_video_path = src_root / src_video_rel
            if not src_video_path.exists():
                raise FileNotFoundError(f"Missing source video file: {src_video_path}")

            dst_video_path = legacy_video_file_path(dst_root, video_key, episode_index)
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)

            if file_usage[video_key][(chunk_index, file_index)] == 1 and abs(from_timestamp) < 1e-6:
                shutil.copy(src_video_path, dst_video_path)
                continue

            _keep_episodes_from_video_with_av(
                input_path=src_video_path,
                output_path=dst_video_path,
                episodes_to_keep=[(from_timestamp, to_timestamp)],
                fps=fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
            )


def build_legacy_info(
    info: dict[str, Any],
    episodes_df: pd.DataFrame,
    used_task_indices: set[int],
    acp_indicator_field: str | None,
) -> dict[str, Any]:
    legacy_info = copy.deepcopy(info)
    legacy_info["codebase_version"] = V21
    legacy_info["total_episodes"] = int(len(episodes_df))
    legacy_info["total_frames"] = int(episodes_df["length"].sum())
    legacy_info["total_tasks"] = int(len(used_task_indices))
    legacy_info["total_chunks"] = int((max(episodes_df["episode_index"]) // DEFAULT_CHUNK_SIZE) + 1) if len(episodes_df) else 0
    legacy_info["total_videos"] = int(
        len([key for key, feature in info["features"].items() if feature["dtype"] == "video"]) * len(episodes_df)
    )
    legacy_info["data_path"] = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
    legacy_info["video_path"] = (
        "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        if info["video_path"] is not None
        else None
    )
    if acp_indicator_field is not None:
        legacy_info["acp_export_indicator_field"] = acp_indicator_field
    return legacy_info


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    episodes: list[int] | None,
    acp_indicator_field: str | None,
    force: bool,
    vcodec: str,
    pix_fmt: str,
) -> None:
    info = load_info(src_root)
    if info.get("codebase_version") != V30:
        raise ValueError(
            f"Source dataset at {src_root} has codebase_version={info.get('codebase_version')!r}, expected {V30!r}."
        )

    tasks_df = load_tasks(src_root)
    episodes_df = ensure_selected_episodes(load_episodes_dataframe(src_root), episodes)

    if dst_root.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {dst_root}. Use --force to overwrite.")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    task_by_index, acp_task_remap = build_exported_task_maps(tasks_df, acp_indicator_field)

    episode_tasks, used_task_indices = export_data_files(
        src_root=src_root,
        dst_root=dst_root,
        info=info,
        episodes=episodes,
        task_by_index=task_by_index,
        acp_indicator_field=acp_indicator_field,
        acp_task_remap=acp_task_remap,
    )
    if not used_task_indices:
        raise ValueError("No frames were exported. Check --episodes or the source dataset contents.")

    export_tasks_jsonl(dst_root, task_by_index, used_task_indices)
    export_episodes_jsonl(dst_root, episodes_df, episode_tasks)

    stats = load_stats(src_root)
    if stats is not None:
        write_stats(stats, dst_root)
    export_episode_stats_jsonl(src_root, dst_root, episodes)

    export_videos(
        src_root=src_root,
        dst_root=dst_root,
        info=info,
        episodes_df=episodes_df,
        vcodec=vcodec,
        pix_fmt=pix_fmt,
    )

    legacy_info = build_legacy_info(info, episodes_df, used_task_indices, acp_indicator_field)
    write_info(legacy_info, dst_root)

    logging.info("Export completed")
    logging.info(f"  source: {src_root}")
    logging.info(f"  output: {dst_root}")
    logging.info(f"  episodes: {len(episodes_df)}")
    logging.info(f"  tasks: {len(used_task_indices)}")


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, default=None, help="Local v3.0 dataset root to export.")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional HF dataset repo id to download first.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Base directory for HF downloads when --repo-id is used. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output directory for the exported legacy dataset.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional episode indices to export. Useful for smoke tests and partial exports.",
    )
    parser.add_argument(
        "--acp-indicator-field",
        type=str,
        default=None,
        help="If set, rewrite task strings/task_index using this 0/1 frame field, e.g. complementary_info.acp_indicator.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Codec used when slicing shared source video files into per-episode legacy files.",
    )
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        help="Pixel format used when re-encoding legacy video clips.",
    )
    args = parser.parse_args()

    src_root = resolve_source_root(args.src_root, args.repo_id, args.root, args.episodes)
    dst_root = Path(args.dst_root).expanduser().resolve()

    convert_dataset(
        src_root=src_root,
        dst_root=dst_root,
        episodes=args.episodes,
        acp_indicator_field=args.acp_indicator_field,
        force=args.force,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
    )


if __name__ == "__main__":
    main()
