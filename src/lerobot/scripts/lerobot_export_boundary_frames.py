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
Export the first and last frame of selected episodes from a LeRobot dataset.

Examples:

```bash
lerobot-export-boundary-frames \
    --dataset ~/.cache/huggingface/lerobot/javadcc/evorl_0 \
    --episodes 0-99 \
    --camera-key observation.images.right_front \
    --output-dir outputs/evorl_0_right_front_first_last_ep000_099
```

```bash
lerobot-export-boundary-frames \
    --dataset local/my_dataset \
    --episodes all \
    --output-dir outputs/my_dataset_boundaries
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_default_hf_lerobot_home() -> Path:
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.getenv("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser()


def resolve_dataset_root(dataset: str, root: Path | None) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base_root = root.expanduser().resolve() if root is not None else get_default_hf_lerobot_home().resolve()
    candidates = [base_root / dataset]
    if "/" not in dataset.strip("/"):
        candidates.append(base_root / "local" / dataset)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Dataset path not found. Searched:\n{searched}")


def resolve_repo_id(dataset: str, dataset_root: Path, root: Path | None) -> str:
    dataset_path = Path(dataset).expanduser()
    if not dataset_path.exists():
        return dataset.strip("/")

    base_root = root.expanduser().resolve() if root is not None else get_default_hf_lerobot_home().resolve()
    if dataset_root.is_relative_to(base_root):
        return dataset_root.relative_to(base_root).as_posix()

    return f"local/{dataset_root.name}"


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open() as f:
        return json.load(f)


def get_camera_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    return [
        key
        for key, value in sorted(features.items())
        if key.startswith("observation.images.") and value.get("dtype") in {"image", "video"}
    ]


def select_camera_key(info: dict, camera_key: str | None) -> str:
    camera_keys = get_camera_keys(info)
    if not camera_keys:
        raise ValueError("Dataset does not contain any image or video observation keys.")

    if camera_key is not None:
        if camera_key not in camera_keys:
            available = "\n".join(f"- {key}" for key in camera_keys)
            raise ValueError(f"Unknown camera key: {camera_key}\nAvailable cameras:\n{available}")
        return camera_key

    features = info["features"]

    def sort_key(key: str) -> tuple[int, int, str]:
        shape = features[key].get("shape", [0, 0, 0])
        height = int(shape[0]) if len(shape) > 0 else 0
        width = int(shape[1]) if len(shape) > 1 else 0
        pixel_count = height * width
        prefers_front = 0 if "front" in key else 1
        return (prefers_front, -pixel_count, key)

    return sorted(camera_keys, key=sort_key)[0]


def parse_episode_indices(spec: str, total_episodes: int) -> list[int]:
    cleaned = spec.strip().lower()
    if total_episodes <= 0:
        raise ValueError("Dataset has no episodes to export.")

    if cleaned == "all":
        return list(range(total_episodes))

    episode_indices: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue

        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]

        for episode_index in values:
            if episode_index < 0 or episode_index >= total_episodes:
                raise ValueError(
                    f"Episode index {episode_index} is out of range for dataset with {total_episodes} episodes."
                )
            if episode_index not in seen:
                episode_indices.append(episode_index)
                seen.add(episode_index)

    if not episode_indices:
        raise ValueError("No episodes were selected.")

    return episode_indices


def tensor_to_pil_image(image: "torch.Tensor") -> "Image.Image":
    from PIL import Image
    import torch

    array = image.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)


def export_episode_boundary_frames(
    dataset: "LeRobotDataset",
    output_dir: Path,
    episode_indices: list[int],
    camera_key: str,
) -> Path:
    pad_width = max(3, len(str(max(episode_indices))))
    episodes = dataset.meta.episodes

    manifest_rows: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        from_idx = int(episodes["dataset_from_index"][episode_index])
        to_idx = int(episodes["dataset_to_index"][episode_index])
        length = int(episodes["length"][episode_index])
        success = str(episodes["episode_success"][episode_index]) if "episode_success" in episodes else ""

        first_name = f"episode_{episode_index:0{pad_width}d}_first.png"
        last_name = f"episode_{episode_index:0{pad_width}d}_last.png"

        tensor_to_pil_image(dataset[from_idx][camera_key]).save(output_dir / first_name)
        tensor_to_pil_image(dataset[to_idx - 1][camera_key]).save(output_dir / last_name)

        manifest_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "episode_success": success,
                "first_dataset_index": from_idx,
                "last_dataset_index": to_idx - 1,
                "first_file": first_name,
                "last_file": last_name,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "length",
                "episode_success",
                "first_dataset_index",
                "last_dataset_index",
                "first_file",
                "last_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output_dir / "README.txt").write_text(
        f"Dataset root: {dataset.root}\n"
        f"Dataset repo_id: {dataset.repo_id}\n"
        f"Camera: {camera_key}\n"
        f"Episodes exported: {episode_indices[0]}-{episode_indices[-1]} ({len(episode_indices)} total)\n"
        "Files per episode: episode_XXX_first.png, episode_XXX_last.png\n"
    )

    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repo id or local dataset path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional local dataset root to search under when --dataset is not an existing path.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode selection like 'all', '0-99', or '0-9,12,20-25'.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Camera key to export. Defaults to the front camera when available, otherwise the largest image key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where boundary frames and manifest.csv will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output-dir first if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    repo_id = resolve_repo_id(args.dataset, dataset_root, args.root)
    info = load_info(dataset_root)
    camera_key = select_camera_key(info, args.camera_key)
    episode_indices = parse_episode_indices(args.episodes, int(info["total_episodes"]))

    prepare_output_dir(args.output_dir, args.overwrite)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    manifest_path = export_episode_boundary_frames(
        dataset=dataset,
        output_dir=args.output_dir,
        episode_indices=episode_indices,
        camera_key=camera_key,
    )

    print(f"dataset_root={dataset_root}")
    print(f"repo_id={repo_id}")
    print(f"camera_key={camera_key}")
    print(f"episodes_exported={len(episode_indices)}")
    print(f"manifest={manifest_path}")

    ssh_connection = os.environ.get("SSH_CONNECTION")
    if ssh_connection:
        parts = ssh_connection.split()
        client_ip = parts[0]
        abs_output = args.output_dir.resolve()
        hostname = os.uname().nodename
        print(
            f"\nTip: You are in an SSH session (client {client_ip}).\n"
            f"To pull these files to your local machine, run on your local terminal:\n"
            f"  scp -r {hostname}:{abs_output} ~/Downloads/"
        )


if __name__ == "__main__":
    main()
