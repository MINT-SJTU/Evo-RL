#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Recompute global LeRobot dataset statistics directly from dataset samples.

This script is intended for datasets whose existing `meta/stats.json` is missing
quantiles or should be regenerated from scratch. Unlike approaches that aggregate
episode- or shard-level stats, this script scans the dataset itself and computes
new global feature statistics over the full merged dataset contents.

Usage:

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from requests import HTTPError
from tqdm import tqdm

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging


def has_quantile_stats(stats: dict[str, dict] | None, quantile_list_keys: list[str] | None = None) -> bool:
    """Check if dataset statistics already contain quantile information.

    Args:
        stats: Dataset statistics dictionary

    Returns:
        True if quantile statistics are present, False otherwise
    """
    if quantile_list_keys is None:
        quantile_list_keys = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]

    if stats is None:
        return False

    for feature_stats in stats.values():
        if any(q_key in feature_stats for q_key in quantile_list_keys):
            return True

    return False


def _stack_feature_values(values: list[torch.Tensor | np.ndarray]) -> np.ndarray:
    tensors = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values]
    return torch.stack(tensors).cpu().numpy()


def _compute_feature_stats(dataset: LeRobotDataset, key: str) -> dict[str, np.ndarray]:
    feature = dataset.features[key]
    dtype = feature["dtype"]
    if dtype == "string":
        raise ValueError(f"String feature '{key}' does not have numeric statistics.")

    values = []
    for idx in tqdm(range(dataset.num_frames), desc=f"Feature {key}", leave=False):
        item = dataset[idx]
        values.append(item[key])

    if not values:
        raise ValueError(f"No samples found for feature '{key}'.")

    data = _stack_feature_values(values)
    if dtype in ["image", "video"]:
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        axes_to_reduce = (0, 2, 3)
        keepdims = True
    else:
        axes_to_reduce = 0
        keepdims = data.ndim == 1

    stats = get_feature_stats(data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES)
    if dtype in ["image", "video"]:
        stats = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in stats.items()}
    return stats


def _compute_selected_feature_stats(dataset: LeRobotDataset, feature_keys: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Compute stats for multiple selected non-visual features in a single dataset scan."""
    if not feature_keys:
        raise ValueError("feature_keys must not be empty.")

    feature_values: dict[str, list[torch.Tensor | np.ndarray]] = {key: [] for key in feature_keys}
    for idx in tqdm(range(dataset.num_frames), desc="Selected features", leave=False):
        item = dataset[idx]
        for key in feature_keys:
            feature_values[key].append(item[key])

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in feature_keys:
        feature = dataset.features[key]
        dtype = feature["dtype"]
        if dtype in {"string", "image", "video"}:
            raise ValueError(f"Selected feature '{key}' must be a non-visual numeric feature, got dtype={dtype}.")

        values = feature_values[key]
        if not values:
            raise ValueError(f"No samples found for feature '{key}'.")

        data = _stack_feature_values(values)
        stats[key] = get_feature_stats(
            data,
            axis=0,
            keepdims=data.ndim == 1,
            quantile_list=DEFAULT_QUANTILES,
        )

    return stats


def compute_quantile_stats_for_dataset(
    dataset: LeRobotDataset,
    feature_keys: list[str] | None = None,
) -> dict[str, dict]:
    """Compute dataset-level statistics directly from all frames in the dataset."""
    logging.info(
        "Computing global dataset statistics from %d frames across %d episodes",
        dataset.num_frames,
        dataset.num_episodes,
    )

    if feature_keys is None:
        feature_keys = [key for key, ft in dataset.features.items() if ft["dtype"] != "string"]
    if not feature_keys:
        raise ValueError("Dataset has no numeric/image/video features to compute statistics for.")

    stats: dict[str, dict] = {}
    for key in feature_keys:
        logging.info("Computing global stats for feature '%s'", key)
        stats[key] = _compute_feature_stats(dataset, key)

    return stats


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    include_visual_stats: bool = True,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        push_to_hub: Whether to push updated metadata to the Hugging Face Hub
        include_visual_stats: Whether to recompute stats for image/video features
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    if not overwrite and has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info("Dataset does not contain quantile statistics. Computing them now...")

    if include_visual_stats:
        new_stats = compute_quantile_stats_for_dataset(dataset)
    else:
        logging.info(
            "Skipping visual features and restricting stats to policy-critical features: observation.state, action."
        )
        selected_feature_keys = [
            key for key in ["observation.state", "action"] if key in dataset.features
        ]
        if not selected_feature_keys:
            raise ValueError(
                "Neither 'observation.state' nor 'action' exists in the dataset features."
            )

        logging.info(
            "Fast path: computing selected non-visual features in one scan: %s",
            ", ".join(selected_feature_keys),
        )
        new_stats = _compute_selected_feature_stats(dataset, selected_feature_keys)

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    if not push_to_hub:
        logging.info("Skipping push_to_hub; local dataset metadata has been updated only.")
        return

    dataset.push_to_hub()

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        logging.info(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        pass
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=None, repo_type="dataset")


def main():
    """Main function to run the augmentation script."""
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with quantile statistics")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht')",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Local root directory for the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile statistics if they already exist",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push updated dataset metadata to the Hugging Face Hub after recomputing stats.",
    )
    parser.add_argument(
        "--exclude-visual-stats",
        action="store_true",
        help="Skip image/video features. Useful for policies such as pi0/pi05 where VISUAL=IDENTITY.",
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        include_visual_stats=not args.exclude_visual_stats,
    )


if __name__ == "__main__":
    main()
