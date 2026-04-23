#!/usr/bin/env python

import numpy as np
import torch

from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    build_bin_centers,
    compute_normalized_value_targets,
    expected_value_from_logits,
    project_values_to_bins,
)


def test_compute_normalized_value_targets():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    episode_info = {
        0: EpisodeTargetInfo(episode_index=0, task_index=0, length=3, success=True),
        1: EpisodeTargetInfo(episode_index=1, task_index=0, length=2, success=False),
    }
    task_scales = {0: 3.0}

    targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_scales=task_scales,
        c_fail_coef=1.0,
    )
    expected_targets = np.array([-2 / 6, -1 / 6, 0.0, -4 / 6, -3 / 6], dtype=np.float32)
    assert np.allclose(targets, expected_targets)


def test_project_values_to_bins_interpolates_between_neighbors():
    centers = build_bin_centers(num_bins=3, bin_min=-1.0, bin_max=0.0)
    values = torch.tensor([-1.0, -0.5, -0.25, 0.0], dtype=torch.float32)
    target = project_values_to_bins(values, centers)

    assert target.shape == (4, 3)
    assert torch.allclose(target[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(target[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(target[2], torch.tensor([0.0, 0.5, 0.5]), atol=1e-6)
    assert torch.allclose(target[3], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(target.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_expected_value_from_logits_matches_bin_center_weighting():
    centers = build_bin_centers(num_bins=3, bin_min=-1.0, bin_max=0.0)
    logits = torch.tensor(
        [
            [100.0, -100.0, -100.0],
            [-100.0, -100.0, 100.0],
        ],
        dtype=torch.float32,
    )
    expected = expected_value_from_logits(logits, centers)
    assert torch.allclose(expected, torch.tensor([-1.0, 0.0]), atol=1e-5)
