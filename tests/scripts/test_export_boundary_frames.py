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

import pytest

from lerobot.scripts.lerobot_export_boundary_frames import (
    parse_episode_indices,
    select_camera_key,
)


def test_parse_episode_indices_supports_ranges_and_deduplicates():
    assert parse_episode_indices("0-2,4,2,6-7", total_episodes=10) == [0, 1, 2, 4, 6, 7]


def test_parse_episode_indices_supports_all_keyword():
    assert parse_episode_indices("all", total_episodes=4) == [0, 1, 2, 3]


def test_parse_episode_indices_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="out of range"):
        parse_episode_indices("0-5", total_episodes=5)


def test_select_camera_key_prefers_front_camera():
    info = {
        "features": {
            "observation.images.left_wrist": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.right_front": {"dtype": "video", "shape": [320, 320, 3]},
            "observation.images.right_wrist": {"dtype": "video", "shape": [720, 1280, 3]},
        }
    }

    assert select_camera_key(info, None) == "observation.images.right_front"


def test_select_camera_key_falls_back_to_largest_resolution():
    info = {
        "features": {
            "observation.images.left_wrist": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.right_wrist": {"dtype": "video", "shape": [720, 1280, 3]},
        }
    }

    assert select_camera_key(info, None) == "observation.images.right_wrist"
