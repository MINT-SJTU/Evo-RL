#!/usr/bin/env python

from pathlib import Path

import pyarrow.parquet as pq
import torch

from lerobot.rl.acp_instance_table import (
    ACP_INSTANCE_ROLE_FIELD,
    ACP_INSTANCE_SOURCE_INDEX_FIELD,
    ACP_ROLE_NEGATIVE,
    ACP_ROLE_POSITIVE,
    ACP_ROLE_UNLABELED,
    ACPInstanceTableDataset,
    build_acp_instance_table,
)
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG


def test_build_acp_instance_table_creates_expected_role_distribution():
    table = build_acp_instance_table(
        source_indices=torch.arange(10, dtype=torch.int64).numpy(),
        source_episode_indices=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int64).numpy(),
        source_frame_indices=torch.arange(10, dtype=torch.int64).numpy(),
        advantages=torch.arange(10, dtype=torch.float32).numpy(),
        negative_bottom_ratio=0.3,
        positive_top_duplicate_ratio=0.2,
    )

    roles = table[ACP_INSTANCE_ROLE_FIELD].to_pylist()
    assert len(roles) == 12
    assert roles.count(ACP_ROLE_NEGATIVE) == 3
    assert roles.count(ACP_ROLE_UNLABELED) == 7
    assert roles.count(ACP_ROLE_POSITIVE) == 2


class _DummyBaseDataset:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "episodes": {
                    "dataset_from_index": [0, 3],
                    "dataset_to_index": [3, 6],
                }
            },
        )()
        self.root = Path("/tmp/dummy")
        self.repo_id = "local/dummy"
        self.episodes = None
        self.image_transforms = None
        self.delta_timestamps = None
        self.tolerance_s = 1e-4
        self.revision = "v3.0"
        self.video_backend = None
        self._absolute_to_relative_idx = {i: i for i in range(6)}
        self.features = {}
        self.items = [
            {"task": "task 0", "index": torch.tensor([0]), "episode_index": torch.tensor([0])},
            {"task": "task 1", "index": torch.tensor([1]), "episode_index": torch.tensor([0])},
            {"task": "task 2", "index": torch.tensor([2]), "episode_index": torch.tensor([0])},
            {"task": "task 3", "index": torch.tensor([3]), "episode_index": torch.tensor([1])},
            {"task": "task 4", "index": torch.tensor([4]), "episode_index": torch.tensor([1])},
            {"task": "task 5", "index": torch.tensor([5]), "episode_index": torch.tensor([1])},
        ]

    @property
    def num_episodes(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        return dict(self.items[idx])


def test_acp_instance_table_dataset_applies_prompt_roles_and_drop_last_frames(tmp_path: Path):
    instance_table = build_acp_instance_table(
        source_indices=torch.arange(6, dtype=torch.int64).numpy(),
        source_episode_indices=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64).numpy(),
        source_frame_indices=torch.arange(6, dtype=torch.int64).numpy(),
        advantages=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy(),
        negative_bottom_ratio=1 / 3,
        positive_top_duplicate_ratio=1 / 3,
    )
    table_path = tmp_path / "instances.parquet"
    pq.write_table(instance_table, table_path)

    dataset = ACPInstanceTableDataset(
        base_dataset=_DummyBaseDataset(),
        instance_table_path=table_path,
        drop_n_last_frames=1,
    )

    assert len(dataset) == 5

    tasks = [dataset[i]["task"] for i in range(len(dataset))]
    assert any(task.endswith(ACP_NEGATIVE_TAG) for task in tasks)
    assert any(task.endswith(ACP_POSITIVE_TAG) for task in tasks)
    assert any("Advantage:" not in task for task in tasks)

    roles = [dataset[i]["complementary_info.acp_prompt_role"] for i in range(len(dataset))]
    assert ACP_ROLE_NEGATIVE in roles
    assert ACP_ROLE_POSITIVE in roles
    assert ACP_ROLE_UNLABELED in roles

    source_indices = [
        int(dataset[i]["complementary_info.acp_source_index"].reshape(-1)[0].item()) for i in range(len(dataset))
    ]
    assert 2 not in source_indices
    assert 5 not in source_indices
