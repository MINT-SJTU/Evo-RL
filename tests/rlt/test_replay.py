from __future__ import annotations

import torch
import pytest

from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.interfaces import ChunkTransition

C = 10
ACTION_DIM = 14
STATE_DIM = 2062


def _make_transition() -> ChunkTransition:
    return ChunkTransition(
        state_vec=torch.randn(STATE_DIM),
        exec_chunk=torch.randn(C, ACTION_DIM),
        ref_chunk=torch.randn(C, ACTION_DIM),
        reward_seq=torch.randn(C),
        next_state_vec=torch.randn(STATE_DIM),
        next_ref_chunk=torch.randn(C, ACTION_DIM),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(C),
    )


def test_add_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.add(_make_transition())
    assert len(buf) == 1


def test_capacity_limit():
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.add(_make_transition())
    assert len(buf) == 5


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(_make_transition())

    batch = buf.sample(8)
    assert batch["state_vec"].shape == (8, STATE_DIM)
    assert batch["exec_chunk_flat"].shape == (8, C * ACTION_DIM)
    assert batch["ref_chunk_flat"].shape == (8, C * ACTION_DIM)
    assert batch["reward_seq"].shape == (8, C)
    assert batch["next_state_vec"].shape == (8, STATE_DIM)
    assert batch["next_ref_flat"].shape == (8, C * ACTION_DIM)
    assert batch["done"].shape == (8,)
    assert batch["source"].shape == (8,)
    assert batch["episode_id"].shape == (8,)
    assert batch["is_critical"].shape == (8,)


def test_sample_capped_by_buffer_size():
    buf = ReplayBuffer(capacity=100)
    for _ in range(3):
        buf.add(_make_transition())
    batch = buf.sample(10)
    assert batch["state_vec"].shape[0] == 3


def test_batch_keys():
    buf = ReplayBuffer(capacity=100)
    buf.add(_make_transition())
    batch = buf.sample(1)
    expected_keys = {
        "state_vec", "exec_chunk_flat", "ref_chunk_flat",
        "reward_seq", "next_state_vec", "next_ref_flat", "done", "actual_steps",
        "source", "episode_id", "is_critical",
    }
    assert set(batch.keys()) == expected_keys
