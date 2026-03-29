from __future__ import annotations

import pytest
import torch

from lerobot.rlt.interfaces import ChunkTransition
from lerobot.rlt.offline_dataset import (
    split_episode_indices,
    save_cached_buffer,
    load_cached_buffer,
)


def test_split_episode_indices_counts():
    splits = split_episode_indices(100, train_ratio=0.8, val_ratio=0.1, seed=42)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 100


def test_split_episode_indices_no_overlap():
    splits = split_episode_indices(50, train_ratio=0.7, val_ratio=0.15, seed=0)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_episode_indices_deterministic():
    s1 = split_episode_indices(80, seed=123)
    s2 = split_episode_indices(80, seed=123)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]


def _make_transitions(n: int, state_dim: int = 10, action_dim: int = 4, C: int = 3):
    transitions = []
    for _ in range(n):
        transitions.append(ChunkTransition(
            state_vec=torch.randn(state_dim),
            exec_chunk=torch.randn(C, action_dim),
            ref_chunk=torch.randn(C, action_dim),
            reward_seq=torch.randn(C),
            next_state_vec=torch.randn(state_dim),
            next_ref_chunk=torch.randn(C, action_dim),
            done=torch.tensor(0.0),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(C),
        ))
    return transitions


def test_save_load_cached_buffer(tmp_path):
    transitions = _make_transitions(15)
    save_cached_buffer(transitions, str(tmp_path), split="train")
    buf = load_cached_buffer(str(tmp_path), split="train", capacity=100)
    assert len(buf) == 15

    # Verify round-trip preserves tensor values
    batch = buf.sample(15)
    original_states = torch.stack([t.state_vec for t in transitions])
    loaded_states = batch["state_vec"]
    # Each loaded state should match one original (order may differ due to sampling)
    for i in range(15):
        found = any(
            torch.allclose(loaded_states[i], original_states[j], atol=1e-6)
            for j in range(15)
        )
        assert found, f"Loaded state {i} not found in originals"


def test_build_transitions_basic():
    """build_transitions_from_demos requires a demo loader and agent.

    We test the internal _encoded_to_transitions helper directly since
    build_transitions_from_demos depends on a real DataLoader with Observation.
    """
    from lerobot.rlt.offline_dataset import _encoded_to_transitions

    state_dim, action_dim, C = 10, 4, 3
    encoded = []
    for _ in range(5):
        s = torch.randn(state_dim)
        r = torch.randn(C, action_dim)
        e = torch.randn(C, action_dim)
        rew = torch.zeros(C)
        encoded.append((s, r, e, rew))

    transitions = _encoded_to_transitions(encoded, C)
    assert len(transitions) == 5

    # Last transition should have done=1
    assert transitions[-1].done.item() == 1.0
    # Non-last transitions should have done=0
    for t in transitions[:-1]:
        assert t.done.item() == 0.0

    # next_state of transition i should be state of transition i+1
    for i in range(4):
        assert torch.allclose(transitions[i].next_state_vec, encoded[i + 1][0])

    # Shapes
    for t in transitions:
        assert t.state_vec.shape == (state_dim,)
        assert t.exec_chunk.shape == (C, action_dim)
        assert t.actual_steps.item() == C
