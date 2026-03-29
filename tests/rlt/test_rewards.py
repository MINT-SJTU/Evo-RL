from __future__ import annotations

import pytest
import torch

from lerobot.rlt.rewards import build_reward_seq, REWARD_MODES


C = 6
ACTION_DIM = 4


def _make_chunks(C: int = C, action_dim: int = ACTION_DIM):
    expert = torch.randn(C, action_dim)
    exec_ = expert + torch.randn_like(expert) * 0.1
    return expert, exec_


def test_action_matching_mode():
    expert, exec_ = _make_chunks()
    reward = build_reward_seq(expert, exec_, mode="action_matching")
    expected = -(exec_ - expert).pow(2).sum(dim=-1)
    assert reward.shape == (C,)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_terminal_mode():
    expert, exec_ = _make_chunks()
    reward = build_reward_seq(
        expert, exec_, mode="terminal", episode_success=True,
        is_terminal_chunk=True, success_bonus=5.0,
    )
    assert reward.shape == (C,)
    assert reward[-1].item() == pytest.approx(5.0)
    assert reward[:-1].abs().sum().item() == 0.0


def test_hybrid_mode():
    expert, exec_ = _make_chunks()
    matching = build_reward_seq(expert, exec_, mode="action_matching")
    terminal = build_reward_seq(
        expert, exec_, mode="terminal", episode_success=True,
        is_terminal_chunk=True, success_bonus=10.0,
    )
    hybrid = build_reward_seq(
        expert, exec_, mode="hybrid", episode_success=True,
        is_terminal_chunk=True, success_bonus=10.0,
    )
    assert torch.allclose(hybrid, matching + terminal, atol=1e-6)


def test_actual_steps_padding():
    expert, exec_ = _make_chunks()
    actual_steps = 3
    reward = build_reward_seq(
        expert, exec_, mode="action_matching", actual_steps=actual_steps,
    )
    assert reward[actual_steps:].abs().sum().item() == 0.0
    assert reward[:actual_steps].abs().sum().item() > 0.0


def test_no_terminal_when_not_terminal_chunk():
    expert, exec_ = _make_chunks()
    reward = build_reward_seq(
        expert, exec_, mode="terminal", episode_success=True,
        is_terminal_chunk=False, success_bonus=10.0,
    )
    assert reward.abs().sum().item() == 0.0


def test_invalid_mode_raises():
    expert, exec_ = _make_chunks()
    with pytest.raises(ValueError, match="Unknown reward mode"):
        build_reward_seq(expert, exec_, mode="nonexistent")


def test_tensor_actual_steps():
    expert, exec_ = _make_chunks()
    reward_int = build_reward_seq(expert, exec_, mode="action_matching", actual_steps=5)
    reward_tensor = build_reward_seq(expert, exec_, mode="action_matching", actual_steps=torch.tensor(5))
    assert torch.allclose(reward_int, reward_tensor, atol=1e-6)
