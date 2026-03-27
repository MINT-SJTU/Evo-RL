from __future__ import annotations

import torch
import pytest

from lerobot.rlt.vla_adapter import DummyVLAAdapter
from lerobot.rlt.interfaces import Observation


@pytest.fixture
def dummy_vla():
    return DummyVLAAdapter(token_dim=64, num_tokens=8, action_dim=7, horizon=20)


@pytest.fixture
def obs():
    return Observation(
        images={"base": torch.randn(4, 3, 64, 64)},
        proprio=torch.randn(4, 7),
    )


def test_forward_shapes(dummy_vla, obs):
    out = dummy_vla.forward_vla(obs)
    assert out.final_tokens.shape == (4, 8, 64)
    assert out.sampled_action_chunk.shape == (4, 20, 7)


def test_properties(dummy_vla):
    assert dummy_vla.token_dim == 64
    assert dummy_vla.action_dim == 7


def test_supervised_loss_scalar(dummy_vla, obs):
    expert = torch.randn(4, 20, 7)
    loss = dummy_vla.supervised_loss(obs, expert)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_dummy_vla_has_parameters(dummy_vla):
    params = list(dummy_vla.parameters())
    assert len(params) > 0
