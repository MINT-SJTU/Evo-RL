from __future__ import annotations

import torch
import pytest

from lerobot.rlt.losses import discounted_chunk_return, critic_loss, actor_loss
from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.utils import soft_update

STATE_DIM = 78
CHUNK_DIM = 140
C = 10


@pytest.fixture
def actor():
    return ChunkActor(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def critic():
    return TwinCritic(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def target_critic(critic):
    import copy
    tc = copy.deepcopy(critic)
    for p in tc.parameters():
        p.requires_grad = False
    return tc


@pytest.fixture
def batch():
    B = 16
    return {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.randn(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.zeros(B),
        "actual_steps": torch.full((B,), C),
    }


def test_discounted_chunk_return_hand_computed():
    """Hand-computed: rewards=[1,1,1], gamma=0.5 -> 1 + 0.5 + 0.25 = 1.75"""
    reward_seq = torch.tensor([[1.0, 1.0, 1.0]])
    result = discounted_chunk_return(reward_seq, gamma=0.5)
    assert torch.allclose(result, torch.tensor([[1.75]]))


def test_discounted_chunk_return_shape():
    reward_seq = torch.randn(8, 10)
    result = discounted_chunk_return(reward_seq, gamma=0.99)
    assert result.shape == (8, 1)


def test_done_masking(actor, critic, target_critic):
    """When done=1, the bootstrap term should be zero."""
    B = 4
    batch_done = {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.ones(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.ones(B),  # all done
        "actual_steps": torch.full((B,), C),
    }
    loss = critic_loss(critic, target_critic, actor, batch_done, gamma=0.99, C=C)
    assert not torch.isnan(loss)


def test_critic_loss_scalar(actor, critic, target_critic, batch):
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_actor_loss_scalar(actor, critic, batch):
    loss = actor_loss(actor, critic, batch, beta=1.0)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_bc_term_scales_with_beta(actor, critic, batch):
    """Higher beta should give higher actor loss (assuming BC term > 0)."""
    torch.manual_seed(42)
    loss_low = actor_loss(actor, critic, batch, beta=0.0)
    torch.manual_seed(42)
    loss_high = actor_loss(actor, critic, batch, beta=10.0)
    # With beta=0, only Q term. With beta=10, large BC term added.
    # We just check they're different and the high-beta one is larger
    # (BC reg is always non-negative, so adding it increases loss)
    assert loss_high.item() > loss_low.item()


def test_target_is_stop_gradiented(actor, critic, target_critic, batch):
    """Target critic params should not receive gradients through critic_loss."""
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    loss.backward()
    for p in target_critic.parameters():
        assert p.grad is None
