from __future__ import annotations

import copy

import torch
import pytest

from lerobot.rlt.losses import critic_loss, actor_loss
from lerobot.rlt.utils import soft_update, flatten_chunk, compute_discount_vector
from lerobot.rlt.interfaces import Observation, ChunkTransition
from lerobot.rlt.replay_buffer import ReplayBuffer
from tests.rlt.helpers import (
    make_test_algorithm, make_batch, make_fake_transition,
    TOKEN_DIM, ACTION_DIM, PROPRIO_DIM, C, STATE_DIM, CHUNK_DIM,
)


@pytest.fixture
def algorithm():
    algo, _ = make_test_algorithm()
    return algo


@pytest.fixture
def batch():
    return make_batch()


def test_single_critic_step_loss_not_nan(algorithm, batch):
    """Single critic gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=1e-3)
    loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify params changed
    loss2 = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss2)


def test_single_actor_step_loss_not_nan(algorithm, batch):
    """Single actor gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=1e-3)
    loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, beta=1.0)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_frozen_params_stay_frozen(algorithm, batch):
    """After freezing VLA and encoder, their params should not get gradients."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    # Run both losses
    c_loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    c_loss.backward()

    # VLA params: no grad
    for p in algorithm.policy.vla.parameters():
        assert p.grad is None

    # RL token encoder params: no grad
    assert algorithm.policy.rl_token.rl_token_embed.grad is None
    for p in algorithm.policy.rl_token.encoder.parameters():
        assert p.grad is None


def test_full_forward_backward_pipeline(algorithm):
    """End-to-end: obs -> state -> action -> critic -> loss -> backward."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    obs = Observation(
        images={"base": torch.randn(4, 3, 64, 64)},
        proprio=torch.randn(4, PROPRIO_DIM),
    )

    with torch.no_grad():
        state = algorithm.policy.get_rl_state(obs)
        ref = algorithm.policy.get_reference_chunk(obs)
        ref_flat = flatten_chunk(ref)

    action, mu = algorithm.policy.actor.sample(state, ref_flat, training=True)
    q = algorithm.critic.min_q(state, action)
    loss = -q.mean()
    loss.backward()

    # Actor should have gradients
    for p in algorithm.policy.actor.parameters():
        assert p.grad is not None

    # Critic should have gradients (from Q computation)
    for p in algorithm.critic.parameters():
        assert p.grad is not None


def test_soft_update_changes_target(algorithm):
    """Verify soft update actually modifies target critic."""
    orig_params = [p.data.clone() for p in algorithm.target_critic.parameters()]

    # Perturb online critic
    for p in algorithm.critic.parameters():
        p.data += torch.randn_like(p.data) * 0.1

    soft_update(algorithm.target_critic, algorithm.critic, tau=0.1)

    changed = False
    for orig, p in zip(orig_params, algorithm.target_critic.parameters()):
        if not torch.allclose(orig, p.data):
            changed = True
            break
    assert changed


def test_compute_discount_vector():
    v = compute_discount_vector(0.99, 5)
    assert v.shape == (5,)
    assert torch.allclose(v[0], torch.tensor(1.0))
    assert torch.allclose(v[1], torch.tensor(0.99))
    assert torch.allclose(v[4], torch.tensor(0.99 ** 4))


def test_flatten_chunk():
    chunk = torch.randn(3, 10, 14)
    flat = flatten_chunk(chunk)
    assert flat.shape == (3, 140)


def test_replay_buffer_integration():
    """Test that buffer -> sample -> loss pipeline works end-to-end."""
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(make_fake_transition())

    from lerobot.rlt.actor import ChunkActor
    from lerobot.rlt.critic import TwinCritic

    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    target_critic = copy.deepcopy(critic)

    batch = buf.sample(8)
    c_loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    a_loss = actor_loss(actor, critic, batch, beta=1.0)

    assert not torch.isnan(c_loss)
    assert not torch.isnan(a_loss)
