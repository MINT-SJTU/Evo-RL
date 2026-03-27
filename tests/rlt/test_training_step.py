from __future__ import annotations

import copy

import torch
import pytest

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.vla_adapter import DummyVLAAdapter
from lerobot.rlt.losses import critic_loss, actor_loss
from lerobot.rlt.utils import soft_update, flatten_chunk, compute_discount_vector
from lerobot.rlt.interfaces import Observation
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.interfaces import ChunkTransition

TOKEN_DIM = 64
ACTION_DIM = 7
PROPRIO_DIM = 7
C = 4
STATE_DIM = TOKEN_DIM + PROPRIO_DIM
CHUNK_DIM = C * ACTION_DIM


@pytest.fixture
def agent():
    cfg = RLTConfig(
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        chunk_length=C,
        vla_horizon=10,
    )
    cfg.rl_token.token_dim = TOKEN_DIM
    cfg.rl_token.nhead = 4
    cfg.rl_token.enc_layers = 1
    cfg.rl_token.dec_layers = 1
    cfg.rl_token.ff_dim = 128
    cfg.actor.hidden_dim = 32
    cfg.actor.num_layers = 1
    cfg.critic.hidden_dim = 32
    cfg.critic.num_layers = 1

    vla = DummyVLAAdapter(token_dim=TOKEN_DIM, num_tokens=8, action_dim=ACTION_DIM, horizon=10)
    return RLTAgent(cfg, vla)


@pytest.fixture
def batch():
    B = 8
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


def test_single_critic_step_loss_not_nan(agent, batch):
    """Single critic gradient step produces finite loss."""
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)
    loss = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify params changed
    loss2 = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss2)


def test_single_actor_step_loss_not_nan(agent, batch):
    """Single actor gradient step produces finite loss."""
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    loss = actor_loss(agent.actor, agent.critic, batch, beta=1.0)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_frozen_params_stay_frozen(agent, batch):
    """After freezing VLA and encoder, their params should not get gradients."""
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    # Run both losses
    c_loss = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, gamma=0.99, C=C)
    c_loss.backward()

    # VLA params: no grad
    for p in agent.vla.parameters():
        assert p.grad is None

    # RL token encoder params: no grad
    assert agent.rl_token.rl_token_embed.grad is None
    for p in agent.rl_token.encoder.parameters():
        assert p.grad is None


def test_full_forward_backward_pipeline(agent):
    """End-to-end: obs -> state -> action -> critic -> loss -> backward."""
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    obs = Observation(
        images={"base": torch.randn(4, 3, 64, 64)},
        proprio=torch.randn(4, PROPRIO_DIM),
    )

    with torch.no_grad():
        state = agent.get_rl_state(obs)
        ref = agent.get_reference_chunk(obs)
        ref_flat = flatten_chunk(ref)

    action, mu = agent.actor.sample(state, ref_flat, training=True)
    q = agent.critic.min_q(state, action)
    loss = -q.mean()
    loss.backward()

    # Actor should have gradients
    for p in agent.actor.parameters():
        assert p.grad is not None

    # Critic should have gradients (from Q computation)
    for p in agent.critic.parameters():
        assert p.grad is not None


def test_soft_update_changes_target(agent):
    """Verify soft update actually modifies target critic."""
    orig_params = [p.data.clone() for p in agent.target_critic.parameters()]

    # Perturb online critic
    for p in agent.critic.parameters():
        p.data += torch.randn_like(p.data) * 0.1

    soft_update(agent.target_critic, agent.critic, tau=0.1)

    changed = False
    for orig, p in zip(orig_params, agent.target_critic.parameters()):
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
        t = ChunkTransition(
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
        buf.add(t)

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
