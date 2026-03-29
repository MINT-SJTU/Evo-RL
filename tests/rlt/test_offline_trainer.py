from __future__ import annotations

import math

import torch

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.interfaces import ChunkTransition
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.trainer import offline_rl_loop
from lerobot.rlt.vla_adapter import DummyVLAAdapter


TOKEN_DIM = 64
ACTION_DIM = 7
PROPRIO_DIM = 7
C = 4
STATE_DIM = TOKEN_DIM + PROPRIO_DIM


def _make_agent():
    cfg = RLTConfig(
        action_dim=ACTION_DIM, proprio_dim=PROPRIO_DIM,
        chunk_length=C, vla_horizon=10,
    )
    cfg.rl_token.token_dim = TOKEN_DIM
    cfg.rl_token.nhead = 4
    cfg.rl_token.enc_layers = 1
    cfg.rl_token.dec_layers = 1
    cfg.rl_token.ff_dim = 128
    cfg.rl_token.num_rl_tokens = 2
    cfg.actor.hidden_dim = 32
    cfg.actor.num_layers = 1
    cfg.critic.hidden_dim = 32
    cfg.critic.num_layers = 1
    cfg.training.batch_size = 8
    cfg.training.actor_update_interval = 2
    cfg.offline_rl.num_gradient_steps = 20
    cfg.offline_rl.log_every = 10
    cfg.offline_rl.eval_every = 10
    cfg.offline_rl.save_every = 50  # no save during test
    vla = DummyVLAAdapter(
        token_dim=TOKEN_DIM, num_tokens=8,
        action_dim=ACTION_DIM, horizon=10,
    )
    return RLTAgent(cfg, vla), cfg


def _make_fake_transition():
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


def _fill_buffer(n: int = 50) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200)
    for _ in range(n):
        buf.add(_make_fake_transition())
    return buf


def test_offline_rl_loop_runs():
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    buf = _fill_buffer()

    metrics = offline_rl_loop(agent, cfg, buf)

    assert len(metrics.critic_losses) == cfg.offline_rl.num_gradient_steps
    assert len(metrics.actor_losses) > 0
    assert all(not math.isnan(l) for l in metrics.critic_losses)
    assert all(not math.isnan(l) for l in metrics.actor_losses)


def test_offline_rl_loop_frozen_params():
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    vla_params_before = {n: p.data.clone() for n, p in agent.vla.named_parameters()}
    enc_params_before = {n: p.data.clone() for n, p in agent.rl_token.encoder.named_parameters()}
    rl_embed_before = agent.rl_token.rl_token_embed.data.clone()

    buf = _fill_buffer()
    offline_rl_loop(agent, cfg, buf)

    for name, p in agent.vla.named_parameters():
        assert torch.equal(p.data, vla_params_before[name]), f"VLA param {name} changed"
    for name, p in agent.rl_token.encoder.named_parameters():
        assert torch.equal(p.data, enc_params_before[name]), f"Encoder param {name} changed"
    assert torch.equal(agent.rl_token.rl_token_embed.data, rl_embed_before)


def test_offline_rl_loop_actor_critic_update():
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    actor_params_before = {n: p.data.clone() for n, p in agent.actor.named_parameters()}
    critic_params_before = {n: p.data.clone() for n, p in agent.critic.named_parameters()}

    buf = _fill_buffer()
    offline_rl_loop(agent, cfg, buf)

    actor_changed = any(
        not torch.equal(p.data, actor_params_before[n])
        for n, p in agent.actor.named_parameters()
    )
    critic_changed = any(
        not torch.equal(p.data, critic_params_before[n])
        for n, p in agent.critic.named_parameters()
    )
    assert actor_changed, "Actor params did not change after training"
    assert critic_changed, "Critic params did not change after training"


def test_offline_rl_loop_with_val_buffer():
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    train_buf = _fill_buffer()
    val_buf = _fill_buffer(20)

    metrics = offline_rl_loop(agent, cfg, train_buf, val_buffer=val_buf)

    assert len(metrics.critic_losses) == cfg.offline_rl.num_gradient_steps
