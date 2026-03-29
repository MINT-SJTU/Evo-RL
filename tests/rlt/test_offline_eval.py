from __future__ import annotations

import math

import torch

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.evaluator import evaluate_offline, OfflineEvalMetrics
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
    cfg.offline_rl.save_every = 50
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


def test_evaluate_offline_returns_metrics():
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    val_buf = _fill_buffer(30)

    metrics = evaluate_offline(agent, val_buf, cfg, num_batches=3)

    assert isinstance(metrics, OfflineEvalMetrics)
    assert not math.isnan(metrics.expert_action_mse)
    assert not math.isnan(metrics.ref_action_mse)
    assert not math.isnan(metrics.mean_q_policy)
    assert not math.isnan(metrics.mean_q_expert)
    assert not math.isnan(metrics.q_gap)
    assert not math.isnan(metrics.mean_critic_td_error)
    assert metrics.expert_action_mse >= 0.0
    assert metrics.ref_action_mse >= 0.0


def test_evaluate_offline_q_gap_sign():
    """After some training, q_gap = Q(policy) - Q(expert) should not be NaN."""
    agent, cfg = _make_agent()
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    buf = _fill_buffer(50)

    offline_rl_loop(agent, cfg, buf)

    val_buf = _fill_buffer(20)
    metrics = evaluate_offline(agent, val_buf, cfg, num_batches=5)

    assert not math.isnan(metrics.q_gap)
    # q_gap is mean_q_policy - mean_q_expert: should be a finite number
    assert math.isfinite(metrics.q_gap)
