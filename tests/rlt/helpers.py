"""Shared test helpers for RLT tests.

Centralizes repeated factory functions (algorithm, transitions, buffers)
used across test_algorithm, test_training_step, test_offline_trainer,
and test_offline_eval.
"""
from __future__ import annotations

import torch

from lerobot.rlt.algorithm import RLTAlgorithm
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.interfaces import ChunkTransition
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.vla_adapter import DummyVLAAdapter

TOKEN_DIM = 64
ACTION_DIM = 7
PROPRIO_DIM = 7
C = 4
STATE_DIM = TOKEN_DIM + PROPRIO_DIM
CHUNK_DIM = C * ACTION_DIM


def make_test_algorithm(
    *,
    batch_size: int = 8,
    actor_update_interval: int = 2,
    num_gradient_steps: int = 20,
    log_every: int = 10,
    eval_every: int = 10,
    save_every: int = 50,
) -> tuple[RLTAlgorithm, RLTConfig]:
    """Build a small test algorithm + config.

    Returns (algorithm, config) tuple.
    """
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
    cfg.training.batch_size = batch_size
    cfg.training.actor_update_interval = actor_update_interval
    cfg.offline_rl.num_gradient_steps = num_gradient_steps
    cfg.offline_rl.log_every = log_every
    cfg.offline_rl.eval_every = eval_every
    cfg.offline_rl.save_every = save_every

    vla = DummyVLAAdapter(
        token_dim=TOKEN_DIM, num_tokens=8,
        action_dim=ACTION_DIM, horizon=10,
    )
    policy = RLTPolicy(cfg, vla)
    return RLTAlgorithm(policy, cfg), cfg


def make_fake_transition() -> ChunkTransition:
    """Create a single random ChunkTransition for testing."""
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


def fill_buffer(n: int = 50, capacity: int = 200) -> ReplayBuffer:
    """Create a ReplayBuffer filled with n random transitions."""
    buf = ReplayBuffer(capacity=capacity)
    for _ in range(n):
        buf.add(make_fake_transition())
    return buf


def make_batch(B: int = 8) -> dict[str, torch.Tensor]:
    """Create a random batch dict matching replay buffer sample format."""
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
