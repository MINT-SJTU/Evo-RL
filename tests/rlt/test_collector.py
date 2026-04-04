from __future__ import annotations

import torch
import pytest

from lerobot.rlt.collector import DummyEnvironment, warmup_collect, rl_collect_step
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.vla_adapter import DummyVLAAdapter


def _make_policy(token_dim=64, action_dim=7, proprio_dim=7):
    cfg = RLTConfig(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        chunk_length=4,
        vla_horizon=10,
        rl_token=RLTConfig.__dataclass_fields__["rl_token"].default_factory(),
    )
    cfg.rl_token.token_dim = token_dim
    cfg.rl_token.nhead = 4
    cfg.rl_token.enc_layers = 1
    cfg.rl_token.dec_layers = 1
    cfg.rl_token.ff_dim = 128

    cfg.actor.hidden_dim = 32
    cfg.actor.num_layers = 1

    vla = DummyVLAAdapter(token_dim=token_dim, num_tokens=8, action_dim=action_dim, horizon=10)
    return RLTPolicy(cfg, vla)


def test_warmup_fills_buffer():
    policy = _make_policy()
    env = DummyEnvironment(proprio_dim=7, action_dim=7)
    buf = ReplayBuffer(capacity=100)

    steps = warmup_collect(env, policy, buf, num_steps=20, chunk_length=4)
    assert len(buf) > 0
    assert steps >= 20


def test_rl_collect_step():
    policy = _make_policy()
    env = DummyEnvironment(proprio_dim=7, action_dim=7)
    buf = ReplayBuffer(capacity=100)

    obs = env.reset()
    next_obs, done, steps = rl_collect_step(env, policy, obs, buf, chunk_length=4)

    assert len(buf) == 1
    assert steps == 4
    assert next_obs.proprio.shape[1] == 7


def test_dummy_environment():
    env = DummyEnvironment(proprio_dim=14, action_dim=14)
    obs = env.reset()
    assert obs.proprio.shape == (1, 14)

    action = torch.randn(14)
    next_obs, reward, done, info = env.step(action)
    assert next_obs.proprio.shape == (1, 14)
    assert isinstance(reward, float)
