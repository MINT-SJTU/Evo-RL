from __future__ import annotations

import torch
import pytest

from lerobot.rlt.envs.reaching import ReachingEnvironment
from lerobot.rlt.envs import make_env


def test_reaching_env_reset():
    """Reset returns valid Observation with correct shapes."""
    env = ReachingEnvironment()
    obs = env.reset()
    assert obs.proprio.shape == (1, 12)
    assert "base" in obs.images
    assert obs.images["base"].shape == (1, 3, 64, 64)


def test_reaching_env_step_shapes():
    """Step returns correct types and shapes."""
    env = ReachingEnvironment()
    env.reset()
    action = torch.zeros(12)
    obs, reward, done, info = env.step(action)
    assert obs.proprio.shape == (1, 12)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "distance" in info


def test_reaching_env_step_batched_action():
    """Step handles batched (1, 12) actions."""
    env = ReachingEnvironment()
    env.reset()
    action = torch.zeros(1, 12)
    obs, reward, done, info = env.step(action)
    assert obs.proprio.shape == (1, 12)


def test_reaching_env_reward_direction():
    """Moving toward target should increase reward."""
    env = ReachingEnvironment()
    env.reset()
    # Step with zero action
    _, r0, _, info0 = env.step(torch.zeros(12))
    # Step toward target
    direction = info0["target"] - env.position
    small_step = direction.clamp(-0.1, 0.1)
    _, r1, _, _ = env.step(small_step)
    assert r1 > r0  # closer = less negative = higher reward


def test_reaching_env_done_condition():
    """Setting position very close to target triggers done."""
    env = ReachingEnvironment()
    env.reset()
    env.position = env.target.clone()  # cheat: put on top of target
    _, _, done, _ = env.step(torch.zeros(12))
    assert done is True


def test_reaching_env_custom_dims():
    """Custom proprio_dim and action_dim work."""
    env = ReachingEnvironment(proprio_dim=6, action_dim=6)
    assert env.proprio_dim == 6
    assert env.action_dim == 6
    obs = env.reset()
    assert obs.proprio.shape == (1, 6)


def test_make_env_factory():
    """make_env creates correct environment types."""
    env = make_env("reaching")
    assert isinstance(env, ReachingEnvironment)
    env2 = make_env("dummy")
    assert env2.proprio_dim > 0
    with pytest.raises(ValueError):
        make_env("nonexistent")
