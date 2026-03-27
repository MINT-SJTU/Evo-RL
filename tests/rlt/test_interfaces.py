from __future__ import annotations

import torch
import pytest

from lerobot.rlt.interfaces import Observation, VLAOutput, ChunkTransition
from lerobot.rlt.config import RLTConfig


def test_observation_creation():
    obs = Observation(
        images={"base": torch.randn(2, 3, 64, 64)},
        proprio=torch.randn(2, 14),
    )
    assert obs.proprio.shape == (2, 14)
    assert obs.images["base"].shape == (2, 3, 64, 64)
    assert obs.instruction_ids is None
    assert obs.timestamp is None


def test_vla_output_creation():
    out = VLAOutput(
        final_tokens=torch.randn(2, 64, 2048),
        sampled_action_chunk=torch.randn(2, 50, 14),
    )
    assert out.final_tokens.shape == (2, 64, 2048)
    assert out.sampled_action_chunk.shape == (2, 50, 14)
    assert out.extra == {}


def test_chunk_transition_creation():
    C, action_dim, state_dim = 10, 14, 2062
    t = ChunkTransition(
        state_vec=torch.randn(state_dim),
        exec_chunk=torch.randn(C, action_dim),
        ref_chunk=torch.randn(C, action_dim),
        reward_seq=torch.randn(C),
        next_state_vec=torch.randn(state_dim),
        next_ref_chunk=torch.randn(C, action_dim),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(C),
    )
    assert t.exec_chunk.shape == (C, action_dim)
    assert t.done.shape == ()


def test_config_defaults():
    cfg = RLTConfig()
    assert cfg.action_dim == 14
    assert cfg.chunk_length == 10
    assert cfg.training.gamma == 0.99
    assert cfg.rl_token.token_dim == 2048
    assert cfg.rl_token.ff_dim == 4 * 2048


def test_config_from_yaml():
    import tempfile
    from pathlib import Path

    yaml_content = """
seed: 42
action_dim: 7
rl_token:
  token_dim: 512
  nhead: 4
  enc_layers: 2
  dec_layers: 2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = RLTConfig.from_yaml(f.name)

    assert cfg.seed == 42
    assert cfg.action_dim == 7
    assert cfg.rl_token.token_dim == 512
    assert cfg.rl_token.nhead == 4
    # ff_dim should auto-compute
    assert cfg.rl_token.ff_dim == 4 * 512


def test_config_default_yaml():
    cfg = RLTConfig.default()
    assert cfg.action_dim == 14
    assert cfg.training.utd_ratio == 5
