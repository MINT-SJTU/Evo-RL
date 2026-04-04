"""Tests for RLT LeRobot ecosystem integration.

Validates config serialization, state_dict exclusion, model construction,
and action selection using DummyVLAAdapter (no real pi0.5 required).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import draccus
import pytest
import torch

from lerobot.policies.rlt.configuration_rlt import RLTPretrainedConfig
from lerobot.policies.rlt.modeling_rlt import RLTPretrainedPolicy, _VLA_PREFIX
from lerobot.rlt.vla_adapter import DummyVLAAdapter


def _make_policy_with_dummy_vla(
    action_dim: int = 12,
    proprio_dim: int = 12,
    chunk_length: int = 10,
    device: str = "cpu",
) -> RLTPretrainedPolicy:
    """Build an RLTPretrainedPolicy with a DummyVLAAdapter for testing."""
    cfg = RLTPretrainedConfig(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        chunk_length=chunk_length,
        device=device,
        # Simplify camera/proprio keys for testing
        camera_keys=["cam0"],
        proprio_keys=["proprio"],
        action_keys=[f"a{i}" for i in range(action_dim)],
    )
    policy = RLTPretrainedPolicy(cfg)
    dummy_vla = DummyVLAAdapter(
        token_dim=cfg.rl_token_dim,
        num_tokens=64,
        action_dim=action_dim,
        horizon=50,
    )
    policy.set_vla(dummy_vla)
    policy.eval()
    return policy


def _make_batch(
    action_dim: int = 12,
    proprio_dim: int = 12,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a minimal batch dict with the keys the policy expects."""
    return {
        "cam0": torch.randn(1, 3, 224, 224, device=device),
        "proprio": torch.randn(1, proprio_dim, device=device),
    }


class TestConfigSerialization:
    def test_round_trip_json(self):
        """Config should survive save/load via PreTrainedConfig.from_pretrained."""
        cfg = RLTPretrainedConfig(
            vla_pretrained_path="lerobot/pi05_base",
            task_instruction="insert screw",
            rl_token_num_rl_tokens=4,
            actor_residual=True,
            phase_mode="always_rl",
            action_dim=12,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save_pretrained(tmpdir)

            # Verify config.json was written
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()
            with open(config_path) as f:
                data = json.load(f)
            assert data["type"] == "rlt"

            # Load via the standard from_pretrained path
            from lerobot.configs.policies import PreTrainedConfig
            loaded = PreTrainedConfig.from_pretrained(tmpdir)

        assert loaded.vla_pretrained_path == "lerobot/pi05_base"
        assert loaded.task_instruction == "insert screw"
        assert loaded.rl_token_num_rl_tokens == 4
        assert loaded.actor_residual is True
        assert loaded.phase_mode == "always_rl"
        assert loaded.action_dim == 12

    def test_type_field(self):
        """Config type should be 'rlt' (from draccus registry)."""
        cfg = RLTPretrainedConfig(device="cpu")
        assert cfg.type == "rlt"

    def test_invalid_phase_mode_raises(self):
        with pytest.raises(ValueError, match="phase_mode"):
            RLTPretrainedConfig(phase_mode="invalid", device="cpu")


class TestStateDictExclusion:
    def test_state_dict_excludes_vla_keys(self):
        """state_dict() must not contain any vla.* keys."""
        policy = _make_policy_with_dummy_vla()
        sd = policy.state_dict()
        vla_keys = [k for k in sd if k.startswith(_VLA_PREFIX)]
        assert len(vla_keys) == 0, f"VLA keys leaked into state_dict: {vla_keys}"

    def test_state_dict_contains_rl_head(self):
        """state_dict() must contain rl_token.* and actor.* keys."""
        policy = _make_policy_with_dummy_vla()
        sd = policy.state_dict()
        rl_token_keys = [k for k in sd if k.startswith("rl_token.")]
        actor_keys = [k for k in sd if k.startswith("actor.")]
        assert len(rl_token_keys) > 0, "Missing rl_token keys"
        assert len(actor_keys) > 0, "Missing actor keys"


class TestModelConstruction:
    def test_from_config(self):
        """Policy should be constructible from config alone."""
        policy = _make_policy_with_dummy_vla()
        assert isinstance(policy, RLTPretrainedPolicy)
        assert isinstance(policy.rl_token, torch.nn.Module)
        assert isinstance(policy.actor, torch.nn.Module)

    def test_rl_token_inference_only(self):
        """RL token should NOT have decoder in inference mode."""
        policy = _make_policy_with_dummy_vla()
        assert not hasattr(policy.rl_token, "decoder") or policy.rl_token.inference_only


class TestSelectAction:
    def test_select_action_shape(self):
        """select_action should return (B, action_dim)."""
        policy = _make_policy_with_dummy_vla()
        batch = _make_batch()
        action = policy.select_action(batch)
        assert action.shape == (1, 12)

    def test_predict_action_chunk_shape(self):
        """predict_action_chunk should return (B, chunk_length, action_dim)."""
        policy = _make_policy_with_dummy_vla()
        batch = _make_batch()
        chunk = policy.predict_action_chunk(batch)
        assert chunk.shape == (1, 10, 12)

    def test_action_queue_drains(self):
        """After one predict, select_action should yield chunk_length actions."""
        policy = _make_policy_with_dummy_vla(chunk_length=5)
        batch = _make_batch()
        actions = [policy.select_action(batch) for _ in range(5)]
        assert len(actions) == 5
        for a in actions:
            assert a.shape == (1, 12)

    def test_reset_clears_queue(self):
        """reset() should clear the action queue."""
        policy = _make_policy_with_dummy_vla()
        batch = _make_batch()
        policy.select_action(batch)
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0


class TestSaveLoad:
    def test_save_and_load_roundtrip(self):
        """Save a policy, reload it, verify weights match."""
        policy = _make_policy_with_dummy_vla()
        sd_before = {k: v.clone() for k, v in policy.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "model.safetensors").exists()
            assert (Path(tmpdir) / "config.json").exists()

            # Check config.json has correct type
            with open(Path(tmpdir) / "config.json") as f:
                config_data = json.load(f)
            assert config_data["type"] == "rlt"

            # Reload
            loaded = RLTPretrainedPolicy.from_pretrained(tmpdir, device="cpu")
            sd_after = loaded.state_dict()

        # Same keys
        assert set(sd_before.keys()) == set(sd_after.keys())

        # Same values
        for key in sd_before:
            assert torch.allclose(sd_before[key], sd_after[key]), f"Mismatch on {key}"


class TestValidateFeatures:
    def test_validate_features_populates(self):
        cfg = RLTPretrainedConfig(device="cpu")
        cfg.input_features = {}
        cfg.output_features = {}
        cfg.validate_features()
        assert "observation.state" in cfg.input_features
        assert "action" in cfg.output_features
