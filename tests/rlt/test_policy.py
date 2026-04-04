from __future__ import annotations

import torch
import pytest

from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.vla_adapter import DummyVLAAdapter
from lerobot.rlt.interfaces import Observation
from lerobot.rlt.phase_controller import PhaseController, Phase, HandoverClassifier
from lerobot.rlt.utils import soft_update


TOKEN_DIM = 64
ACTION_DIM = 7
PROPRIO_DIM = 7
C = 4


@pytest.fixture
def policy():
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

    vla = DummyVLAAdapter(token_dim=TOKEN_DIM, num_tokens=8, action_dim=ACTION_DIM, horizon=10)
    return RLTPolicy(cfg, vla)


@pytest.fixture
def obs():
    return Observation(
        images={"base": torch.randn(2, 3, 64, 64)},
        proprio=torch.randn(2, PROPRIO_DIM),
    )


def test_get_rl_state_shape(policy, obs):
    state = policy.get_rl_state(obs)
    assert state.shape == (2, TOKEN_DIM + PROPRIO_DIM)


def test_get_reference_chunk_shape(policy, obs):
    ref = policy.get_reference_chunk(obs)
    assert ref.shape == (2, C, ACTION_DIM)


def test_select_action_shape(policy, obs):
    action, mu, state_vec, ref_chunk = policy.select_action(obs)
    assert action.shape == (2, C, ACTION_DIM)
    assert mu.shape == (2, C, ACTION_DIM)
    assert state_vec.shape[0] == 2
    assert ref_chunk.shape == (2, C, ACTION_DIM)


def test_freeze_vla(policy):
    policy.freeze_vla()
    for p in policy.vla.parameters():
        assert not p.requires_grad


def test_freeze_rl_token_encoder(policy):
    policy.freeze_rl_token_encoder()
    assert not policy.rl_token.rl_token_embed.requires_grad
    for p in policy.rl_token.encoder.parameters():
        assert not p.requires_grad
    # Policy rl_token is inference_only, so no decoder to check


def test_soft_update_correctness():
    """Verify soft_update computes correct Polyak average."""
    from lerobot.rlt.critic import ChunkCritic

    source = ChunkCritic(state_dim=10, chunk_dim=10, hidden_dim=8, num_layers=1)
    target = ChunkCritic(state_dim=10, chunk_dim=10, hidden_dim=8, num_layers=1)

    # Store original target params
    orig_params = [p.data.clone() for p in target.parameters()]

    tau = 0.1
    soft_update(target, source, tau)

    for orig, tp, sp in zip(orig_params, target.parameters(), source.parameters()):
        expected = (1.0 - tau) * orig + tau * sp.data
        assert torch.allclose(tp.data, expected, atol=1e-6)


class TestPhaseController:
    def test_initial_phase(self):
        pc = PhaseController(mode="manual")
        assert pc.phase == Phase.VLA_PHASE
        assert not pc.is_critical

    def test_manual_trigger(self):
        pc = PhaseController(mode="manual")
        pc.trigger_critical()
        assert pc.is_critical
        pc.trigger_vla()
        assert not pc.is_critical

    def test_reset(self):
        pc = PhaseController(mode="manual")
        pc.trigger_critical()
        pc.reset()
        assert pc.phase == Phase.VLA_PHASE

    def test_learned_mode(self):
        classifier = HandoverClassifier(input_dim=64, hidden_dim=32)
        pc = PhaseController(mode="learned", classifier=classifier)

        # Force classifier to predict critical
        z_rl = torch.randn(1, 64)
        # Just test that update runs without error
        phase = pc.update(z_rl)
        assert isinstance(phase, Phase)

    def test_classifier_training(self):
        classifier = HandoverClassifier(input_dim=64, hidden_dim=32)
        pc = PhaseController(mode="learned", classifier=classifier)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        z_rl = torch.randn(8, 64)
        labels = torch.randint(0, 2, (8,))
        loss = pc.train_classifier(z_rl, labels, optimizer)
        assert isinstance(loss, float)
        assert loss > 0
