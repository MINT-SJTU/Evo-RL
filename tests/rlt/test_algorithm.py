from __future__ import annotations

import torch

from tests.rlt.helpers import make_batch, make_test_algorithm


def test_build_rl_token_full():
    algorithm, _ = make_test_algorithm()
    full = algorithm.build_rl_token_full("cpu")
    assert hasattr(full, "decoder"), "Full rl_token should have a decoder"
    assert hasattr(full, "out_proj"), "Full rl_token should have out_proj"
    # Encoder weights should match
    for (n1, p1), (n2, p2) in zip(
        algorithm.policy.rl_token.encoder.named_parameters(),
        full.encoder.named_parameters(),
    ):
        assert torch.equal(p1.data, p2.data), f"Encoder param {n1} mismatch"
    assert torch.equal(algorithm.policy.rl_token.rl_token_embed.data, full.rl_token_embed.data)


def test_sync_encoder_from_full():
    algorithm, _ = make_test_algorithm()
    full = algorithm.build_rl_token_full("cpu")
    # Modify full encoder
    with torch.no_grad():
        for p in full.encoder.parameters():
            p.add_(1.0)
        full.rl_token_embed.add_(1.0)
    # Sync back
    algorithm.sync_encoder_from_full(full)
    for (n1, p1), (n2, p2) in zip(
        algorithm.policy.rl_token.encoder.named_parameters(),
        full.encoder.named_parameters(),
    ):
        assert torch.equal(p1.data, p2.data), f"Param {n1} not synced"
    assert torch.equal(algorithm.policy.rl_token.rl_token_embed.data, full.rl_token_embed.data)


def test_critic_update():
    algorithm, _ = make_test_algorithm()
    batch = make_batch()
    optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=1e-3)
    loss = algorithm.critic_update(batch, optimizer, gamma=0.99, C=4)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))


def test_actor_update():
    algorithm, _ = make_test_algorithm()
    batch = make_batch()
    optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=1e-3)
    loss = algorithm.actor_update(batch, optimizer, beta=1.0)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
