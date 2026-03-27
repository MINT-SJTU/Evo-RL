from __future__ import annotations

import torch
import pytest

from lerobot.rlt.rl_token import RLTokenModule


@pytest.fixture
def rl_token():
    return RLTokenModule(token_dim=64, nhead=4, num_enc_layers=1, num_dec_layers=1, ff_dim=128)


def test_encode_shape(rl_token):
    tokens = torch.randn(3, 8, 64)
    z_rl = rl_token.encode(tokens)
    assert z_rl.shape == (3, 64)


def test_decode_shape(rl_token):
    z_rl = torch.randn(3, 64)
    teacher = torch.randn(3, 8, 64)
    pred = rl_token.decode(z_rl, teacher)
    assert pred.shape == (3, 8, 64)


def test_reconstruction_loss_scalar(rl_token):
    tokens = torch.randn(3, 8, 64)
    loss = rl_token.reconstruction_loss(tokens)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_reconstruction_loss_decreases(rl_token):
    """Verify that reconstruction loss decreases on fixed data after optimization."""
    tokens = torch.randn(4, 8, 64)
    optimizer = torch.optim.Adam(rl_token.parameters(), lr=1e-3)

    initial_loss = rl_token.reconstruction_loss(tokens).item()
    for _ in range(50):
        loss = rl_token.reconstruction_loss(tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = rl_token.reconstruction_loss(tokens).item()

    assert final_loss < initial_loss


def test_gradients_flow_to_encoder_decoder(rl_token):
    """Verify gradients flow to encoder/decoder params but not input tokens."""
    tokens = torch.randn(2, 8, 64, requires_grad=True)
    loss = rl_token.reconstruction_loss(tokens)
    loss.backward()

    # Input tokens are detached inside reconstruction_loss, so no grad
    assert tokens.grad is None or (tokens.grad == 0).all()

    # Encoder/decoder params should have gradients
    assert rl_token.rl_token_embed.grad is not None
    for p in rl_token.encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_causal_mask_applied(rl_token):
    """Verify causal masking by checking decode doesn't leak future information.

    We modify a teacher token at position k and verify that output positions
    before k (in shifted-input terms) are unaffected.
    """
    rl_token.eval()  # disable dropout for deterministic comparison

    z_rl = torch.randn(1, 64)
    teacher = torch.randn(1, 8, 64)

    pred1 = rl_token.decode(z_rl, teacher)

    # Modify teacher token at position 3. In shifted input, positions 0..3 are
    # [z_rl, teacher_0, teacher_1, teacher_2], which are unchanged.
    # So output positions 0..3 should be identical.
    teacher_modified = teacher.clone()
    teacher_modified[:, 3, :] = torch.randn(1, 64)

    pred2 = rl_token.decode(z_rl, teacher_modified)

    # Positions 0..3 should be identical (causal mask: they can't attend to position 4+)
    assert torch.allclose(pred1[:, :4, :], pred2[:, :4, :], atol=1e-5)
    # Position 4+ may differ (since shifted input at position 4 = teacher[:, 3] which changed)
    # We don't assert they differ (they might or might not depending on weights)
