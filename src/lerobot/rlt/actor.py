from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.rlt.utils import build_mlp


class ChunkActor(nn.Module):
    """Actor that predicts an action chunk conditioned on RL state and VLA reference.

    Uses binary reference dropout (per batch element) during training to avoid
    over-reliance on the VLA reference chunk.
    """

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
        ref_dropout_p: float = 0.5,
    ):
        super().__init__()
        self.net = build_mlp(state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers)
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, std)."""
        if training:
            mask = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device) > self.ref_dropout_p
            ).float()
            ref_chunk_flat = ref_chunk_flat * mask
        x = torch.cat([state_vec, ref_chunk_flat], dim=-1)
        mu = self.net(x)
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with Gaussian noise. Returns (action, mu)."""
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        return mu + std * torch.randn_like(std), mu
