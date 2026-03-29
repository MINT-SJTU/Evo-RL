from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.rlt.actor import ResidualMLP
from lerobot.rlt.utils import build_mlp


class ChunkCritic(nn.Module):
    """Single Q-network for chunk-level actions."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        if residual:
            self.net = ResidualMLP(
                state_dim + chunk_dim, hidden_dim, 1, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        else:
            self.net = build_mlp(
                state_dim + chunk_dim, hidden_dim, 1, num_layers,
                activation=activation, layer_norm=layer_norm,
            )

    def forward(self, state_vec: torch.Tensor, action_flat: torch.Tensor) -> torch.Tensor:
        """Returns Q-value (B, 1)."""
        return self.net(torch.cat([state_vec, action_flat], dim=-1))


class TwinCritic(nn.Module):
    """Twin Q-networks for TD3-style clipped double Q-learning."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.q1 = ChunkCritic(
            state_dim, chunk_dim, hidden_dim, num_layers,
            activation=activation, layer_norm=layer_norm, residual=residual,
        )
        self.q2 = ChunkCritic(
            state_dim, chunk_dim, hidden_dim, num_layers,
            activation=activation, layer_norm=layer_norm, residual=residual,
        )

    def forward(
        self, state_vec: torch.Tensor, action_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1, q2), each (B, 1)."""
        return self.q1(state_vec, action_flat), self.q2(state_vec, action_flat)

    def min_q(self, state_vec: torch.Tensor, action_flat: torch.Tensor) -> torch.Tensor:
        """Element-wise minimum of twin Q-values, (B, 1)."""
        q1, q2 = self.forward(state_vec, action_flat)
        return torch.minimum(q1, q2)
