from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.rlt.utils import build_mlp, _get_activation


class ResidualMLP(nn.Module):
    """MLP with residual connections between hidden layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block_layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(_get_activation(activation))
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


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
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        if residual:
            self.net = ResidualMLP(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        else:
            self.net = build_mlp(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
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
