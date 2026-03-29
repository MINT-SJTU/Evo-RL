from __future__ import annotations

import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = (1-tau)*target + tau*source."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def flatten_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Flatten action chunk from (B, C, action_dim) to (B, C*action_dim)."""
    return chunk.flatten(start_dim=-2)


def unflatten_chunk(flat: torch.Tensor, chunk_length: int) -> torch.Tensor:
    """Unflatten from (B, C*action_dim) to (B, C, action_dim)."""
    B = flat.shape[0]
    action_dim = flat.shape[-1] // chunk_length
    return flat.view(B, chunk_length, action_dim)


def subsample_indices(source_len: int, target_len: int) -> torch.Tensor:
    """Compute integer indices to subsample from source_len to target_len via linspace."""
    if source_len == target_len:
        return torch.arange(target_len)
    return torch.linspace(0, source_len - 1, target_len).long()


def compute_discount_vector(gamma: float, length: int, device: torch.device | None = None) -> torch.Tensor:
    """Return [1, gamma, gamma^2, ..., gamma^(length-1)]."""
    return gamma ** torch.arange(length, device=device, dtype=torch.float32)


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """Build a simple MLP: [Linear+ReLU]*num_layers + Linear."""
    layers: list[nn.Module] = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)
