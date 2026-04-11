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



def compute_discount_vector(gamma: float, length: int, device: torch.device | None = None) -> torch.Tensor:
    """Return [1, gamma, gamma^2, ..., gamma^(length-1)]."""
    return gamma ** torch.arange(length, device=device, dtype=torch.float32)


def _get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    return activations[name]()


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    activation: str = "relu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build an MLP with configurable activation and optional LayerNorm."""
    layers: list[nn.Module] = []
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_get_activation(activation))
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def filter_encoder_only(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Keep only encoder keys from an rl_token state_dict.

    Drops decoder.* and out_proj.* keys which are only needed during demo
    adaptation training, not inference.

    Returns:
        filtered: state_dict with encoder-only keys
        skipped: list of dropped key names
    """
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k.startswith("decoder.") or k.startswith("out_proj."):
            skipped.append(k)
        else:
            filtered[k] = v
    return filtered, skipped
