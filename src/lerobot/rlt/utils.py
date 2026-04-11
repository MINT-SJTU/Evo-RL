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


def infer_actor_architecture(
    actor_state_dict: dict[str, torch.Tensor],
    *,
    default_activation: str = "relu",
    default_fixed_std: float = 0.05,
    default_ref_dropout_p: float = 0.5,
) -> dict[str, int | float | bool | str]:
    """Infer ChunkActor construction kwargs from a saved actor state_dict."""
    if "net.input_proj.weight" in actor_state_dict:
        hidden_dim = actor_state_dict["net.input_proj.weight"].shape[0]
        block_indices = {
            int(key.split(".")[2])
            for key in actor_state_dict
            if key.startswith("net.blocks.") and key.endswith(".0.weight")
        }
        layer_norm = any(
            key.startswith("net.blocks.") and key.endswith(".1.weight")
            for key in actor_state_dict
        )
        return {
            "hidden_dim": hidden_dim,
            "num_layers": len(block_indices),
            "activation": default_activation,
            "layer_norm": layer_norm,
            "residual": True,
            "fixed_std": default_fixed_std,
            "ref_dropout_p": default_ref_dropout_p,
        }

    linear_keys = sorted(
        key for key, value in actor_state_dict.items()
        if key.startswith("net.") and key.endswith(".weight") and value.ndim == 2
    )
    if not linear_keys:
        raise ValueError("Could not infer actor architecture: no linear weights found")

    hidden_dim = actor_state_dict[linear_keys[0]].shape[0]
    num_layers = len(linear_keys) - 1
    layer_norm = any(
        key.startswith("net.") and key.endswith(".weight") and value.ndim == 1
        for key, value in actor_state_dict.items()
    )
    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "activation": default_activation,
        "layer_norm": layer_norm,
        "residual": False,
        "fixed_std": default_fixed_std,
        "ref_dropout_p": default_ref_dropout_p,
    }
