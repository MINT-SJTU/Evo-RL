from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lerobot.rlt.interfaces import Observation, VLAOutput


class VLAAdapter(ABC, nn.Module):
    """Abstract adapter for any chunked VLA model.

    One forward pass returns both token embeddings and the sampled action chunk,
    avoiding redundant VLA inference.
    """

    @abstractmethod
    def forward_vla(self, obs: Observation) -> VLAOutput:
        """Single forward pass returning final token embeddings + sampled action chunk."""
        ...

    @abstractmethod
    def supervised_loss(self, obs: Observation, expert_actions: torch.Tensor) -> torch.Tensor:
        """Action prediction loss for optional VLA fine-tuning on demos."""
        ...

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Dimension of each token in final_tokens."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Per-timestep action dimension."""
        ...


class DummyVLAAdapter(VLAAdapter):
    """Random-output adapter for shape testing and development."""

    def __init__(
        self,
        token_dim: int = 2048,
        num_tokens: int = 64,
        action_dim: int = 14,
        horizon: int = 50,
    ):
        super().__init__()
        self._token_dim = token_dim
        self._action_dim = action_dim
        self._num_tokens = num_tokens
        self._horizon = horizon
        # Dummy parameter so .parameters() is non-empty
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward_vla(self, obs: Observation) -> VLAOutput:
        B = obs.proprio.shape[0]
        device = obs.proprio.device
        return VLAOutput(
            final_tokens=torch.randn(B, self._num_tokens, self._token_dim, device=device),
            sampled_action_chunk=torch.randn(B, self._horizon, self._action_dim, device=device),
        )

    def supervised_loss(self, obs: Observation, expert_actions: torch.Tensor) -> torch.Tensor:
        vla_out = self.forward_vla(obs)
        # Dummy loss: MSE between predicted chunk (truncated) and expert
        T = min(vla_out.sampled_action_chunk.shape[1], expert_actions.shape[1])
        return ((vla_out.sampled_action_chunk[:, :T] - expert_actions[:, :T]) ** 2).mean()

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim
