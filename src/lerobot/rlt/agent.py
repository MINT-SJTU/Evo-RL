from __future__ import annotations

import copy

import torch
import torch.nn as nn

from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.interfaces import Observation, VLAOutput
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import flatten_chunk, unflatten_chunk
from lerobot.rlt.vla_adapter import VLAAdapter


class RLTAgent(nn.Module):
    """RLT Agent facade tying VLA adapter + RL token + actor + critic.

    Provides high-level methods for action selection and state computation
    used by the collector and trainer.
    """

    def __init__(self, config: RLTConfig, vla: VLAAdapter):
        super().__init__()
        self.config = config
        self.vla = vla

        token_dim = vla.token_dim
        action_dim = vla.action_dim
        chunk_dim = config.chunk_length * action_dim
        state_dim = token_dim + config.proprio_dim

        self.rl_token = RLTokenModule(
            token_dim=config.rl_token.token_dim,
            nhead=config.rl_token.nhead,
            num_enc_layers=config.rl_token.enc_layers,
            num_dec_layers=config.rl_token.dec_layers,
            ff_dim=config.rl_token.ff_dim,
        )

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor.hidden_dim,
            num_layers=config.actor.num_layers,
            fixed_std=config.actor.fixed_std,
            ref_dropout_p=config.actor.ref_dropout_p,
        )

        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic.hidden_dim,
            num_layers=config.critic.num_layers,
        )

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self._subsample_indices: torch.Tensor | None = None

    def _get_subsample_indices(self, H: int) -> torch.Tensor:
        """Lazily compute subsample indices for H -> C."""
        C = self.config.chunk_length
        if self._subsample_indices is None or self._subsample_indices.shape[0] != C:
            self._subsample_indices = torch.linspace(0, H - 1, C).long()
        return self._subsample_indices

    def _extract_state_and_ref(
        self, obs: Observation, vla_out: VLAOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """From a single VLA output, extract state_vec and ref_chunk.

        Returns:
            state_vec: (B, state_dim)
            ref_chunk: (B, C, action_dim)
        """
        z_rl = self.rl_token.encode(vla_out.final_tokens.detach())
        state_vec = torch.cat([z_rl, obs.proprio], dim=-1)
        indices = self._get_subsample_indices(vla_out.sampled_action_chunk.shape[1])
        ref_chunk = vla_out.sampled_action_chunk[:, indices, :]
        return state_vec, ref_chunk

    def get_rl_state(self, obs: Observation) -> torch.Tensor:
        """Compute RL state: VLA forward -> RL token encode -> concat proprio.

        Returns:
            state_vec: (B, state_dim)
        """
        vla_out = self.vla.forward_vla(obs)
        z_rl = self.rl_token.encode(vla_out.final_tokens.detach())
        return torch.cat([z_rl, obs.proprio], dim=-1)

    def get_reference_chunk(self, obs: Observation) -> torch.Tensor:
        """Get VLA reference chunk, subsampled to length C.

        Returns:
            ref_chunk: (B, C, action_dim)
        """
        vla_out = self.vla.forward_vla(obs)
        indices = self._get_subsample_indices(vla_out.sampled_action_chunk.shape[1])
        return vla_out.sampled_action_chunk[:, indices, :]

    def select_action(
        self, obs: Observation, vla_out: VLAOutput | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full action selection pipeline with single VLA forward.

        Returns:
            action_chunk: (B, C, action_dim) -- sampled action
            mu_chunk: (B, C, action_dim) -- deterministic mean
            state_vec: (B, state_dim) -- RL state (for collector to store)
            ref_chunk: (B, C, action_dim) -- VLA reference (for collector to store)
        """
        if vla_out is None:
            vla_out = self.vla.forward_vla(obs)
        state_vec, ref_chunk = self._extract_state_and_ref(obs, vla_out)
        ref_flat = flatten_chunk(ref_chunk)
        action_flat, mu_flat = self.actor.sample(state_vec, ref_flat)
        C = self.config.chunk_length
        action_chunk = unflatten_chunk(action_flat, C)
        mu_chunk = unflatten_chunk(mu_flat, C)
        return action_chunk, mu_chunk, state_vec, ref_chunk

    def freeze_vla(self) -> None:
        """Freeze VLA parameters (after demo adaptation)."""
        for p in self.vla.parameters():
            p.requires_grad = False

    def freeze_rl_token_encoder(self) -> None:
        """Freeze RL token encoder (after demo adaptation)."""
        for p in self.rl_token.encoder.parameters():
            p.requires_grad = False
        self.rl_token.rl_token_embed.requires_grad = False
