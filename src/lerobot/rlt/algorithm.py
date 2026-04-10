from __future__ import annotations

import copy

import torch

from lerobot.rlt.config import RLTConfig
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.losses import actor_loss, critic_loss
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import soft_update


class RLTAlgorithm:
    """Training wrapper: policy + critic + target critic.

    Not an nn.Module itself -- owns an RLTPolicy (nn.Module) and the
    critic/target-critic pair used only during training.
    """

    def __init__(self, policy: RLTPolicy, config: RLTConfig):
        self.policy = policy
        self.config = config

        token_dim = policy.vla.token_dim
        action_dim = policy.vla.action_dim
        chunk_dim = config.chunk_length * action_dim
        state_dim = token_dim + config.proprio_dim

        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic.hidden_dim,
            num_layers=config.critic.num_layers,
            activation=config.critic.activation,
            layer_norm=config.critic.layer_norm,
            residual=config.critic.residual,
        )

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Training update helpers
    # ------------------------------------------------------------------

    def critic_update(
        self,
        batch: dict[str, torch.Tensor],
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        C: int,
        grad_clip: float = 1.0,
    ) -> float:
        """Single critic gradient step with gradient clipping."""
        loss = critic_loss(self.critic, self.target_critic, self.policy.actor, batch, gamma, C)
        critic_optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        critic_optimizer.step()
        return loss.item()

    def actor_update(
        self,
        batch: dict[str, torch.Tensor],
        actor_optimizer: torch.optim.Optimizer,
        beta: float,
        grad_clip: float = 1.0,
    ) -> float:
        """Single actor gradient step with gradient clipping."""
        loss = actor_loss(self.policy.actor, self.critic, batch, beta)
        actor_optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), grad_clip)
        actor_optimizer.step()
        return loss.item()

    def soft_update_target(self, tau: float) -> None:
        """Polyak-average online critic into target critic."""
        soft_update(self.target_critic, self.critic, tau)

    # ------------------------------------------------------------------
    # RL Token full model helpers (for demo adaptation)
    # ------------------------------------------------------------------

    def build_rl_token_full(self, device: torch.device | str) -> RLTokenModule:
        """Create a full rl_token (with decoder) and copy encoder weights from policy."""
        cfg = self.config.rl_token
        rl_token_full = RLTokenModule(
            token_dim=cfg.token_dim,
            nhead=cfg.nhead,
            num_enc_layers=cfg.enc_layers,
            num_dec_layers=cfg.dec_layers,
            ff_dim=cfg.ff_dim,
            num_rl_tokens=cfg.num_rl_tokens,
            inference_only=False,
        ).to(device)
        self.sync_full_from_encoder(rl_token_full)
        return rl_token_full

    def sync_full_from_encoder(self, rl_token_full: RLTokenModule) -> None:
        """Copy encoder weights + rl_token_embed from policy.rl_token into a full module."""
        rl_token_full.encoder.load_state_dict(self.policy.rl_token.encoder.state_dict())
        rl_token_full.rl_token_embed.data.copy_(self.policy.rl_token.rl_token_embed.data)

    def sync_encoder_from_full(self, rl_token_full: RLTokenModule) -> None:
        """Copy encoder weights + rl_token_embed back from full module to policy.rl_token."""
        self.policy.rl_token.encoder.load_state_dict(rl_token_full.encoder.state_dict())
        self.policy.rl_token.rl_token_embed.data.copy_(rl_token_full.rl_token_embed.data)

    # ------------------------------------------------------------------
    # Device / mode helpers
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> RLTAlgorithm:
        """Move all sub-modules to *device*."""
        self.policy.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        return self

    def train(self) -> None:
        """Set all sub-modules to training mode."""
        self.policy.train()
        self.critic.train()
        self.target_critic.train()

    def eval(self) -> None:
        """Set all sub-modules to eval mode."""
        self.policy.eval()
        self.critic.eval()
        self.target_critic.eval()

    def parameters(self):
        """Iterate over all trainable parameters (policy + critic)."""
        yield from self.policy.parameters()
        yield from self.critic.parameters()
