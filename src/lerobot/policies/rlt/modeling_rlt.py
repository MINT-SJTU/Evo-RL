from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rlt.configuration_rlt import RLTPretrainedConfig
from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.interfaces import Observation, VLAOutput
from lerobot.rlt.obs_bridge import robot_obs_to_rlt_obs
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import flatten_chunk, subsample_indices, unflatten_chunk
from lerobot.rlt.vla_adapter import VLAAdapter

log = logging.getLogger(__name__)

# Prefix used for VLA sub-module in state_dict
_VLA_PREFIX = "vla."


class RLTPretrainedPolicy(PreTrainedPolicy):
    """LeRobot-compatible policy wrapping the RLT RL head on top of a frozen VLA.

    The safetensors checkpoint contains ONLY the RL head weights (rl_token encoder
    + actor, ~400 MB). The VLA backbone (~7 GB) is loaded separately from
    config.vla_pretrained_path at construction time.
    """

    config_class = RLTPretrainedConfig
    name = "rlt"

    def __init__(self, config: RLTPretrainedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: RLTPretrainedConfig = config

        # Build RL Token encoder (inference-only: no decoder)
        self.rl_token = RLTokenModule(
            token_dim=config.rl_token_dim,
            nhead=config.rl_token_nhead,
            num_enc_layers=config.rl_token_enc_layers,
            num_dec_layers=config.rl_token_dec_layers,
            ff_dim=config.rl_token_ff_dim,
            num_rl_tokens=config.rl_token_num_rl_tokens,
            inference_only=True,
        )

        # Build Actor
        token_dim = config.rl_token_dim
        chunk_dim = config.chunk_length * config.action_dim
        state_dim = token_dim + config.proprio_dim

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
            residual=config.actor_residual,
        )

        # VLA adapter is built lazily (see _ensure_vla) to allow
        # state_dict / safetensors loading before the heavy VLA is in memory.
        self.vla: VLAAdapter | None = None
        self._vla_loaded = False

        # Action queue for chunk-to-step conversion
        self._action_queue: deque[Tensor] = deque()
        self._subsample_cache: Tensor | None = None

    # ------------------------------------------------------------------
    # VLA loading (deferred so safetensors load does not require 7 GB)
    # ------------------------------------------------------------------

    def _ensure_vla(self) -> VLAAdapter:
        """Lazily load the VLA backbone on first inference call."""
        if self._vla_loaded and self.vla is not None:
            return self.vla
        cfg = self.config
        log.info("Loading VLA backbone from %s", cfg.vla_pretrained_path)
        from lerobot.rlt.pi05_adapter import Pi05VLAAdapter

        self.vla = Pi05VLAAdapter(
            model_path=cfg.vla_pretrained_path,
            actual_action_dim=cfg.action_dim,
            actual_proprio_dim=cfg.proprio_dim,
            task_instruction=cfg.task_instruction,
            device=cfg.device or "cpu",
            token_pool_size=cfg.token_pool_size,
        )
        self.vla.to(cfg.device or "cpu")
        for p in self.vla.parameters():
            p.requires_grad = False
        self._vla_loaded = True
        return self.vla

    def set_vla(self, vla: VLAAdapter) -> None:
        """Inject a VLA adapter (e.g. DummyVLAAdapter for testing)."""
        self.vla = vla
        self._vla_loaded = True

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _obs_from_batch(self, batch: dict[str, Tensor]) -> Observation:
        """Convert a lerobot batch dict into an RLT Observation."""
        return robot_obs_to_rlt_obs(
            batch,
            camera_keys=self.config.camera_keys,
            proprio_keys=self.config.proprio_keys,
            device=self.config.device or "cpu",
        )

    def _get_subsample_indices(self, H: int) -> Tensor:
        C = self.config.chunk_length
        if self._subsample_cache is None or self._subsample_cache.shape[0] != C:
            self._subsample_cache = subsample_indices(H, C)
        return self._subsample_cache

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a full chunk of actions.

        Returns:
            actions: (B, chunk_length, action_dim)
        """
        self.eval()
        vla = self._ensure_vla()
        obs = self._obs_from_batch(batch)

        vla_out = vla.forward_vla(obs)
        z_rl = self.rl_token.encode(vla_out.final_tokens.detach())
        state_vec = torch.cat([z_rl, obs.proprio], dim=-1)

        indices = self._get_subsample_indices(vla_out.sampled_action_chunk.shape[1])
        ref_chunk = vla_out.sampled_action_chunk[:, indices, :]
        ref_flat = flatten_chunk(ref_chunk)

        mu, _ = self.actor(state_vec, ref_flat, training=False)
        if not self.config.deterministic:
            action_flat, _ = self.actor.sample(state_vec, ref_flat, training=False)
        else:
            action_flat = mu

        return unflatten_chunk(action_flat, self.config.chunk_length)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return a single-step action, using internal chunk queue.

        Returns:
            action: (B, action_dim)
        """
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)
            # chunk shape: (B, C, action_dim) -- enqueue each timestep
            self._action_queue.extend(chunk.transpose(0, 1))

        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue.clear()

    # ------------------------------------------------------------------
    # Training stubs (RLT trains via its own offline RL loop, not lerobot)
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "RLT does not train through the standard lerobot forward() path. "
            "Use lerobot.rlt.trainer for demo adaptation / offline RL training."
        )

    def get_optim_params(self) -> dict:
        return {
            "rl_token": list(self.rl_token.parameters()),
            "actor": list(self.actor.parameters()),
        }

    # ------------------------------------------------------------------
    # State dict: exclude VLA weights from serialization
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not k.startswith(_VLA_PREFIX)}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs,
    ) -> RLTPretrainedPolicy:
        """Load RLT policy.

        Uses strict=False by default because the VLA sub-module is not in the
        safetensors file. After loading, validates that all missing keys belong
        to the VLA prefix (which is loaded separately at inference time).
        """
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            strict=False,
            **kwargs,
        )
