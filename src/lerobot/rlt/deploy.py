from __future__ import annotations

import logging
import time
from collections import deque

import torch
import torch.nn as nn

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.config import ActorConfig, CriticConfig, RLTConfig, RLTokenConfig
from lerobot.rlt.deploy_config import DeployConfig
from lerobot.rlt.interfaces import Observation
from lerobot.rlt.obs_bridge import robot_obs_to_rlt_obs
from lerobot.rlt.phase_controller import Phase, PhaseController
from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
from lerobot.rlt.utils import subsample_indices

log = logging.getLogger(__name__)


class RLTDeployPolicy(nn.Module):
    """Deployment wrapper that converts LeRobot obs dicts to RLT actions.

    Maintains an internal action queue: runs VLA + actor once per chunk (C=10 steps),
    then pops one action per call to select_action().
    """

    def __init__(self, config: DeployConfig):
        super().__init__()
        self.config = config
        self._action_queue: deque[torch.Tensor] = deque()

        log.info("Loading VLA model from %s", config.vla_model_path)
        self.vla = Pi05VLAAdapter(
            model_path=config.vla_model_path,
            actual_action_dim=len(config.action_keys),
            actual_proprio_dim=len(config.proprio_keys),
            task_instruction=config.task_instruction,
            device=config.device,
            token_pool_size=config.token_pool_size,
        )

        rlt_config = _build_rlt_config(config, self.vla.action_dim)
        self.agent = RLTAgent(rlt_config, self.vla)
        self.agent.to(config.device)

        _load_checkpoints(self.agent, config)

        self.agent.freeze_vla()
        self.agent.freeze_rl_token_encoder()
        self.agent.eval()

        self.phase_controller = PhaseController(mode="manual")
        self._phase_mode = config.phase_mode

        self._timing: dict[str, float] = {}
        log.info(
            "RLTDeployPolicy ready: action_dim=%d, chunk=%d, phase=%s",
            self.vla.action_dim, config.chunk_length, config.phase_mode,
        )

    @torch.no_grad()
    def select_action(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single-step action from LeRobot observation dict.

        Returns:
            action: (1, action_dim) tensor
        """
        if not self._action_queue:
            self._fill_action_queue(obs_dict)
        return self._action_queue.popleft()

    def _fill_action_queue(self, obs_dict: dict[str, torch.Tensor]) -> None:
        """Run VLA + actor to produce a chunk of actions and enqueue them."""
        t0 = time.monotonic()
        obs = robot_obs_to_rlt_obs(
            obs_dict,
            camera_keys=self.config.camera_keys,
            proprio_keys=self.config.proprio_keys,
            device=self.config.device,
        )

        action_chunk = self._compute_action_chunk(obs)

        # Enqueue each timestep as (1, action_dim)
        for t in range(action_chunk.shape[1]):
            self._action_queue.append(action_chunk[:, t, :])

        elapsed = time.monotonic() - t0
        self._timing["last_chunk_compute_ms"] = elapsed * 1000
        log.debug("Chunk compute: %.1f ms (%d actions queued)", elapsed * 1000, len(self._action_queue))

    def _compute_action_chunk(self, obs: Observation) -> torch.Tensor:
        """Compute action chunk based on phase mode.

        Returns:
            action_chunk: (1, C, action_dim)
        """
        vla_out = self.vla.forward_vla(obs)

        if self._phase_mode == "always_vla":
            return self._vla_reference_chunk(vla_out)

        if self._phase_mode == "always_rl":
            return self._rl_action_chunk(obs, vla_out)

        # manual mode: use phase controller
        state_vec, ref_chunk = self.agent._extract_state_and_ref(obs, vla_out)
        self.phase_controller.update(z_rl=state_vec[:, :self.vla.token_dim])
        if self.phase_controller.phase == Phase.CRITICAL_PHASE:
            return self._rl_action_chunk(obs, vla_out)
        return self._vla_reference_chunk(vla_out)

    def _rl_action_chunk(self, obs: Observation, vla_out) -> torch.Tensor:
        """Run RL actor to get action chunk."""
        action_chunk, _, _, _ = self.agent.select_action(
            obs, vla_out=vla_out, deterministic=self.config.deterministic,
        )
        return action_chunk

    def _vla_reference_chunk(self, vla_out) -> torch.Tensor:
        """Extract VLA reference chunk, subsampled to C steps."""
        H = vla_out.sampled_action_chunk.shape[1]
        C = self.config.chunk_length
        indices = subsample_indices(H, C)
        return vla_out.sampled_action_chunk[:, indices, :]

    def reset(self) -> None:
        """Reset for a new episode."""
        self._action_queue.clear()
        self.phase_controller.reset()
        log.info("Episode reset")

    @property
    def timing(self) -> dict[str, float]:
        return dict(self._timing)


def _build_rlt_config(config: DeployConfig, action_dim: int) -> RLTConfig:
    """Build an RLTConfig from DeployConfig to construct the agent."""
    return RLTConfig(
        action_dim=action_dim,
        proprio_dim=len(config.proprio_keys),
        chunk_length=config.chunk_length,
        rl_token=RLTokenConfig(
            token_dim=2048,
            nhead=8,
            enc_layers=3,
            dec_layers=3,
            ff_dim=4096,
            num_rl_tokens=4,
        ),
        actor=ActorConfig(
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            residual=config.actor_residual,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
        ),
        critic=CriticConfig(
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            residual=config.actor_residual,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
        ),
    )


def _load_checkpoints(agent: RLTAgent, config: DeployConfig) -> None:
    """Load RL token and actor-critic checkpoints into the agent."""
    device = config.device

    if config.rl_token_checkpoint:
        log.info("Loading RL token checkpoint: %s", config.rl_token_checkpoint)
        ckpt = torch.load(config.rl_token_checkpoint, map_location=device, weights_only=True)
        agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])
        log.info("RL token loaded (trained %d steps)", ckpt.get("step", -1))

    if config.ac_checkpoint:
        log.info("Loading actor-critic checkpoint: %s", config.ac_checkpoint)
        ckpt = torch.load(config.ac_checkpoint, map_location=device, weights_only=True)
        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        agent.critic.load_state_dict(ckpt["critic_state_dict"])
        # Overwrite rl_token if checkpoint has it (v2 checkpoints include rl_token)
        if "rl_token_state_dict" in ckpt:
            agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])
        log.info("Actor-critic loaded")


def load_rlt_deploy_policy(config: DeployConfig) -> RLTDeployPolicy:
    """Factory function to create a ready-to-use RLTDeployPolicy."""
    return RLTDeployPolicy(config)
