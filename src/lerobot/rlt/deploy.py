from __future__ import annotations

import logging
import time
from collections import deque

import torch
import torch.nn as nn

from lerobot.rlt.config import ActorConfig, CriticConfig, RLTConfig, RLTokenConfig
from lerobot.rlt.deploy_config import DeployConfig
from lerobot.rlt.interfaces import Observation
from lerobot.rlt.obs_bridge import robot_obs_to_rlt_obs
from lerobot.rlt.phase_controller import Phase, PhaseController
from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.utils import filter_encoder_only, subsample_indices
from lerobot.utils.recording_annotations import PHASE_CRITICAL, PHASE_PREFIX, SOURCE_RL, SOURCE_VLA

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
        self._meta_queue: deque[dict] = deque()

        log.info("Loading VLA model from %s", config.vla_model_path)
        self.vla = Pi05VLAAdapter(
            model_path=config.vla_model_path,
            actual_action_dim=len(config.action_keys),
            actual_proprio_dim=len(config.proprio_keys),
            task_instruction=config.task_instruction,
            device=config.device,
            token_pool_size=config.token_pool_size,
            tokenizer_path=config.tokenizer_path,
        )

        rlt_config = _build_rlt_config(config, self.vla.action_dim)
        self.policy = RLTPolicy(rlt_config, self.vla)
        self.policy.to(config.device)

        _load_checkpoints(self.policy, config)

        self.policy.freeze_vla()
        self.policy.freeze_rl_token_encoder()
        self.policy.eval()

        self._phase_ctrl = PhaseController(mode="manual")
        self.phase_controller = self._phase_ctrl
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

        action_chunk, state_vec, ref_chunk = self._compute_action_chunk(obs)
        phase = self._phase_ctrl.phase
        phase_val = PHASE_CRITICAL if phase == Phase.CRITICAL_PHASE else PHASE_PREFIX
        source_val = SOURCE_RL if phase == Phase.CRITICAL_PHASE else SOURCE_VLA
        state_cpu = state_vec.squeeze(0).detach().cpu()
        ref_cpu = ref_chunk.squeeze(0).detach().cpu()

        # Enqueue each timestep as (1, action_dim)
        for t in range(action_chunk.shape[1]):
            self._action_queue.append(action_chunk[:, t, :])
            self._meta_queue.append(
                {
                    "phase": phase_val,
                    "source_type": source_val,
                    "state_vec": state_cpu,
                    "ref_chunk": ref_cpu,
                }
            )

        elapsed = time.monotonic() - t0
        self._timing["last_chunk_compute_ms"] = elapsed * 1000
        log.debug("Chunk compute: %.1f ms (%d actions queued)", elapsed * 1000, len(self._action_queue))

    def _compute_action_chunk(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action chunk based on phase mode.

        Returns:
            action_chunk: (1, C, action_dim)
            state_vec: (1, state_dim)
            ref_chunk: (1, C, action_dim)
        """
        vla_out = self.vla.forward_vla(obs)
        state_vec, ref_chunk = self.policy._extract_state_and_ref(obs, vla_out)

        if self._phase_mode == "always_vla":
            self._phase_ctrl.trigger_vla()
            return self._vla_reference_chunk(vla_out), state_vec, ref_chunk

        if self._phase_mode == "always_rl":
            self._phase_ctrl.trigger_critical()
            return self._rl_action_chunk(obs, vla_out), state_vec, ref_chunk

        # manual mode: use phase controller
        self._phase_ctrl.update(z_rl=state_vec[:, :self.vla.token_dim])
        if self._phase_ctrl.phase == Phase.CRITICAL_PHASE:
            return self._rl_action_chunk(obs, vla_out), state_vec, ref_chunk
        return self._vla_reference_chunk(vla_out), state_vec, ref_chunk

    def _rl_action_chunk(self, obs: Observation, vla_out) -> torch.Tensor:
        """Run RL actor to get action chunk."""
        action_chunk, _, _, _ = self.policy.select_action(
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
        self._meta_queue.clear()
        self._phase_ctrl.reset()
        log.info("Episode reset")

    def pop_step_metadata(self) -> dict | None:
        return self._meta_queue.popleft() if self._meta_queue else None

    def trigger_critical_phase(self) -> None:
        """Toggle between VLA (prefix) and RL (critical) phases.

        Clears queued actions so the new phase takes effect immediately
        on the next select_action() call.
        """
        self._action_queue.clear()
        self._meta_queue.clear()
        if self._phase_ctrl.is_critical:
            self._phase_ctrl.trigger_vla()
            log.info("Phase toggled back to VLA (prefix)")
        else:
            self._phase_ctrl.trigger_critical()
            log.info("Phase toggled to CRITICAL")

    @property
    def timing(self) -> dict[str, float]:
        return dict(self._timing)


def _build_rlt_config(config: DeployConfig, action_dim: int) -> RLTConfig:
    """Build an RLTConfig from DeployConfig to construct the policy."""
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


def _load_checkpoints(policy: RLTPolicy, config: DeployConfig) -> None:
    """Load RL token encoder and actor checkpoints (no decoder/critic for VRAM savings)."""
    device = config.device

    if config.rl_token_checkpoint:
        log.info("Loading RL token checkpoint: %s", config.rl_token_checkpoint)
        ckpt = torch.load(config.rl_token_checkpoint, map_location="cpu", weights_only=True)
        filtered, skipped = filter_encoder_only(ckpt["rl_token_state_dict"])
        unexpected = policy.rl_token.load_state_dict(filtered, strict=False)
        assert not unexpected.unexpected_keys, f"Unexpected keys: {unexpected.unexpected_keys}"
        log.info(
            "RL token encoder loaded (trained %d steps, skipped %d decoder keys)",
            ckpt.get("step", -1), len(skipped),
        )
        del ckpt
        policy.rl_token.to(device)

    if config.ac_checkpoint:
        log.info("Loading actor-critic checkpoint: %s", config.ac_checkpoint)
        ckpt = torch.load(config.ac_checkpoint, map_location="cpu", weights_only=True)
        policy.actor.load_state_dict(ckpt["actor_state_dict"])
        if "rl_token_state_dict" in ckpt:
            filtered, skipped = filter_encoder_only(ckpt["rl_token_state_dict"])
            unexpected = policy.rl_token.load_state_dict(filtered, strict=False)
            assert not unexpected.unexpected_keys, f"Unexpected keys: {unexpected.unexpected_keys}"
            log.info("RL token encoder overwritten from AC checkpoint (skipped %d decoder keys)", len(skipped))
        log.info("Actor loaded (critic skipped for inference)")
        del ckpt
        policy.actor.to(device)


def load_rlt_deploy_policy(config: DeployConfig) -> RLTDeployPolicy:
    """Factory function to create a ready-to-use RLTDeployPolicy."""
    return RLTDeployPolicy(config)
