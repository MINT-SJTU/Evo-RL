from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rlt.action_modifier import (
    PrefixOutputCapture,
    RLTActionModifier,
    RLTStepMetadata,
)
from lerobot.policies.rlt.configuration_rlt import RLTPretrainedConfig
from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.phase_controller import PhaseController
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import filter_encoder_only

log = logging.getLogger(__name__)

# Prefix used for PI05 sub-module in state_dict
_PI05_PREFIX = "_pi05."


def _load_pi05_config(config_cls, pretrained_path: str):
    """Load PI05Config from a directory, stripping the ``type`` field.

    draccus ChoiceRegistry uses ``type`` for polymorphic dispatch but
    concrete sub-classes (PI05Config) don't declare it — so parsing a
    config.json that contains ``"type": "pi05"`` fails.  We strip it
    before handing the file to draccus.
    """
    import json
    import tempfile

    import draccus

    config_path = Path(pretrained_path) / "config.json"
    with open(config_path) as fh:
        raw = json.load(fh)
    raw.pop("type", None)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(raw, tmp)
        tmp_path = tmp.name

    try:
        with draccus.config_type("json"):
            return draccus.parse(config_cls, tmp_path, args=[])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _tie_embed_tokens(pi05) -> None:
    """Copy lm_head weight into embed_tokens when the latter is missing.

    SFT checkpoints only store ``lm_head.weight``; the language model's
    ``embed_tokens.weight`` is a tied copy.  After ``strict=False``
    loading, embed_tokens is left at random init — fix it here.
    """
    lm = pi05.model.paligemma_with_expert.paligemma
    embed = lm.model.language_model.embed_tokens
    if embed is not None and lm.lm_head.weight.data_ptr() != embed.weight.data_ptr():
        embed.weight = lm.lm_head.weight


class RLTPretrainedPolicy(PreTrainedPolicy):
    """LeRobot-compatible policy wrapping the RLT RL head on top of a frozen VLA.

    Internally holds a standard ``PI05Policy`` for VLA inference and uses a
    monkey-patched ``forward`` to capture prefix hidden states.  The RL Token
    encoder and Actor operate in normalised space, producing action chunks that
    are then post-processed by the standard lerobot postprocessor.

    The safetensors checkpoint contains ONLY the RL head weights (rl_token
    encoder + actor, ~400 MB).  The VLA backbone (~7 GB) is loaded lazily from
    ``config.vla_pretrained_path``.
    """

    config_class = RLTPretrainedConfig
    name = "rlt"

    def __init__(self, config: RLTPretrainedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: RLTPretrainedConfig = config

        # Build RL Token encoder (inference-only: no decoder)
        rl_token = RLTokenModule(
            token_dim=config.rl_token_dim,
            nhead=config.rl_token_nhead,
            num_enc_layers=config.rl_token_enc_layers,
            num_dec_layers=config.rl_token_dec_layers,
            ff_dim=config.rl_token_ff_dim,
            num_rl_tokens=config.rl_token_num_rl_tokens,
            inference_only=True,
        )

        # Build Actor (ResidualMLP by default)
        chunk_dim = config.chunk_length * config.action_dim
        state_dim = config.rl_token_dim + config.proprio_dim

        actor = ChunkActor(
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

        # Phase controller
        phase_ctrl = self._build_phase_controller(config)

        # Action modifier (owns rl_token + actor + phase_ctrl + queues)
        self.modifier = RLTActionModifier(
            rl_token=rl_token,
            actor=actor,
            phase_ctrl=phase_ctrl,
            chunk_length=config.chunk_length,
            action_dim=config.action_dim,
            proprio_dim=config.proprio_dim,
        )

        # Prefix output capture (hook attached lazily when PI05 loads)
        self.prefix_capture = PrefixOutputCapture(
            token_pool_size=config.token_pool_size,
        )

        # Inner PI05Policy -- deferred to avoid 7 GB memory at init time
        self._pi05 = None

        # Load RL Token + Actor weights from checkpoint files if paths given
        self._load_rlt_checkpoints()

    # ------------------------------------------------------------------
    # Convenience accessors (sub-modules registered via modifier)
    # ------------------------------------------------------------------

    @property
    def rl_token(self) -> RLTokenModule:
        return self.modifier.rl_token

    @property
    def actor(self) -> ChunkActor:
        return self.modifier.actor

    # ------------------------------------------------------------------
    # Phase controller construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_phase_controller(config: RLTPretrainedConfig) -> PhaseController:
        ctrl = PhaseController(mode="manual")
        if config.phase_mode == "always_rl":
            ctrl.trigger_critical()
        elif config.phase_mode == "always_vla":
            ctrl.trigger_vla()
        # "manual" starts in VLA phase by default
        return ctrl

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_rlt_checkpoints(self) -> None:
        """Load RL Token encoder and Actor weights from .pt checkpoint files."""
        cfg = self.config

        if cfg.rl_token_ckpt_path:
            self._load_rl_token_ckpt(cfg.rl_token_ckpt_path)

        if cfg.ac_ckpt_path:
            self._load_ac_ckpt(cfg.ac_ckpt_path)

    def _load_rl_token_ckpt(self, path: str) -> None:
        log.info("Loading RL Token checkpoint from %s", path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        raw_sd = ckpt.get("rl_token_state_dict", ckpt)
        filtered, skipped = filter_encoder_only(raw_sd)
        if skipped:
            log.info("Stripped %d decoder keys from RL Token checkpoint", len(skipped))
        self.rl_token.load_state_dict(filtered, strict=False)

    def _load_ac_ckpt(self, path: str) -> None:
        log.info("Loading Actor checkpoint from %s", path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        actor_sd = ckpt.get("actor_state_dict", ckpt)
        self.actor.load_state_dict(actor_sd)

    # ------------------------------------------------------------------
    # PI05 lazy loading
    # ------------------------------------------------------------------

    def _ensure_pi05(self):
        """Lazily load the PI05Policy backbone on first inference call."""
        if self._pi05 is not None:
            return self._pi05

        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        cfg = self.config
        log.info("Loading PI05 backbone from %s", cfg.vla_pretrained_path)

        pi05_config = _load_pi05_config(PI05Config, cfg.vla_pretrained_path)
        if cfg.task_instruction:
            pi05_config.task_instruction = cfg.task_instruction

        pi05 = PI05Policy.from_pretrained(
            cfg.vla_pretrained_path,
            config=pi05_config,
            revision=cfg.vla_revision,
            strict=False,
        )
        # SFT checkpoints store lm_head but not embed_tokens (tied weight).
        _tie_embed_tokens(pi05)
        pi05.to(cfg.device or "cpu")
        pi05.eval()
        for p in pi05.parameters():
            p.requires_grad = False

        # Ensure RL head (rl_token + actor) is on the same device as PI05
        self.modifier.to(cfg.device or "cpu")

        # Attach prefix output hook
        self.prefix_capture.attach(pi05)

        self._pi05 = pi05
        return self._pi05

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Predict a full chunk of actions.

        Returns:
            chunk: (B, chunk_length, action_dim)
        """
        self.eval()
        pi05 = self._ensure_pi05()

        # VLA forward -- hook captures prefix_output as a side effect
        vla_chunk = pi05.predict_action_chunk(batch, **kwargs)
        # PI05 already unpads to its output_features action_dim; truncate
        # further if RLT action_dim is smaller (e.g. 12 vs PI05's 14)
        vla_chunk = vla_chunk[:, :, : self.config.action_dim]

        # Retrieve hook-captured prefix tokens
        prefix_tokens = self.prefix_capture.consume()

        # Normalised proprio from preprocessed batch
        proprio = batch["observation.state"][:, : self.config.proprio_dim]

        # Compute chunk (VLA pass-through or RL Actor)
        return self.modifier.compute_chunk(vla_chunk, proprio, prefix_tokens)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Return a single-step action, using internal chunk queue.

        When the modifier's action queue is empty a full VLA forward is run,
        prefix tokens are captured via hook, and the modifier computes a new
        chunk (VLA pass-through or Actor-refined depending on phase).

        Returns:
            action: (B, action_dim)
        """
        if self.modifier.needs_new_chunk:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self.modifier.enqueue(chunk)

        return self.modifier.pop_action()

    # ------------------------------------------------------------------
    # Phase control delegation (duck-typed for recording_loop)
    # ------------------------------------------------------------------

    def set_rl_mode(self) -> None:
        self.modifier.set_rl_mode()

    def set_vla_mode(self) -> None:
        self.modifier.set_vla_mode()

    def trigger_critical_phase(self) -> None:
        self.modifier.trigger_critical_phase()

    def interrupt_chunk(self) -> None:
        self.modifier.interrupt_chunk()

    def pop_step_metadata(self) -> RLTStepMetadata | None:
        return self.modifier.pop_step_metadata()

    def reset(self) -> None:
        self.modifier.reset()

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
    # State dict: exclude PI05 weights from serialization
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not k.startswith(_PI05_PREFIX)}

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

        Uses strict=False by default because the PI05 sub-module is not in the
        safetensors file.
        """
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            strict=False,
            **kwargs,
        )
