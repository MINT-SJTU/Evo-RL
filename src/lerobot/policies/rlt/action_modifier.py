from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.phase_controller import PhaseController
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import flatten_chunk, subsample_indices, unflatten_chunk


class PrefixOutputCapture:
    """Capture prefix hidden states from PI05Policy's PaliGemmaWithExpertModel.

    PI05's ``sample_actions`` calls ``paligemma_with_expert.forward()``
    directly (not via ``__call__``), so standard PyTorch forward hooks
    never fire.  Instead we monkey-patch ``forward`` to intercept the
    prefix-only call (``inputs_embeds=[prefix_embs, None]``).

    After capture the raw prefix tokens are pooled from (B, ~968, 2048) to
    (B, token_pool_size, 2048) via adaptive average pooling.
    """

    def __init__(self, token_pool_size: int = 64):
        self.token_pool_size = token_pool_size
        self._captured: Tensor | None = None
        self._original_forward = None
        self._target = None

    def attach(self, policy) -> None:
        """Monkey-patch ``forward`` on ``policy.model.paligemma_with_expert``."""
        target = policy.model.paligemma_with_expert
        self._target = target
        self._original_forward = target.forward

        capture = self  # closure reference

        def patched_forward(*args, **kwargs):
            result = capture._original_forward(*args, **kwargs)
            outputs, _past_kv = result
            prefix_tokens = outputs[0]
            if prefix_tokens is not None:
                capture._captured = capture._pool(prefix_tokens.detach().float())
            return result

        target.forward = patched_forward

    def _pool(self, tensor: Tensor) -> Tensor:
        # adaptive_avg_pool1d expects (B, C, L); tensor is (B, L, D)
        x = tensor.transpose(1, 2)  # (B, D, L)
        x = F.adaptive_avg_pool1d(x, self.token_pool_size)  # (B, D, pool)
        return x.transpose(1, 2)  # (B, pool, D)

    def consume(self) -> Tensor:
        """Return and clear the captured prefix tokens.

        Raises AssertionError if no prefix output has been captured yet (i.e.
        the VLA forward pass has not run since the last consume).
        """
        assert self._captured is not None, (
            "No prefix_output captured -- VLA forward not yet called"
        )
        result = self._captured
        self._captured = None
        return result

    def detach(self) -> None:
        """Restore the original forward method."""
        if self._original_forward is not None and self._target is not None:
            self._target.forward = self._original_forward
            self._original_forward = None
            self._target = None


@dataclass
class RLTStepMetadata:
    """Per-step metadata emitted alongside each popped action."""

    phase: float  # 0.0 = VLA, 1.0 = critical/RL
    source_type: float  # 0.0 = VLA action, 1.0 = RL action


class RLTActionModifier(nn.Module):
    """RL Token Encoder + Actor + Phase Controller + Chunk Queue.

    Sits between the VLA action output and the final action used by the robot.
    In VLA phase the VLA chunk is passed through unchanged; in RL phase the
    Actor refines the chunk conditioned on the RL-token state representation.
    """

    def __init__(
        self,
        rl_token: RLTokenModule,
        actor: ChunkActor,
        phase_ctrl: PhaseController,
        chunk_length: int,
        action_dim: int,
        proprio_dim: int,
    ):
        super().__init__()
        self.rl_token = rl_token
        self.actor = actor
        self.phase_ctrl = phase_ctrl
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self._action_queue: deque[Tensor] = deque()
        self._step_metadata: deque[RLTStepMetadata] = deque()
        self._subsample_cache: Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_rl_phase(self) -> bool:
        return self.phase_ctrl.is_critical

    @property
    def needs_new_chunk(self) -> bool:
        return len(self._action_queue) == 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _get_subsample_indices(self, H: int, device: torch.device | None = None) -> Tensor:
        C = self.chunk_length
        if self._subsample_cache is None or self._subsample_cache.shape[0] != C:
            self._subsample_cache = subsample_indices(H, C)
        if device is not None and self._subsample_cache.device != device:
            self._subsample_cache = self._subsample_cache.to(device)
        return self._subsample_cache

    @torch.no_grad()
    def compute_chunk(
        self,
        vla_chunk: Tensor,
        proprio: Tensor,
        prefix_tokens: Tensor,
    ) -> Tensor:
        """Compute action chunk, either VLA pass-through or RL-refined.

        Args:
            vla_chunk: (B, H, action_dim) normalised VLA action chunk.
            proprio: (B, proprio_dim) normalised proprioceptive state.
            prefix_tokens: (B, pool_size, token_dim) pooled VLA prefix tokens.

        Returns:
            chunk: (B, chunk_length, action_dim) in [-1, 1].
        """
        indices = self._get_subsample_indices(vla_chunk.shape[1], vla_chunk.device)
        ref_chunk = vla_chunk[:, indices, :]

        phase_val = 1.0 if self.is_rl_phase else 0.0
        source_val = phase_val

        if not self.is_rl_phase:
            self._enqueue_metadata(phase_val, source_val)
            return ref_chunk

        z_rl = self.rl_token.encode(prefix_tokens)
        state_vec = torch.cat([z_rl, proprio], dim=-1)
        ref_flat = flatten_chunk(ref_chunk)
        mu, _ = self.actor(state_vec, ref_flat, training=False)
        chunk = unflatten_chunk(mu, self.chunk_length).clamp(-1, 1)

        self._enqueue_metadata(phase_val, source_val)
        return chunk

    def _enqueue_metadata(self, phase: float, source: float) -> None:
        """Enqueue metadata entries for every step in the upcoming chunk."""
        for _ in range(self.chunk_length):
            self._step_metadata.append(RLTStepMetadata(phase=phase, source_type=source))

    def enqueue(self, chunk: Tensor) -> None:
        """Enqueue chunk steps into the action queue.

        Args:
            chunk: (B, C, action_dim).
        """
        self._action_queue.extend(chunk.transpose(0, 1))

    def pop_action(self) -> Tensor:
        """Pop and return the next single-step action from the queue."""
        return self._action_queue.popleft()

    def pop_step_metadata(self) -> RLTStepMetadata | None:
        """Pop and return the next step's metadata, or None if empty."""
        if len(self._step_metadata) == 0:
            return None
        return self._step_metadata.popleft()

    # ------------------------------------------------------------------
    # Phase control (duck-typed interface for recording_loop)
    # ------------------------------------------------------------------

    def set_rl_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_critical()

    def set_vla_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_vla()

    def trigger_critical_phase(self) -> None:
        """Toggle between VLA and critical phase, clearing queues."""
        self.interrupt_chunk()
        if self.phase_ctrl.is_critical:
            self.phase_ctrl.trigger_vla()
        else:
            self.phase_ctrl.trigger_critical()

    def interrupt_chunk(self) -> None:
        """Clear both action and metadata queues (e.g. on phase switch)."""
        self._action_queue.clear()
        self._step_metadata.clear()

    def reset(self) -> None:
        """Reset queues and phase controller to initial state."""
        self._action_queue.clear()
        self._step_metadata.clear()
        self.phase_ctrl.reset()
