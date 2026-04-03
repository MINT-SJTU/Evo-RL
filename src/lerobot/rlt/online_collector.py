from __future__ import annotations

import logging

import torch

from lerobot.rlt.interfaces import ChunkTransition
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.utils.recording_annotations import SOURCE_HUMAN

logger = logging.getLogger(__name__)


class RLTOnlineCollector:
    """Accumulates per-frame robot data and emits ChunkTransitions every C frames."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_length: int,
        action_dim: int,
    ):
        self._buffer = replay_buffer
        self._C = chunk_length
        self._action_dim = action_dim
        self._frame_actions: list[torch.Tensor] = []
        self._frame_sources: list[float] = []
        self._chunk_state: torch.Tensor | None = None
        self._chunk_ref: torch.Tensor | None = None
        self._chunk_is_critical: float = 0.0
        self._episode_id: int = -1
        self._prev_transition: ChunkTransition | None = None

    def start_episode(self, episode_id: int) -> None:
        self._episode_id = episode_id
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._prev_transition = None

    def on_frame(
        self,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
    ) -> ChunkTransition | None:
        if len(self._frame_actions) == 0:
            # Capture state at chunk start. During intervention state_vec may be None;
            # fall back to the previous transition's state so human chunks are not dropped.
            if state_vec is not None:
                self._chunk_state = state_vec
                self._chunk_ref = ref_chunk
            elif self._prev_transition is not None:
                self._chunk_state = self._prev_transition.next_state_vec
                self._chunk_ref = self._prev_transition.next_ref_chunk
            self._chunk_is_critical = is_critical

        self._frame_actions.append(action.detach().cpu())
        self._frame_sources.append(source_type)

        if len(self._frame_actions) >= self._C:
            return self._emit_transition(done=False)
        return None

    def flush_episode(self, episode_success: bool) -> ChunkTransition | None:
        if self._frame_actions:
            return self._emit_transition(done=True)
        # Episode length was exact multiple of C — mark last emitted chunk as terminal
        if self._prev_transition is not None:
            self._prev_transition.done = torch.tensor(1.0)
        return None

    def _emit_transition(self, done: bool) -> ChunkTransition | None:
        if self._chunk_state is None or self._chunk_ref is None:
            self._frame_actions.clear()
            self._frame_sources.clear()
            self._chunk_is_critical = 0.0
            return None

        actual = len(self._frame_actions)
        exec_list = self._frame_actions[: self._C]
        exec_chunk = torch.stack(exec_list)
        if actual < self._C:
            pad = torch.zeros(self._C - actual, self._action_dim, dtype=exec_chunk.dtype)
            exec_chunk = torch.cat([exec_chunk, pad])

        # Deterministic tie-break: human wins ties (highest priority)
        dominant_source = max(set(self._frame_sources), key=lambda s: (self._frame_sources.count(s), s))
        ref = self._chunk_ref.cpu()
        if dominant_source == SOURCE_HUMAN:
            ref = exec_chunk.clone()

        state = self._chunk_state.cpu()
        transition = ChunkTransition(
            state_vec=state,
            exec_chunk=exec_chunk,
            ref_chunk=ref,
            reward_seq=torch.zeros(self._C),
            next_state_vec=state,
            next_ref_chunk=ref,
            done=torch.tensor(float(done)),
            intervention=torch.tensor(float(dominant_source == SOURCE_HUMAN)),
            actual_steps=torch.tensor(actual),
            source=torch.tensor(int(dominant_source)),
            episode_id=torch.tensor(self._episode_id),
            is_critical=torch.tensor(self._chunk_is_critical),
        )

        if self._prev_transition is not None:
            self._prev_transition.next_state_vec = state.clone()
            self._prev_transition.next_ref_chunk = ref.clone()

        self._buffer.add(transition)
        self._prev_transition = transition

        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        return transition
