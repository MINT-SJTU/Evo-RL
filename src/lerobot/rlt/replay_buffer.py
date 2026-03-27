from __future__ import annotations

import random
from collections import deque

import torch

from lerobot.rlt.interfaces import (
    ACTUAL_STEPS,
    DONE,
    EXEC_CHUNK_FLAT,
    NEXT_REF_FLAT,
    NEXT_STATE_VEC,
    REF_CHUNK_FLAT,
    REWARD_SEQ,
    STATE_VEC,
    ChunkTransition,
)


class ReplayBuffer:
    """Deque-based chunk-level replay buffer.

    Stores single (unbatched) ChunkTransition objects and collates them
    into batched dicts at sample time.

    TODO: Replace with tensor-backed circular buffer for performance at scale.
    """

    def __init__(self, capacity: int = 200_000):
        self.buffer: deque[ChunkTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def capacity(self) -> int:
        if self.buffer.maxlen is None:
            raise RuntimeError("Buffer has no capacity limit")
        return self.buffer.maxlen

    def add(self, transition: ChunkTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch and collate into a dict of stacked tensors."""
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        batch = [self.buffer[i] for i in indices]
        stacked_exec = torch.stack([t.exec_chunk for t in batch])
        stacked_ref = torch.stack([t.ref_chunk for t in batch])
        stacked_next_ref = torch.stack([t.next_ref_chunk for t in batch])
        return {
            STATE_VEC: torch.stack([t.state_vec for t in batch]),
            EXEC_CHUNK_FLAT: stacked_exec.flatten(start_dim=-2),
            REF_CHUNK_FLAT: stacked_ref.flatten(start_dim=-2),
            REWARD_SEQ: torch.stack([t.reward_seq for t in batch]),
            NEXT_STATE_VEC: torch.stack([t.next_state_vec for t in batch]),
            NEXT_REF_FLAT: stacked_next_ref.flatten(start_dim=-2),
            DONE: torch.stack([t.done for t in batch]),
            ACTUAL_STEPS: torch.stack([t.actual_steps for t in batch]),
        }
