from __future__ import annotations

from dataclasses import dataclass, field

import torch


# Batch dictionary key constants to avoid typos
STATE_VEC = "state_vec"
EXEC_CHUNK_FLAT = "exec_chunk_flat"
REF_CHUNK_FLAT = "ref_chunk_flat"
REWARD_SEQ = "reward_seq"
NEXT_STATE_VEC = "next_state_vec"
NEXT_REF_FLAT = "next_ref_flat"
DONE = "done"
ACTUAL_STEPS = "actual_steps"


@dataclass
class Observation:
    """Observation from the environment."""

    images: dict[str, torch.Tensor]  # camera_name -> (B, C, H, W)
    proprio: torch.Tensor  # (B, proprio_dim)
    instruction_ids: torch.Tensor | None = None
    timestamp: float | None = None


@dataclass
class VLAOutput:
    """Output from a single VLA forward pass."""

    final_tokens: torch.Tensor  # (B, M, token_dim)
    sampled_action_chunk: torch.Tensor  # (B, H, action_dim)
    extra: dict = field(default_factory=dict)


@dataclass
class ChunkTransition:
    """Single (unbatched) chunk-level transition for replay."""

    state_vec: torch.Tensor  # (state_dim,)
    exec_chunk: torch.Tensor  # (C, action_dim)
    ref_chunk: torch.Tensor  # (C, action_dim)
    reward_seq: torch.Tensor  # (C,)
    next_state_vec: torch.Tensor  # (state_dim,)
    next_ref_chunk: torch.Tensor  # (C, action_dim)
    done: torch.Tensor  # scalar
    intervention: torch.Tensor  # scalar, 0/1 flag
    actual_steps: torch.Tensor  # scalar int, steps actually executed (<= C)
