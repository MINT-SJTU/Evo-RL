from __future__ import annotations

import torch

REWARD_MODES = ("terminal", "action_matching", "hybrid")


def _resolve_actual_steps(chunk_length: int, actual_steps: int | torch.Tensor | None) -> int:
    """Resolve actual_steps to an int, clamped to [0, chunk_length]."""
    if actual_steps is None:
        return chunk_length
    if isinstance(actual_steps, torch.Tensor):
        actual_steps = int(actual_steps.item())
    return max(0, min(chunk_length, int(actual_steps)))


def _terminal_reward(
    chunk_size: int,
    episode_success: bool,
    is_terminal_chunk: bool,
    actual_steps: int,
    success_bonus: float,
    device: torch.device,
) -> torch.Tensor:
    """(C,) reward with optional terminal bonus at the last valid step."""
    reward = torch.zeros(chunk_size, device=device)
    if is_terminal_chunk and episode_success and actual_steps > 0:
        reward[actual_steps - 1] = success_bonus
    return reward


def _action_matching_reward(
    expert_chunk: torch.Tensor,
    exec_chunk: torch.Tensor,
    actual_steps: int,
    progress_scale: float,
) -> torch.Tensor:
    """(C,) dense reward = -||exec - expert||^2, zero-padded beyond actual_steps."""
    C = expert_chunk.shape[0]
    sq_err = (exec_chunk - expert_chunk).pow(2).sum(dim=-1)  # (C,)
    reward = -sq_err * progress_scale
    if actual_steps < C:
        reward[actual_steps:] = 0.0
    return reward


def build_reward_seq(
    expert_chunk: torch.Tensor,
    exec_chunk: torch.Tensor,
    mode: str = "hybrid",
    episode_success: bool = True,
    is_terminal_chunk: bool = False,
    actual_steps: int | torch.Tensor | None = None,
    success_bonus: float = 10.0,
    progress_scale: float = 1.0,
) -> torch.Tensor:
    """Build a (C,) reward sequence for a single chunk transition.

    Args:
        expert_chunk: (C, action_dim) expert actions from demo.
        exec_chunk: (C, action_dim) executed actions.
        mode: One of "terminal", "action_matching", "hybrid".
        episode_success: Whether the episode succeeded.
        is_terminal_chunk: Whether this is the last chunk in the episode.
        actual_steps: Number of valid steps in chunk (rest are padding).
            Accepts int, scalar Tensor, or None (meaning all C steps valid).
        success_bonus: Bonus value for terminal success.
        progress_scale: Scaling factor for action-matching reward.

    Returns:
        (C,) reward tensor on the same device as expert_chunk.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode {mode!r}, expected one of {REWARD_MODES}")
    if expert_chunk.shape != exec_chunk.shape:
        raise ValueError(
            f"Shape mismatch: expert_chunk {expert_chunk.shape} vs exec_chunk {exec_chunk.shape}"
        )

    C = expert_chunk.shape[0]
    steps = _resolve_actual_steps(C, actual_steps)

    if mode == "terminal":
        return _terminal_reward(C, episode_success, is_terminal_chunk, steps, success_bonus, expert_chunk.device)

    matching = _action_matching_reward(expert_chunk, exec_chunk, steps, progress_scale)

    if mode == "action_matching":
        return matching

    # hybrid: action_matching + terminal bonus
    terminal = _terminal_reward(C, episode_success, is_terminal_chunk, steps, success_bonus, expert_chunk.device)
    return matching + terminal
