from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.utils import compute_discount_vector


def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted return over a chunk of rewards.

    Args:
        reward_seq: (B, C) rewards for each timestep (padded with 0 beyond actual_steps)
        gamma: discount factor
        actual_steps: (B,) number of valid steps per chunk (if None, assume all C are valid)

    Returns:
        (B, 1) discounted return
    """
    C = reward_seq.shape[1]
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
) -> torch.Tensor:
    """TD3-style chunk-level TD loss with correct truncated-chunk handling.

    Uses actual_steps to compute the correct bootstrap exponent gamma^k
    instead of always using gamma^C.
    """
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style)
        mu_next, _ = actor.forward(x_next, ref_next)
        q_next = target_critic.min_q(x_next, mu_next)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Q-maximization + BC regularization toward VLA reference.

    Uses deterministic mean (not noisy samples) for stable optimization.
    """
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    q = critic.min_q(x, mu)
    bc_reg = F.mse_loss(mu, ref)
    return -q.mean() + beta * bc_reg
