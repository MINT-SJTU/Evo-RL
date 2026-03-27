from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.collector import Environment, rl_collect_step, warmup_collect
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.losses import actor_loss, critic_loss
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.utils import soft_update

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Accumulated metrics for a training run."""

    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    reconstruction_losses: list[float] = field(default_factory=list)
    env_steps: int = 0


def _cosine_lr(step: int, warmup: int, total: int, peak_lr: float, min_lr: float) -> float:
    """Cosine LR schedule with linear warmup."""
    import math
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def demo_adaptation(
    agent: RLTAgent,
    config: RLTConfig,
    demo_loader,
    demo_optimizer: torch.optim.Optimizer | None = None,
) -> list[float]:
    """Demo adaptation phase: L_ro + alpha * L_vla.

    After adaptation, freezes VLA and RL token encoder.
    Includes gradient clipping and cosine LR schedule with warmup.
    """
    if demo_optimizer is None:
        demo_params = list(agent.rl_token.parameters()) + list(agent.vla.parameters())
        demo_optimizer = torch.optim.Adam(demo_params, lr=config.demo_adaptation.lr)

    alpha = config.demo_adaptation.vla_ft_weight
    grad_clip = config.demo_adaptation.grad_clip_norm
    warmup = config.demo_adaptation.warmup_steps
    total_steps = config.demo_adaptation.steps
    peak_lr = config.demo_adaptation.lr
    min_lr = config.demo_adaptation.min_lr
    losses: list[float] = []
    agent.train()

    step = 0
    for obs, expert_actions in demo_loader:
        if step >= total_steps:
            break

        # Update learning rate
        lr = _cosine_lr(step, warmup, total_steps, peak_lr, min_lr)
        for pg in demo_optimizer.param_groups:
            pg["lr"] = lr

        vla_out = agent.vla.forward_vla(obs)
        l_ro = agent.rl_token.reconstruction_loss(vla_out.final_tokens)
        l_vla = agent.vla.supervised_loss(obs, expert_actions)
        loss = l_ro + alpha * l_vla

        demo_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        trainable_params = [p for p in agent.rl_token.parameters() if p.requires_grad]
        if alpha > 0:
            trainable_params += [p for p in agent.vla.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        demo_optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0:
            logger.info("Step %d/%d loss=%.4f lr=%.2e", step, total_steps, loss_val, lr)

        step += 1

    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    logger.info("Demo adaptation complete (%d steps). VLA and RL token encoder frozen.", step)

    return losses


def _do_critic_update(
    agent: RLTAgent,
    batch: dict[str, torch.Tensor],
    critic_optimizer: torch.optim.Optimizer,
    gamma: float,
    C: int,
) -> float:
    """Single critic gradient step."""
    loss = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, gamma, C)
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return loss.item()


def _do_actor_update(
    agent: RLTAgent,
    batch: dict[str, torch.Tensor],
    actor_optimizer: torch.optim.Optimizer,
    beta: float,
) -> float:
    """Single actor gradient step."""
    loss = actor_loss(agent.actor, agent.critic, batch, beta)
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()
    return loss.item()


def online_rl_loop(
    agent: RLTAgent,
    config: RLTConfig,
    env: Environment,
    replay_buffer: ReplayBuffer,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> TrainingMetrics:
    """Online RL training loop: warmup + collect + UTD updates."""
    if actor_optimizer is None:
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.actor.lr)
    if critic_optimizer is None:
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=config.critic.lr)

    metrics = TrainingMetrics()
    C = config.chunk_length
    gamma = config.training.gamma
    beta = config.training.beta
    tau = config.training.tau
    utd = config.training.utd_ratio
    actor_interval = config.training.actor_update_interval
    batch_size = config.training.batch_size

    # Warmup
    logger.info("Starting warmup for %d steps", config.collector.warmup_steps)
    warmup_steps = warmup_collect(env, agent, replay_buffer, config.collector.warmup_steps, C)
    metrics.env_steps += warmup_steps
    logger.info("Warmup done. Buffer size: %d", len(replay_buffer))

    # Online RL
    obs = env.reset()
    critic_update_count = 0
    total_steps = 0

    while total_steps < config.collector.total_env_steps:
        agent.eval()
        obs, done, steps = rl_collect_step(env, agent, obs, replay_buffer, C)
        total_steps += steps
        metrics.env_steps += steps

        if done:
            obs = env.reset()

        if len(replay_buffer) < batch_size:
            continue

        agent.train()
        for _ in range(utd):
            batch = replay_buffer.sample(batch_size)

            c_loss = _do_critic_update(agent, batch, critic_optimizer, gamma, C)
            metrics.critic_losses.append(c_loss)
            critic_update_count += 1

            if critic_update_count % actor_interval == 0:
                a_loss = _do_actor_update(agent, batch, actor_optimizer, beta)
                metrics.actor_losses.append(a_loss)

            soft_update(agent.target_critic, agent.critic, tau)

    logger.info(
        "Online RL done. %d env steps, %d critic updates, %d actor updates",
        metrics.env_steps, len(metrics.critic_losses), len(metrics.actor_losses),
    )
    return metrics
