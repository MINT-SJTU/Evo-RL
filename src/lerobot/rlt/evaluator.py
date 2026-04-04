from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lerobot.rlt.algorithm import RLTAlgorithm
from lerobot.rlt.collector import Environment, execute_chunk
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.interfaces import EXEC_CHUNK_FLAT, REF_CHUNK_FLAT, STATE_VEC
from lerobot.rlt.losses import critic_loss
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.utils import flatten_chunk


@dataclass
class EvalMetrics:
    """Evaluation metrics aggregated over episodes."""

    success_rate: float = 0.0
    mean_episode_length: float = 0.0
    throughput: float = 0.0  # successes per 10 minutes
    mean_q: float = 0.0
    mean_ref_deviation: float = 0.0
    episode_returns: list[float] = field(default_factory=list)


def evaluate(
    algorithm: RLTAlgorithm,
    config: RLTConfig,
    env: Environment,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    success_fn: Callable[[dict], bool] | None = None,
) -> EvalMetrics:
    """Run evaluation episodes and compute metrics.

    Args:
        success_fn: optional function(info_dict) -> bool. Receives the last
                    step's info dict from the environment.
    """
    algorithm.eval()
    policy = algorithm.policy
    metrics = EvalMetrics()
    C = config.chunk_length

    successes = 0
    total_lengths = 0
    q_values: list[float] = []
    ref_deviations: list[float] = []
    start_time = time.time()

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        episode_steps = 0
        last_info: dict = {}
        done = False

        while episode_steps < max_steps_per_episode:
            with torch.no_grad():
                action_chunk, _, state_vec, ref_chunk = policy.select_action(obs)
                action_flat = flatten_chunk(action_chunk)
                ref_flat = flatten_chunk(ref_chunk)

                q_val = algorithm.critic.min_q(state_vec, action_flat)
                q_values.append(q_val.mean().item())

                dev = ((action_flat - ref_flat) ** 2).mean().item()
                ref_deviations.append(dev)

            obs_list, rewards, done, last_info = execute_chunk(env, action_chunk, C)
            episode_return += sum(rewards)
            episode_steps += len(rewards)

            if done:
                break
            if obs_list:
                obs = obs_list[-1]

        metrics.episode_returns.append(episode_return)
        total_lengths += episode_steps

        episode_success = success_fn(last_info) if success_fn is not None else done
        if episode_success:
            successes += 1

    elapsed = time.time() - start_time

    metrics.success_rate = successes / max(num_episodes, 1)
    metrics.mean_episode_length = total_lengths / max(num_episodes, 1)
    metrics.throughput = successes / max(elapsed / 600.0, 1e-6)
    metrics.mean_q = sum(q_values) / max(len(q_values), 1)
    metrics.mean_ref_deviation = sum(ref_deviations) / max(len(ref_deviations), 1)

    return metrics


@dataclass
class OfflineEvalMetrics:
    """Evaluation metrics for offline RL (no environment rollouts)."""

    expert_action_mse: float = 0.0  # ||actor(state, ref) - expert||^2
    ref_action_mse: float = 0.0  # ||actor(state, ref) - ref||^2
    mean_q_policy: float = 0.0  # Q(state, actor_action)
    mean_q_expert: float = 0.0  # Q(state, expert_action)
    q_gap: float = 0.0  # mean_q_policy - mean_q_expert (should be <= 0)
    mean_critic_td_error: float = 0.0


def evaluate_offline(
    algorithm: RLTAlgorithm,
    val_buffer: ReplayBuffer,
    config: RLTConfig,
    num_batches: int = 10,
) -> OfflineEvalMetrics:
    """Evaluate algorithm on a held-out replay buffer without environment rollouts.

    Computes action quality metrics (MSE vs expert/reference), Q-value
    diagnostics, and critic TD error on validation data.
    """
    algorithm.eval()
    policy = algorithm.policy

    total_expert_mse = 0.0
    total_ref_mse = 0.0
    total_q_policy = 0.0
    total_q_expert = 0.0
    total_td_error = 0.0

    device = next(algorithm.parameters()).device

    for _ in range(num_batches):
        batch = val_buffer.sample(config.training.batch_size)
        state_vec = batch[STATE_VEC].to(device)
        exec_chunk_flat = batch[EXEC_CHUNK_FLAT].to(device)
        ref_chunk_flat = batch[REF_CHUNK_FLAT].to(device)

        with torch.no_grad():
            mu, _ = policy.actor.forward(state_vec, ref_chunk_flat)

            total_expert_mse += F.mse_loss(mu, exec_chunk_flat).item()
            total_ref_mse += F.mse_loss(mu, ref_chunk_flat).item()

            total_q_policy += algorithm.critic.min_q(state_vec, mu).mean().item()
            total_q_expert += algorithm.critic.min_q(state_vec, exec_chunk_flat).mean().item()

            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            td_err = critic_loss(
                algorithm.critic, algorithm.target_critic, policy.actor,
                batch_on_device, config.training.gamma, config.chunk_length,
            )
            total_td_error += td_err.item()

    n = max(num_batches, 1)
    return OfflineEvalMetrics(
        expert_action_mse=total_expert_mse / n,
        ref_action_mse=total_ref_mse / n,
        mean_q_policy=total_q_policy / n,
        mean_q_expert=total_q_expert / n,
        q_gap=(total_q_policy - total_q_expert) / n,
        mean_critic_td_error=total_td_error / n,
    )
