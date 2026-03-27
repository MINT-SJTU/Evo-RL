from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.rlt.agent import RLTAgent
from lerobot.rlt.collector import Environment, execute_chunk
from lerobot.rlt.config import RLTConfig
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
    agent: RLTAgent,
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
    agent.eval()
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
                action_chunk, _, state_vec, ref_chunk = agent.select_action(obs)
                action_flat = flatten_chunk(action_chunk)
                ref_flat = flatten_chunk(ref_chunk)

                q_val = agent.critic.min_q(state_vec, action_flat)
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
