from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lerobot.rlt.interfaces import (
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
    ChunkTransition,
    Observation,
)
from lerobot.rlt.replay_buffer import ReplayBuffer


class Environment(ABC):
    """Abstract environment protocol for chunk-level control."""

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        ...

    @property
    @abstractmethod
    def proprio_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...


class DummyEnvironment(Environment):
    """Dummy environment for testing."""

    def __init__(self, proprio_dim: int = 14, action_dim: int = 14):
        self._proprio_dim = proprio_dim
        self._action_dim = action_dim

    def reset(self) -> Observation:
        return Observation(
            images={"base": torch.randn(1, 3, 64, 64)},
            proprio=torch.randn(1, self._proprio_dim),
        )

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        obs = Observation(
            images={"base": torch.randn(1, 3, 64, 64)},
            proprio=torch.randn(1, self._proprio_dim),
        )
        reward = float(torch.rand(1).item())
        done = False
        return obs, reward, done, {}

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


def execute_chunk(
    env: Environment,
    action_chunk: torch.Tensor,
    chunk_length: int,
) -> tuple[list[Observation], list[float], bool, dict]:
    """Execute a chunk of actions in the environment.

    Returns:
        obs_list, rewards, done, last_info
    """
    if action_chunk.dim() == 3:
        action_chunk = action_chunk.squeeze(0)

    obs_list: list[Observation] = []
    rewards: list[float] = []
    done = False
    info: dict = {}
    for t in range(chunk_length):
        obs, reward, done, info = env.step(action_chunk[t])
        obs_list.append(obs)
        rewards.append(reward)
        if done:
            break

    return obs_list, rewards, done, info


def pad_rewards(rewards: list[float], chunk_length: int) -> torch.Tensor:
    """Pad reward list to chunk_length, filling missing steps with 0."""
    padded = rewards + [0.0] * (chunk_length - len(rewards))
    return torch.tensor(padded[:chunk_length], dtype=torch.float32)


def _build_transition(
    state_vec: torch.Tensor,
    action_chunk: torch.Tensor,
    ref_chunk: torch.Tensor,
    reward_seq: torch.Tensor,
    next_state_vec: torch.Tensor,
    next_ref_chunk: torch.Tensor,
    done: bool,
    intervention: bool,
    actual_steps: int,
    source: int = 0,
    episode_id: int = -1,
    is_critical: float = 0.0,
) -> ChunkTransition:
    """Build a ChunkTransition, squeezing batch dims if present."""
    sq = lambda t: t.squeeze(0) if t.dim() > 1 and t.shape[0] == 1 else t
    return ChunkTransition(
        state_vec=sq(state_vec),
        exec_chunk=sq(action_chunk),
        ref_chunk=sq(ref_chunk),
        reward_seq=reward_seq,
        next_state_vec=sq(next_state_vec),
        next_ref_chunk=sq(next_ref_chunk),
        done=torch.tensor(float(done)),
        intervention=torch.tensor(float(intervention)),
        actual_steps=torch.tensor(actual_steps),
        source=torch.tensor(source),
        episode_id=torch.tensor(episode_id),
        is_critical=torch.tensor(is_critical),
    )


def warmup_collect(
    env: Environment,
    agent,
    replay_buffer: ReplayBuffer,
    num_steps: int,
    chunk_length: int,
    episode_id: int = -1,
) -> int:
    """Run VLA-only warmup, filling the replay buffer.

    Returns:
        total_steps: number of environment steps taken
    """
    total_steps = 0
    obs = env.reset()

    while total_steps < num_steps:
        with torch.no_grad():
            vla_out = agent.vla.forward_vla(obs)
            state_vec, ref_chunk = agent._extract_state_and_ref(obs, vla_out)
            action_chunk = ref_chunk  # warmup: use VLA reference as action

        obs_list, rewards, done, _ = execute_chunk(env, action_chunk, chunk_length)
        reward_seq = pad_rewards(rewards, chunk_length)
        total_steps += len(rewards)

        next_obs = obs_list[-1] if obs_list else obs
        with torch.no_grad():
            next_vla_out = agent.vla.forward_vla(next_obs)
            next_state_vec, next_ref_chunk = agent._extract_state_and_ref(next_obs, next_vla_out)

        transition = _build_transition(
            state_vec, action_chunk, ref_chunk, reward_seq,
            next_state_vec, next_ref_chunk, done, False, len(rewards),
            source=TRANSITION_SOURCE_WARMUP_VLA, episode_id=episode_id,
        )
        replay_buffer.add(transition)

        obs = env.reset() if done else next_obs

    return total_steps


def rl_collect_step(
    env: Environment,
    agent,
    obs: Observation,
    replay_buffer: ReplayBuffer,
    chunk_length: int,
    intervention: bool = False,
    episode_id: int = -1,
) -> tuple[Observation, bool, int]:
    """Execute one RL collection step with single VLA forward.

    Returns:
        next_obs, done, steps_taken
    """
    with torch.no_grad():
        action_chunk, _, state_vec, ref_chunk = agent.select_action(obs)

    obs_list, rewards, done, _ = execute_chunk(env, action_chunk, chunk_length)
    reward_seq = pad_rewards(rewards, chunk_length)

    next_obs = obs_list[-1] if obs_list else obs
    with torch.no_grad():
        next_vla_out = agent.vla.forward_vla(next_obs)
        next_state_vec, next_ref_chunk = agent._extract_state_and_ref(next_obs, next_vla_out)

    transition = _build_transition(
        state_vec, action_chunk, ref_chunk, reward_seq,
        next_state_vec, next_ref_chunk, done, intervention, len(rewards),
        source=TRANSITION_SOURCE_RL_AUTONOMOUS, episode_id=episode_id,
    )
    replay_buffer.add(transition)

    return next_obs, done, len(rewards)
