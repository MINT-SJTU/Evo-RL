# RLT Training Scripts

This directory contains training-time entry points only.

## Naming

- `demo dataset`: raw LeRobot demonstrations on disk.
- `transition cache`: serialized chunk transitions built from a demo dataset.
- `replay buffer`: in-memory buffer loaded from a transition cache or built directly from a demo dataset.

## Core entry points

- `build_transition_cache.py`: build chunk-transition cache files from a demo dataset.
- `train_rl_token.py`: train the RL token module.
- `train_chunk_actor_critic.py`: train the chunk-level actor-critic from a transition cache or directly from a demo dataset.

## Experiment entry points

- `search_rl_token.py`: RL token architecture search.
- `search_chunk_actor_critic_phase1.py`: actor-critic phase-1 search.
- `search_chunk_actor_critic_phase2.py`: actor-critic phase-2 search.
- `search_chunk_actor_critic_phase3.py`: actor-critic phase-3 search.
- `search_chunk_actor_critic_cp.py`: actor-critic search on critical-phase transition caches.

## Shared module

- `common.py`: shared script bootstrap and Pi0.5 / config helpers.
