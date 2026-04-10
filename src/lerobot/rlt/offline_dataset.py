from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
from lerobot.rlt.interfaces import ChunkTransition, Observation
from lerobot.rlt.replay_buffer import ReplayBuffer
from lerobot.rlt.utils import subsample_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Episode splitting
# ---------------------------------------------------------------------------

def split_episode_indices(
    num_episodes: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split episode indices into train/val/test sets.

    Episode-level splitting avoids data leakage between splits.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_episodes, generator=gen).tolist()
    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)
    return {
        "train": sorted(perm[:n_train]),
        "val": sorted(perm[n_train : n_train + n_val]),
        "test": sorted(perm[n_train + n_val :]),
    }


# ---------------------------------------------------------------------------
# 2. Core: build transitions from demo batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_transitions_from_demos(
    policy,
    demo_loader: DataLoader,
    chunk_length: int,
    reward_fn=None,
    device: str = "cpu",
    source: int = 0,
    episode_id: int = -1,
    is_critical: float = 0.0,
    stride: int = 1,
) -> list[ChunkTransition]:
    """Build ChunkTransitions from a *single-episode* sequential demo loader.

    Frames must arrive in temporal order (shuffle=False). The last frame
    produces a terminal transition (done=True, absorbing next_state).

    In offline RL, exec_chunk = expert_chunk. If reward_fn is None, rewards
    default to zeros.

    Args:
        policy: RLTPolicy (uses .encode_observation for state extraction).
        stride: Frame stride — controls bootstrap discount exponent (γ^stride).
    """
    policy.eval()
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for obs, expert_actions in demo_loader:
        obs = Observation(
            images={k: v.to(device) for k, v in obs.images.items()},
            proprio=obs.proprio.to(device),
        )
        state_vec, ref_chunk = policy.encode_observation(obs)
        B = state_vec.shape[0]
        for i in range(B):
            s = state_vec[i].cpu()
            r = ref_chunk[i].cpu()
            e = _subsample_chunk(expert_actions[i], chunk_length)
            rew = _compute_reward(reward_fn, e, r, chunk_length)
            encoded.append((s, r, e, rew))

    return _encoded_to_transitions(
        encoded,
        chunk_length,
        stride=stride,
        source=source,
        episode_id=episode_id,
        is_critical=is_critical,
    )


def _encoded_to_transitions(
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    chunk_length: int,
    stride: int = 1,
    source: int = 0,
    episode_id: int = -1,
    is_critical: float = 0.0,
) -> list[ChunkTransition]:
    """Convert list of (state, ref, expert, reward) into ChunkTransitions.

    Args:
        stride: Frame stride used during data loading. Controls the bootstrap
            discount exponent (γ^stride) — paper uses stride=2 so next_state
            is 2 control steps away, giving γ^2 ≈ 0.98 instead of γ^C ≈ 0.90.
    """
    transitions: list[ChunkTransition] = []
    for idx in range(len(encoded)):
        s, r, e, rew = encoded[idx]
        is_last = idx == len(encoded) - 1
        ns, nr = (s, r) if is_last else (encoded[idx + 1][0], encoded[idx + 1][1])
        transitions.append(ChunkTransition(
            state_vec=s, exec_chunk=e, ref_chunk=r, reward_seq=rew,
            next_state_vec=ns, next_ref_chunk=nr,
            done=torch.tensor(float(is_last)),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(stride),
            source=torch.tensor(source),
            episode_id=torch.tensor(episode_id),
            is_critical=torch.tensor(is_critical),
        ))
    return transitions


# ---------------------------------------------------------------------------
# 3. High-level: precompute buffer from dataset
# ---------------------------------------------------------------------------

def precompute_offline_buffer(
    policy,
    dataset_path: str,
    config,
    split: str = "train",
    device: str = "cpu",
    episode_ids: list[int] | None = None,
    reward_fn=None,
) -> ReplayBuffer:
    """Build a filled ReplayBuffer from demo data for offline RL.

    Iterates per-episode to correctly handle episode boundaries (done flags).

    Args:
        policy: RLTPolicy (frozen VLA + RL token encoder).
        dataset_path: Path to LeRobotDataset on disk.
        config: RLTConfig (needs vla_horizon, chunk_length, replay.capacity, seed).
        split: Which split to use ("train", "val", "test").
        device: Device for VLA forward passes.
        episode_ids: Override episode ids; if None, auto-split by config.seed.
        reward_fn: Optional callable(exec_chunk, ref_chunk) -> (C,) reward seq.
    """
    off_cfg = config.offline_rl
    dataset = RLTDemoDataset(
        dataset_path=dataset_path, chunk_length=config.vla_horizon,
        normalize_actions=True,
    )

    if episode_ids is None:
        num_eps = _count_episodes(dataset)
        splits = split_episode_indices(
            num_eps, train_ratio=off_cfg.train_ratio,
            val_ratio=off_cfg.val_ratio, seed=config.seed,
        )
        episode_ids = splits[split]

    buf = ReplayBuffer(capacity=config.replay.capacity)
    total_frames = 0

    for ep_idx, ep_id in enumerate(sorted(episode_ids)):
        frame_range = _episode_frame_range(dataset, ep_id)
        indices = list(range(frame_range[0], frame_range[1]))
        total_frames += len(indices)

        loader = DataLoader(
            Subset(dataset, indices), batch_size=32, shuffle=False,
            collate_fn=rlt_demo_collate, num_workers=0, drop_last=False,
        )
        transitions = build_transitions_from_demos(
            policy,
            loader,
            config.chunk_length,
            reward_fn=reward_fn,
            device=device,
            episode_id=ep_id,
        )
        for t in transitions:
            buf.add(t)

        if (ep_idx + 1) % 20 == 0:
            logger.info("Processed %d/%d episodes, %d transitions", ep_idx + 1, len(episode_ids), len(buf))

    logger.info(
        "Precomputed buffer: split=%s, episodes=%d, frames=%d, transitions=%d",
        split, len(episode_ids), total_frames, len(buf),
    )
    return buf


# ---------------------------------------------------------------------------
# 4. Cache: save / load precomputed transitions
# ---------------------------------------------------------------------------

def save_cached_buffer(
    transitions: list[ChunkTransition], cache_dir: str | Path, split: str,
) -> None:
    """Save precomputed transitions to disk as a .pt file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"transitions_{split}.pt"
    data = [
        {
            "state_vec": t.state_vec, "exec_chunk": t.exec_chunk,
            "ref_chunk": t.ref_chunk, "reward_seq": t.reward_seq,
            "next_state_vec": t.next_state_vec, "next_ref_chunk": t.next_ref_chunk,
            "done": t.done, "intervention": t.intervention,
            "actual_steps": t.actual_steps,
            "source": t.source,
            "episode_id": t.episode_id,
            "is_critical": t.is_critical,
        }
        for t in transitions
    ]
    torch.save(data, path)
    logger.info("Saved %d transitions to %s", len(data), path)


def load_cached_buffer(
    cache_dir: str | Path, split: str, capacity: int = 200_000,
) -> ReplayBuffer:
    """Load pre-computed transitions from cache and return a filled ReplayBuffer."""
    path = Path(cache_dir) / f"transitions_{split}.pt"
    data = torch.load(path, weights_only=False)
    buf = ReplayBuffer(capacity=capacity)
    for d in data:
        d["source"] = d.get("source", torch.tensor(0))
        d["episode_id"] = d.get("episode_id", torch.tensor(-1))
        d["is_critical"] = d.get("is_critical", torch.tensor(0.0))
        buf.add(ChunkTransition(**d))
    logger.info("Loaded %d transitions from %s", len(buf), path)
    return buf


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _subsample_chunk(actions: torch.Tensor, target_len: int) -> torch.Tensor:
    """Subsample action trajectory from (H, D) to (target_len, D)."""
    indices = subsample_indices(actions.shape[0], target_len)
    return actions[indices]


def _compute_reward(reward_fn, expert_chunk: torch.Tensor, ref_chunk: torch.Tensor, C: int) -> torch.Tensor:
    """Compute reward sequence, falling back to zeros if no reward_fn."""
    if reward_fn is not None:
        return reward_fn(expert_chunk, ref_chunk)
    return torch.zeros(C)


def _episode_frame_range(dataset: RLTDemoDataset, ep_id: int) -> tuple[int, int]:
    """Return (from_idx, to_idx) for a single episode from dataset metadata."""
    meta = dataset._dataset.meta
    return (meta.episodes["dataset_from_index"][ep_id], meta.episodes["dataset_to_index"][ep_id])


def _count_episodes(dataset: RLTDemoDataset) -> int:
    """Count unique episodes in the underlying LeRobotDataset."""
    meta = dataset._dataset.meta
    if hasattr(meta, "episodes") and meta.episodes is not None:
        return len(meta.episodes["episode_index"])
    return int(dataset._dataset[-1]["episode_index"].item()) + 1
