from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
from lerobot.rlt.interfaces import ChunkTransition, Observation
from lerobot.rlt.replay_buffer import ReplayBuffer

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
    frame_indices: list[int],
    episode_last_frame: int,
    chunk_length: int,
    device: str = "cpu",
    source: int = 0,
    episode_id: int = -1,
    is_critical: float = 0.0,
    stride: int = 1,
    episode_success: bool = True,
    success_bonus: float = 1.0,
) -> list[ChunkTransition]:
    """Build ChunkTransitions from a *single-episode* sequential demo loader.

    Frames must arrive in temporal order (shuffle=False). Each transition
    follows the paper's chunk-level semantics:

      state = x_t
      action = a_t:t+C-1
      reward_seq = r_t:t+C-1
      next_state = x_{t+C}
      bootstrap exponent = C

    Args:
        policy: RLTPolicy (uses .encode_observation for state extraction).
        frame_indices: Raw frame indices for the sampled anchor states.
        episode_last_frame: Raw index of the last frame in the episode.
        stride: Anchor spacing in control steps. Must divide chunk_length.
    """
    policy.eval()
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

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
            encoded.append((s, r, e))

    if len(encoded) != len(frame_indices):
        raise ValueError(
            f"Encoded {len(encoded)} states but received {len(frame_indices)} frame indices"
        )

    return _encoded_to_transitions(
        encoded,
        frame_indices,
        episode_last_frame,
        chunk_length,
        stride=stride,
        episode_success=episode_success,
        success_bonus=success_bonus,
        source=source,
        episode_id=episode_id,
        is_critical=is_critical,
    )


def _encoded_to_transitions(
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    frame_indices: list[int],
    episode_last_frame: int,
    chunk_length: int,
    stride: int = 1,
    episode_success: bool = True,
    success_bonus: float = 1.0,
    source: int = 0,
    episode_id: int = -1,
    is_critical: float = 0.0,
) -> list[ChunkTransition]:
    """Convert list of sampled anchors into chunk-level ChunkTransitions.

    Args:
        stride: Anchor spacing used for overlapping chunk windows. The critic
            still bootstraps from x_{t+C}, so chunk_length must be divisible
            by stride.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if chunk_length % stride != 0:
        raise ValueError(
            f"chunk_length={chunk_length} must be divisible by stride={stride}"
        )

    frame_to_encoded_idx = {frame_idx: idx for idx, frame_idx in enumerate(frame_indices)}
    transitions: list[ChunkTransition] = []
    for idx, start_frame in enumerate(frame_indices):
        next_frame = start_frame + chunk_length
        if next_frame > episode_last_frame:
            continue

        next_idx = frame_to_encoded_idx.get(next_frame)
        if next_idx is None:
            raise ValueError(
                f"Missing next_state anchor for frame {start_frame} -> {next_frame}; "
                f"sampled anchors must include every t+C state"
            )

        s, r, e = encoded[idx]
        ns, nr = encoded[next_idx][0], encoded[next_idx][1]
        is_terminal = next_frame == episode_last_frame
        rew = _terminal_reward_seq(
            chunk_length=chunk_length,
            is_terminal=is_terminal,
            episode_success=episode_success,
            success_bonus=success_bonus,
        )
        transitions.append(ChunkTransition(
            state_vec=s, exec_chunk=e, ref_chunk=r, reward_seq=rew,
            next_state_vec=ns, next_ref_chunk=nr,
            done=torch.tensor(float(is_terminal)),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(chunk_length),
            source=torch.tensor(source),
            episode_id=torch.tensor(episode_id),
            is_critical=torch.tensor(is_critical),
        ))
    if episode_success and transitions and not any(t.done.item() == 1.0 for t in transitions):
        raise ValueError(
            "Successful episode does not contain a terminal chunk under the current "
            f"stride={stride} and chunk_length={chunk_length}"
        )
    return transitions


def build_overlap_frame_indices(
    episode_start: int,
    episode_stop: int,
    chunk_length: int,
    stride: int,
) -> list[int]:
    """Return sampled frames needed for overlapping chunk transitions.

    The returned indices include:
    - start anchors sampled with the paper's stride-based overlap pattern
    - the final valid terminal anchor ``episode_last_frame - C`` when needed
    - any bootstrap state ``x_{t+C}`` required by those anchors

    This guarantees every stored chunk transition has its matching ``next_state``
    under the paper's ``x_t -> x_{t+C}`` semantics.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if episode_stop <= episode_start:
        return []

    episode_last_frame = episode_stop - 1
    indices = set(range(episode_start, episode_stop, stride))
    terminal_anchor = episode_last_frame - chunk_length
    if terminal_anchor < episode_start:
        return sorted(indices)

    indices.add(terminal_anchor)
    start_anchors = sorted(idx for idx in indices if idx + chunk_length <= episode_last_frame)
    for start_frame in start_anchors:
        indices.add(start_frame + chunk_length)
    return sorted(indices)


# ---------------------------------------------------------------------------
# 3. High-level: precompute buffer from dataset
# ---------------------------------------------------------------------------

def build_transition_replay_buffer(
    policy,
    demo_dataset_path: str,
    config,
    split: str = "train",
    device: str = "cpu",
    episode_ids: list[int] | None = None,
) -> ReplayBuffer:
    """Build a replay buffer from a raw demo dataset.

    Iterates per-episode to correctly handle episode boundaries (done flags).

    Args:
        policy: RLTPolicy (frozen VLA + RL token encoder).
        demo_dataset_path: Path to the raw LeRobot demo dataset on disk.
        config: RLTConfig (needs vla_horizon, chunk_length, replay.capacity, seed).
        split: Which split to use ("train", "val", "test").
        device: Device for VLA forward passes.
        episode_ids: Override episode ids; if None, auto-split by config.seed.
    """
    off_cfg = config.offline_rl
    dataset = RLTDemoDataset(
        dataset_path=demo_dataset_path, chunk_length=config.vla_horizon,
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
        indices = build_overlap_frame_indices(
            episode_start=frame_range[0],
            episode_stop=frame_range[1],
            chunk_length=config.chunk_length,
            stride=off_cfg.frame_stride,
        )
        total_frames += len(indices)

        loader = DataLoader(
            Subset(dataset, indices), batch_size=32, shuffle=False,
            collate_fn=rlt_demo_collate, num_workers=0, drop_last=False,
        )
        transitions = build_transitions_from_demos(
            policy,
            loader,
            frame_indices=indices,
            episode_last_frame=frame_range[1] - 1,
            chunk_length=config.chunk_length,
            device=device,
            episode_id=ep_id,
            stride=off_cfg.frame_stride,
            success_bonus=off_cfg.success_bonus,
        )
        for t in transitions:
            buf.add(t)

        if (ep_idx + 1) % 20 == 0:
            logger.info("Processed %d/%d episodes, %d transitions", ep_idx + 1, len(episode_ids), len(buf))

    logger.info(
        "Built replay buffer: split=%s, episodes=%d, frames=%d, transitions=%d",
        split, len(episode_ids), total_frames, len(buf),
    )
    return buf


# ---------------------------------------------------------------------------
# 4. Cache: save / load precomputed transitions
# ---------------------------------------------------------------------------

def save_transition_cache(
    transitions: list[ChunkTransition], transition_cache_dir: str | Path, split: str,
) -> None:
    """Save chunk transitions to disk."""
    transition_cache_dir = Path(transition_cache_dir)
    transition_cache_dir.mkdir(parents=True, exist_ok=True)
    path = transition_cache_dir / f"chunk_transitions_{split}.pt"
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


def load_transition_cache(
    transition_cache_dir: str | Path, split: str, capacity: int = 200_000,
) -> ReplayBuffer:
    """Load a chunk-transition cache into a replay buffer."""
    path = Path(transition_cache_dir) / f"chunk_transitions_{split}.pt"
    data = torch.load(path, weights_only=False)
    buf = ReplayBuffer(capacity=capacity)
    for d in data:
        buf.add(ChunkTransition(**d))
    logger.info("Loaded %d transitions from %s", len(buf), path)
    return buf


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _subsample_chunk(actions: torch.Tensor, target_len: int) -> torch.Tensor:
    """Take first target_len frames from action trajectory (H, D) -> (target_len, D)."""
    return actions[:target_len]


def _terminal_reward_seq(
    chunk_length: int,
    is_terminal: bool,
    episode_success: bool,
    success_bonus: float,
) -> torch.Tensor:
    """Return step-level sparse terminal reward aggregated over a chunk."""
    reward = torch.zeros(chunk_length)
    if is_terminal and episode_success:
        reward[-1] = success_bonus
    return reward


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
