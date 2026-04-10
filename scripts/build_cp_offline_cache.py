#!/usr/bin/env python
"""Build offline RL cache from critical-phase (CP) success datasets.

Two modes:
  1. --placeholder  : No VLA / RL token needed. Uses zero z_rl placeholder.
                      Good for validating pipeline and data structure.
  2. Default        : Requires --model-path and --checkpoint (VLA + RL token).
                      Produces production-ready cache identical to build_rlt_offline_cache.py.

The CP success dataset is a standard LeRobotDataset where each episode is one
successful critical-phase segment. All transitions get is_critical=1.0 and
episode_success=True.

Usage:
    # Placeholder mode (RL token not ready yet):
    PYTHONPATH=src python scripts/build_cp_offline_cache.py \
        --dataset-path /path/to/eval_271ep_sft_cp_success_TIMESTAMP \
        --cache-dir outputs/cp_cache_placeholder \
        --placeholder

    # Full mode (VLA + RL token available):
    PYTHONPATH=src python scripts/build_cp_offline_cache.py \
        --dataset-path /path/to/eval_271ep_sft_cp_success_TIMESTAMP \
        --cache-dir outputs/cp_cache \
        --model-path lerobot/pi05_base \
        --checkpoint outputs/rlt_demo_adapt_v8/demo_adapt_checkpoint.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline RL cache from CP success data.")
    parser.add_argument("--dataset-path", required=True, help="Path to CP success LeRobotDataset.")
    parser.add_argument("--cache-dir", required=True, help="Output directory for cached transitions.")
    parser.add_argument("--placeholder", action="store_true",
                        help="Use zero z_rl placeholder (no VLA/RL token needed).")
    parser.add_argument("--model-path", default="lerobot/pi05_base", help="VLA model path.")
    parser.add_argument("--checkpoint", default=None, help="RL token checkpoint (.pt).")
    parser.add_argument("--config", default=None, help="Path to RLT YAML config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-mode", default="hybrid", choices=("terminal", "action_matching", "hybrid"))
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--proprio-dim", type=int, default=12)
    parser.add_argument("--token-dim", type=int, default=2048)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--vla-horizon", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Placeholder encoder: no VLA needed, uses zeros for z_rl
# ---------------------------------------------------------------------------

class PlaceholderEncoder:
    """Mimics RLTPolicy.encode_observation without needing VLA or RL token."""

    def __init__(self, token_dim: int, proprio_dim: int, chunk_length: int, vla_horizon: int):
        self.token_dim = token_dim
        self.proprio_dim = proprio_dim
        self.chunk_length = chunk_length
        self._indices = _subsample_indices(vla_horizon, chunk_length)

    def eval(self):
        return self

    def encode_observation(self, obs):
        """Return (state_vec, ref_chunk) with zero z_rl placeholder."""
        B = obs.proprio.shape[0]
        device = obs.proprio.device
        z_rl = torch.zeros(B, self.token_dim, device=device)
        state_vec = torch.cat([z_rl, obs.proprio], dim=-1)
        # ref_chunk is not available without VLA — use zeros as placeholder
        ref_chunk = torch.zeros(B, self.chunk_length, obs.proprio.shape[-1], device=device)
        return state_vec, ref_chunk


def _subsample_indices(source_len: int, target_len: int) -> torch.Tensor:
    if source_len == target_len:
        return torch.arange(target_len)
    return torch.linspace(0, source_len - 1, target_len).long()


# ---------------------------------------------------------------------------
# Core transition builder for CP data
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_cp_transitions(
    encoder,
    demo_loader: DataLoader,
    chunk_length: int,
    reward_mode: str,
    success_bonus: float,
    progress_scale: float,
    device: str,
    episode_id: int,
    is_placeholder: bool,
) -> list:
    """Build ChunkTransitions from a single-episode CP demo loader.

    All CP transitions have is_critical=1.0 and episode_success=True.
    """
    from lerobot.rlt.interfaces import ChunkTransition, Observation, TRANSITION_SOURCE_DEMO
    from lerobot.rlt.rewards import build_reward_seq
    from lerobot.rlt.utils import subsample_indices

    encoder.eval()
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for obs, expert_actions in demo_loader:
        obs = Observation(
            images={k: v.to(device) for k, v in obs.images.items()},
            proprio=obs.proprio.to(device),
        )

        if is_placeholder:
            state_vec, _ = encoder.encode_observation(obs)
            # In placeholder mode, subsample expert actions as both ref and exec
            B = state_vec.shape[0]
            indices = subsample_indices(expert_actions.shape[1], chunk_length)
            ref_chunk = expert_actions[:, indices, :].to(device)
        else:
            state_vec, ref_chunk = encoder.encode_observation(obs)

        B = state_vec.shape[0]
        for i in range(B):
            s = state_vec[i].cpu()
            r = ref_chunk[i].cpu()
            # In offline from demos: exec_chunk = expert_chunk
            indices = subsample_indices(expert_actions.shape[1], chunk_length)
            e = expert_actions[i, indices, :].cpu()
            # Reward: not terminal for now (terminal bonus added separately below)
            rew = build_reward_seq(
                expert_chunk=e, exec_chunk=e, mode=reward_mode,
                episode_success=True, is_terminal_chunk=False,
                success_bonus=success_bonus, progress_scale=progress_scale,
            )
            encoded.append((s, r, e, rew))

    # Build transitions with proper next-state linkage
    transitions: list[ChunkTransition] = []
    for idx in range(len(encoded)):
        s, r, e, rew = encoded[idx]
        is_last = idx == len(encoded) - 1
        ns, nr = (s, r) if is_last else (encoded[idx + 1][0], encoded[idx + 1][1])

        # Apply terminal success bonus on last chunk
        if is_last and success_bonus > 0:
            from lerobot.rlt.rewards import _terminal_reward
            terminal_rew = _terminal_reward(
                chunk_length, True, True, chunk_length, success_bonus, rew.device,
            )
            rew = rew + terminal_rew

        transitions.append(ChunkTransition(
            state_vec=s,
            exec_chunk=e,
            ref_chunk=r,
            reward_seq=rew,
            next_state_vec=ns,
            next_ref_chunk=nr,
            done=torch.tensor(float(is_last)),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(chunk_length),
            source=torch.tensor(TRANSITION_SOURCE_DEMO),
            episode_id=torch.tensor(episode_id),
            is_critical=torch.tensor(1.0),
        ))

    return transitions


def main() -> None:
    args = parse_args()

    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
    from lerobot.rlt.offline_dataset import (
        save_cached_buffer,
        split_episode_indices,
        _count_episodes,
        _episode_frame_range,
    )

    if not args.placeholder and args.checkpoint is None:
        logger.error("Either --placeholder or --checkpoint is required.")
        sys.exit(1)

    # --- Config ---
    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = args.action_dim
    config.proprio_dim = args.proprio_dim
    config.vla_horizon = args.vla_horizon
    config.chunk_length = args.chunk_length
    config.offline_rl.train_ratio = args.train_ratio
    config.offline_rl.val_ratio = args.val_ratio

    # --- Build encoder ---
    if args.placeholder:
        logger.info("Using PLACEHOLDER mode (zero z_rl, no VLA/RL token)")
        encoder = PlaceholderEncoder(
            token_dim=args.token_dim,
            proprio_dim=args.proprio_dim,
            chunk_length=args.chunk_length,
            vla_horizon=args.vla_horizon,
        )
        device = "cpu"
    else:
        from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
        from lerobot.rlt.policy import RLTPolicy

        logger.info("Loading pi0.5 from %s", args.model_path)
        vla = Pi05VLAAdapter(
            model_path=args.model_path,
            actual_action_dim=args.action_dim,
            actual_proprio_dim=args.proprio_dim,
            task_instruction=args.task_instruction,
            dtype="bfloat16",
            device=args.device,
            token_pool_size=args.token_pool_size,
        )
        encoder = RLTPolicy(config, vla).to(args.device)
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        encoder.rl_token.load_state_dict(ckpt["rl_token_state_dict"], strict=False)
        logger.info("Loaded RL token checkpoint from %s", args.checkpoint)
        encoder.freeze_vla()
        encoder.freeze_rl_token_encoder()
        device = args.device

    # --- Load dataset ---
    dataset = RLTDemoDataset(
        dataset_path=args.dataset_path, chunk_length=config.vla_horizon,
        normalize_actions=True,
    )
    num_episodes = _count_episodes(dataset)
    logger.info("CP success dataset: %d episodes, %d frames", num_episodes, len(dataset))

    # --- Split episodes ---
    splits = split_episode_indices(
        num_episodes, train_ratio=config.offline_rl.train_ratio,
        val_ratio=config.offline_rl.val_ratio, seed=config.seed,
    )
    logger.info("Split: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # --- Process each split ---
    t_start = time.time()
    stats = {}

    for split_name, episode_ids in splits.items():
        if not episode_ids:
            logger.info("Split %s: 0 episodes, skipping", split_name)
            continue

        split_start = time.time()
        all_transitions = []

        for ep_idx, ep_id in enumerate(sorted(episode_ids)):
            frame_range = _episode_frame_range(dataset, ep_id)
            indices = list(range(frame_range[0], frame_range[1]))

            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=rlt_demo_collate,
                num_workers=0,
                drop_last=False,
            )
            transitions = build_cp_transitions(
                encoder, loader, config.chunk_length,
                reward_mode=args.reward_mode,
                success_bonus=args.success_bonus,
                progress_scale=args.progress_scale,
                device=device,
                episode_id=ep_id,
                is_placeholder=args.placeholder,
            )
            all_transitions.extend(transitions)

            if (ep_idx + 1) % 10 == 0:
                logger.info("  [%s] %d/%d episodes, %d transitions",
                            split_name, ep_idx + 1, len(episode_ids), len(all_transitions))

        save_cached_buffer(all_transitions, args.cache_dir, split_name)
        elapsed = time.time() - split_start
        stats[split_name] = (len(episode_ids), len(all_transitions), elapsed)
        logger.info("Split %s: %d episodes -> %d transitions in %.1fs",
                     split_name, len(episode_ids), len(all_transitions), elapsed)

    total_elapsed = time.time() - t_start

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Cache saved to: %s", args.cache_dir)
    logger.info("Mode: %s", "PLACEHOLDER (z_rl=0)" if args.placeholder else "FULL (VLA+RL token)")
    for split_name, (n_ep, n_trans, elapsed) in stats.items():
        logger.info("  %s: %d episodes, %d transitions (%.1fs)", split_name, n_ep, n_trans, elapsed)
    logger.info("Total: %.1fs", total_elapsed)
    if args.placeholder:
        logger.info("NOTE: z_rl is zeros. Re-run without --placeholder when RL token is ready.")
        logger.info("  Or use build_rlt_offline_cache.py directly on this CP dataset.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
