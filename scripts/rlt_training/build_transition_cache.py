#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging, load_training_config

logger = configure_logging(__name__)


class PlaceholderEncoder:
    """Observation encoder stub used before RL token training is ready."""

    def __init__(self, token_dim: int, action_dim: int, chunk_length: int):
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length

    def eval(self):
        return self

    def encode_observation(self, obs):
        batch_size = obs.proprio.shape[0]
        device = obs.proprio.device
        z_rl = torch.zeros(batch_size, self.token_dim, device=device)
        state_vec = torch.cat([z_rl, obs.proprio], dim=-1)
        ref_chunk = torch.zeros(batch_size, self.chunk_length, self.action_dim, device=device)
        return state_vec, ref_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chunk-transition cache from a raw demo dataset.")
    parser.add_argument("--demo-dataset-path", required=True, help="Path to the raw LeRobot demo dataset.")
    parser.add_argument("--transition-cache-dir", required=True, help="Output directory for chunk-transition cache files.")
    parser.add_argument("--model-path", default="lerobot/pi05_base", help="Pi0.5 model path.")
    parser.add_argument("--rl-token-checkpoint", default=None, help="RL token checkpoint used for observation encoding.")
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--placeholder", action="store_true", help="Build transitions with zero RL-token features.")
    parser.add_argument("--mark-critical", action="store_true", help="Mark every transition as critical-phase data.")
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    return parser.parse_args()


def build_encoder(args: argparse.Namespace, config):
    if args.placeholder:
        logger.info("Using placeholder encoder")
        return PlaceholderEncoder(
            token_dim=config.rl_token.token_dim,
            action_dim=config.action_dim,
            chunk_length=config.chunk_length,
        ), "cpu"

    if args.rl_token_checkpoint is None:
        raise ValueError("--rl-token-checkpoint is required unless --placeholder is set")

    from common import build_pi05_policy

    logger.info("Loading pi0.5 from %s", args.model_path)
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        rl_token_checkpoint=args.rl_token_checkpoint,
    )
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()
    return policy, args.device


def main() -> None:
    args = parse_args()

    from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
    from lerobot.rlt.offline_dataset import (
        build_overlap_frame_indices,
        build_transitions_from_demos,
        save_transition_cache,
        split_episode_indices,
        _count_episodes,
        _episode_frame_range,
    )

    config = load_training_config(args.config)
    config.offline_rl.frame_stride = args.frame_stride
    config.offline_rl.success_bonus = args.success_bonus
    if args.train_ratio is not None:
        config.offline_rl.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        config.offline_rl.val_ratio = args.val_ratio
    encoder, device = build_encoder(args, config)

    dataset = RLTDemoDataset(
        dataset_path=args.demo_dataset_path,
        chunk_length=config.vla_horizon,
        normalize_actions=True,
    )
    num_episodes = _count_episodes(dataset)
    splits = split_episode_indices(
        num_episodes,
        train_ratio=config.offline_rl.train_ratio,
        val_ratio=config.offline_rl.val_ratio,
        seed=config.seed,
    )
    logger.info(
        "Demo episodes: %d total -> train=%d val=%d test=%d",
        num_episodes,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    total_start = time.time()
    for split_name, episode_ids in splits.items():
        if not episode_ids:
            logger.info("Split %s: 0 episodes, skipping", split_name)
            continue

        split_start = time.time()
        transitions = []
        for episode_index, episode_id in enumerate(sorted(episode_ids), start=1):
            frame_start, frame_stop = _episode_frame_range(dataset, episode_id)
            frame_indices = build_overlap_frame_indices(
                episode_start=frame_start,
                episode_stop=frame_stop,
                chunk_length=config.chunk_length,
                stride=config.offline_rl.frame_stride,
            )
            loader = DataLoader(
                Subset(dataset, frame_indices),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=rlt_demo_collate,
                num_workers=0,
                drop_last=False,
            )
            episode_transitions = build_transitions_from_demos(
                policy=encoder,
                demo_loader=loader,
                frame_indices=frame_indices,
                episode_last_frame=frame_stop - 1,
                chunk_length=config.chunk_length,
                device=device,
                episode_id=episode_id,
                is_critical=float(args.mark_critical),
                stride=config.offline_rl.frame_stride,
                success_bonus=config.offline_rl.success_bonus,
            )
            transitions.extend(episode_transitions)
            if episode_index % 20 == 0:
                logger.info(
                    "[%s] %d/%d episodes, %d transitions",
                    split_name,
                    episode_index,
                    len(episode_ids),
                    len(transitions),
                )

        save_transition_cache(transitions, args.transition_cache_dir, split_name)
        logger.info(
            "Split %s: %d transitions in %.1fs",
            split_name,
            len(transitions),
            time.time() - split_start,
        )

    logger.info("Transition cache saved to %s in %.1fs", args.transition_cache_dir, time.time() - total_start)


if __name__ == "__main__":
    main()
