#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(description="Precompute offline RL cache from demo data.")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--cache-dir", required=True, help="Directory to save cached transitions.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Demo adaptation checkpoint (.pt).")
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-mode", default="hybrid", choices=("terminal", "action_matching", "hybrid"))
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
    from lerobot.rlt.offline_dataset import (
        build_transitions_from_demos,
        save_cached_buffer,
        split_episode_indices,
        _count_episodes,
        _episode_frame_range,
    )
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.rewards import build_reward_seq

    # --- Config ---
    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]

    # --- Load VLA ---
    logger.info("Loading pi0.5 from %s", args.model_path)
    vla = Pi05VLAAdapter(
        model_path=args.model_path,
        actual_action_dim=12,
        actual_proprio_dim=12,
        task_instruction=args.task_instruction,
        dtype="bfloat16",
        device=args.device,
        token_pool_size=args.token_pool_size,
    )
    logger.info("pi0.5 loaded")

    # --- Build agent and load checkpoint ---
    agent = RLTAgent(config, vla).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])
    logger.info("Loaded demo adaptation checkpoint from %s", args.checkpoint)

    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    agent.eval()

    # --- Reward function ---
    reward_mode = args.reward_mode
    success_bonus = args.success_bonus
    progress_scale = args.progress_scale

    def reward_fn(expert_chunk: torch.Tensor, ref_chunk: torch.Tensor) -> torch.Tensor:
        return build_reward_seq(
            expert_chunk=expert_chunk,
            exec_chunk=expert_chunk,  # offline: exec = expert
            mode=reward_mode,
            episode_success=True,
            is_terminal_chunk=False,
            success_bonus=success_bonus,
            progress_scale=progress_scale,
        )

    # --- Split episodes ---
    dataset = RLTDemoDataset(dataset_path=args.dataset_path, chunk_length=config.vla_horizon)
    num_episodes = _count_episodes(dataset)
    splits = split_episode_indices(
        num_episodes, train_ratio=config.offline_rl.train_ratio,
        val_ratio=config.offline_rl.val_ratio, seed=config.seed,
    )
    logger.info("Episodes: %d total -> train=%d, val=%d, test=%d",
                num_episodes, len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # --- Process each split ---
    t_start = time.time()

    for split_name, episode_ids in splits.items():
        if not episode_ids:
            logger.info("Split %s: 0 episodes, skipping", split_name)
            continue

        split_start = time.time()
        all_transitions = []

        for ep_idx, ep_id in enumerate(sorted(episode_ids)):
            frame_range = _episode_frame_range(dataset, ep_id)
            indices = list(range(frame_range[0], frame_range[1], args.frame_stride))

            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=rlt_demo_collate,
                num_workers=0,
                drop_last=False,
            )
            transitions = build_transitions_from_demos(
                agent, loader, config.chunk_length, reward_fn=reward_fn, device=args.device,
            )
            # Add terminal bonus to last transition of each episode
            if transitions and success_bonus > 0:
                last_t = transitions[-1]
                last_t.reward_seq[-1] = last_t.reward_seq[-1] + success_bonus
            all_transitions.extend(transitions)

            if (ep_idx + 1) % 20 == 0:
                logger.info("  [%s] %d/%d episodes, %d transitions so far",
                            split_name, ep_idx + 1, len(episode_ids), len(all_transitions))

        save_cached_buffer(all_transitions, args.cache_dir, split_name)
        split_elapsed = time.time() - split_start
        logger.info("Split %s: %d transitions in %.1fs", split_name, len(all_transitions), split_elapsed)

    total_elapsed = time.time() - t_start
    logger.info("All splits cached to %s in %.1fs", args.cache_dir, total_elapsed)


if __name__ == "__main__":
    main()
