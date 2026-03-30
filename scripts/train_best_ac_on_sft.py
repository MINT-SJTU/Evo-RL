#!/usr/bin/env python
"""Train best actor-critic architecture on SFT model RL tokens.

Steps:
1. Load SFT pi0.5 + SFT RL token checkpoint
2. Build offline cache (if not already cached)
3. Train ResidualMLP actor-critic with beta=0.3, 50K steps
4. Evaluate and save checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/home/coder/share/models/pi05_evorl_screw_147_sft")
    p.add_argument("--checkpoint", default="outputs/rlt_demo_adapt_sft_v1/demo_adapt_checkpoint.pt")
    p.add_argument("--dataset-path", default="/home/coder/share/dataset")
    p.add_argument("--cache-dir", default="outputs/rlt_offline_cache_sft")
    p.add_argument("--output-dir", default="outputs/ac_best_sft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--token-pool-size", type=int, default=64)
    p.add_argument("--task-instruction", default="pick up the object")
    p.add_argument("--gradient-steps", type=int, default=50000)
    p.add_argument("--skip-cache", action="store_true", help="Skip cache build if it already exists")
    return p.parse_args()


def build_cache(args) -> None:
    """Build offline transition cache using SFT model."""
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
    from torch.utils.data import DataLoader, Subset

    config = RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]

    logger.info("Loading SFT pi0.5 from %s", args.model_path)
    vla = Pi05VLAAdapter(
        model_path=args.model_path,
        actual_action_dim=12,
        actual_proprio_dim=12,
        task_instruction=args.task_instruction,
        dtype="bfloat16",
        device=args.device,
        token_pool_size=args.token_pool_size,
    )
    logger.info("SFT pi0.5 loaded")

    agent = RLTAgent(config, vla).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])
    logger.info("Loaded SFT RL token checkpoint from %s", args.checkpoint)

    agent.freeze_vla()
    agent.freeze_rl_token_encoder()
    agent.eval()

    def reward_fn(expert_chunk, ref_chunk):
        return build_reward_seq(
            expert_chunk=expert_chunk,
            exec_chunk=expert_chunk,
            mode="hybrid",
            episode_success=True,
            is_terminal_chunk=False,
            success_bonus=10.0,
            progress_scale=1.0,
        )

    dataset = RLTDemoDataset(dataset_path=args.dataset_path, chunk_length=config.vla_horizon)
    num_episodes = _count_episodes(dataset)
    splits = split_episode_indices(num_episodes, train_ratio=0.8, val_ratio=0.1, seed=0)
    logger.info("Episodes: %d -> train=%d, val=%d, test=%d",
                num_episodes, len(splits["train"]), len(splits["val"]), len(splits["test"]))

    t_start = time.time()
    for split_name, episode_ids in splits.items():
        if not episode_ids:
            continue
        all_transitions = []
        for ep_idx, ep_id in enumerate(sorted(episode_ids)):
            frame_range = _episode_frame_range(dataset, ep_id)
            indices = list(range(frame_range[0], frame_range[1]))
            loader = DataLoader(
                Subset(dataset, indices), batch_size=32, shuffle=False,
                collate_fn=rlt_demo_collate, num_workers=0, drop_last=False,
            )
            transitions = build_transitions_from_demos(
                agent, loader, config.chunk_length, reward_fn=reward_fn, device=args.device,
            )
            if transitions:
                transitions[-1].reward_seq[-1] += 10.0
            all_transitions.extend(transitions)
            if (ep_idx + 1) % 20 == 0:
                logger.info("  [%s] %d/%d episodes, %d transitions", split_name, ep_idx + 1, len(episode_ids), len(all_transitions))
        save_cached_buffer(all_transitions, args.cache_dir, split_name)
        logger.info("Split %s: %d transitions", split_name, len(all_transitions))

    logger.info("Cache built in %.1fs", time.time() - t_start)


def train_best_ac(args) -> None:
    """Train ResidualMLP actor-critic with beta=0.3 on cached SFT transitions."""
    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_cached_buffer
    from lerobot.rlt.utils import soft_update
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    config = RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.rl_token.token_dim = 2048
    config.rl_token.enc_layers = 3
    config.rl_token.dec_layers = 3
    config.rl_token.ff_dim = 4096
    config.rl_token.num_rl_tokens = 4

    # Best architecture: ResidualMLP 3L/256h, beta=0.3
    config.actor.residual = True
    config.actor.num_layers = 3
    config.critic.residual = True
    config.critic.num_layers = 3
    config.training.beta = 0.3

    vla = DummyVLAAdapter(token_dim=2048, action_dim=12, num_tokens=64, horizon=50)
    agent = RLTAgent(config, vla).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])
    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    train_buffer = load_cached_buffer(args.cache_dir, "train", capacity=200000)
    val_buffer = load_cached_buffer(args.cache_dir, "val", capacity=200000)
    logger.info("Buffers: train=%d, val=%d", len(train_buffer), len(val_buffer))

    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=3e-4)

    agent.train()
    critic_count = 0
    actor_losses, critic_losses = [], []
    start = time.time()

    for step in range(1, args.gradient_steps + 1):
        batch = {k: v.to(args.device) for k, v in train_buffer.sample(256).items()}

        c_loss = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, 0.99, 10)
        critic_opt.zero_grad()
        c_loss.backward()
        critic_opt.step()
        critic_losses.append(c_loss.item())
        critic_count += 1

        if critic_count % 2 == 0:
            a_loss = actor_loss(agent.actor, agent.critic, batch, 0.3)
            actor_opt.zero_grad()
            a_loss.backward()
            actor_opt.step()
            actor_losses.append(a_loss.item())

        soft_update(agent.target_critic, agent.critic, 0.005)

        if step % 5000 == 0:
            avg_a = sum(actor_losses[-2500:]) / max(len(actor_losses[-2500:]), 1)
            avg_c = sum(critic_losses[-5000:]) / max(len(critic_losses[-5000:]), 1)
            logger.info("Step %d/%d  critic=%.4f  actor=%.4f", step, args.gradient_steps, avg_c, avg_a)

    elapsed = time.time() - start
    logger.info("Training done in %.1fs (%d steps/sec)", elapsed, args.gradient_steps / elapsed)

    agent.eval()
    eval_m = evaluate_offline(agent, val_buffer, config, num_batches=20)
    logger.info("Eval: ref_mse=%.5f expert_mse=%.4f q_gap=%.5f td_err=%.4f",
                eval_m.ref_action_mse, eval_m.expert_action_mse, eval_m.q_gap, eval_m.mean_critic_td_error)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "target_critic_state_dict": agent.target_critic.state_dict(),
        "rl_token_state_dict": agent.rl_token.state_dict(),
        "config": {
            "architecture": "residual_mlp",
            "actor_hidden": 256, "actor_layers": 3,
            "critic_hidden": 256, "critic_layers": 3,
            "activation": "relu", "beta": 0.3,
            "gradient_steps": args.gradient_steps,
            "vla_model": "pi05_evorl_screw_147_sft",
        },
        "metrics": {
            "ref_mse": eval_m.ref_action_mse,
            "expert_mse": eval_m.expert_action_mse,
            "q_gap": eval_m.q_gap,
            "td_error": eval_m.mean_critic_td_error,
            "final_actor_loss": sum(actor_losses[-1000:]) / 1000,
            "final_critic_loss": sum(critic_losses[-1000:]) / 1000,
        },
    }, out / "rl_checkpoint_best.pt")

    with open(out / "metrics.json", "w") as f:
        json.dump({
            "ref_mse": eval_m.ref_action_mse,
            "expert_mse": eval_m.expert_action_mse,
            "q_gap": eval_m.q_gap,
            "td_error": eval_m.mean_critic_td_error,
        }, f, indent=2)

    logger.info("Checkpoint saved to %s", out)


def main() -> None:
    args = parse_args()

    cache_path = Path(args.cache_dir) / "transitions_train.pt"
    if args.skip_cache and cache_path.exists():
        logger.info("Cache exists at %s, skipping build", args.cache_dir)
    else:
        logger.info("=== Step 1: Building SFT offline cache ===")
        build_cache(args)

    logger.info("=== Step 2: Training best actor-critic on SFT cache ===")
    train_best_ac(args)


if __name__ == "__main__":
    main()
