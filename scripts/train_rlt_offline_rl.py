#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="RLT offline RL training on cached/precomputed demo data.")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--cache-dir", default=None, help="Dir with cached transitions (skip VLA inference)")
    parser.add_argument("--dataset-path", default=None, help="LeRobot dataset path (required if no --cache-dir)")
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config")
    parser.add_argument("--output-dir", default="outputs/rlt_offline_rl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None, help="Demo adaptation checkpoint to load")
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens (0=no pool)")
    parser.add_argument("--task-instruction", default="pick up the object")
    return parser.parse_args()


def _apply_overrides(config, args: argparse.Namespace) -> None:
    """Apply CLI overrides to config (mutates in-place)."""
    if args.gradient_steps is not None:
        config.offline_rl.num_gradient_steps = args.gradient_steps
    if args.eval_every is not None:
        config.offline_rl.eval_every = args.eval_every
    if args.save_every is not None:
        config.offline_rl.save_every = args.save_every
    if args.log_every is not None:
        config.offline_rl.log_every = args.log_every
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.beta is not None:
        config.training.beta = args.beta
    if args.actor_lr is not None:
        config.actor.lr = args.actor_lr
    if args.critic_lr is not None:
        config.critic.lr = args.critic_lr


def _load_buffers_from_cache(cache_dir: str, capacity: int) -> tuple:
    from lerobot.rlt.offline_dataset import load_cached_buffer

    train_buffer = load_cached_buffer(cache_dir, "train", capacity=capacity)
    val_buffer = load_cached_buffer(cache_dir, "val", capacity=capacity)
    return train_buffer, val_buffer


def _precompute_buffers(policy, args: argparse.Namespace, config) -> tuple:
    from lerobot.rlt.offline_dataset import precompute_offline_buffer
    from lerobot.rlt.rewards import build_reward_seq
    from functools import partial

    reward_fn = partial(
        build_reward_seq, mode=config.offline_rl.reward_mode,
        success_bonus=config.offline_rl.success_bonus,
        progress_scale=config.offline_rl.progress_scale,
    )
    logger.info("Precomputing train buffer from %s (reward_mode=%s)", args.dataset_path, config.offline_rl.reward_mode)
    train_buffer = precompute_offline_buffer(policy, args.dataset_path, config, split="train", device=args.device, reward_fn=reward_fn)
    logger.info("Precomputing val buffer from %s", args.dataset_path)
    val_buffer = precompute_offline_buffer(policy, args.dataset_path, config, split="val", device=args.device, reward_fn=reward_fn)
    return train_buffer, val_buffer


def _create_algorithm_with_cache(config, checkpoint_path: str | None, device: str):
    """Create algorithm with DummyVLAAdapter (no Pi05 needed when using cached transitions)."""
    from lerobot.rlt.algorithm import RLTAlgorithm
    from lerobot.rlt.policy import RLTPolicy
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    vla = DummyVLAAdapter(
        token_dim=config.rl_token.token_dim,
        action_dim=config.action_dim,
        num_tokens=64,
        horizon=config.vla_horizon,
    )
    policy = RLTPolicy(config, vla).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(ckpt["rl_token_state_dict"], strict=False)
        logger.info("Loaded RL token weights from %s", checkpoint_path)

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(device)
    return algorithm


def _create_algorithm_with_vla(config, args: argparse.Namespace):
    """Create algorithm with real Pi05VLAAdapter (needed for precomputing buffers)."""
    from lerobot.rlt.algorithm import RLTAlgorithm
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.policy import RLTPolicy

    logger.info("Loading pi0.5 from %s", args.model_path)
    vla = Pi05VLAAdapter(
        model_path=args.model_path,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=args.task_instruction,
        dtype="bfloat16",
        device=args.device,
        token_pool_size=args.token_pool_size,
    )
    policy = RLTPolicy(config, vla).to(args.device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        policy.rl_token.load_state_dict(ckpt["rl_token_state_dict"], strict=False)
        logger.info("Loaded RL token weights from %s", args.checkpoint)

    policy.freeze_vla()
    policy.freeze_rl_token_encoder()

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(args.device)
    return algorithm


def main() -> None:
    args = parse_args()

    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.trainer import offline_rl_loop

    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]
    _apply_overrides(config, args)

    # Load or precompute replay buffers
    if args.cache_dir:
        logger.info("Loading cached transitions from %s", args.cache_dir)
        train_buffer, val_buffer = _load_buffers_from_cache(args.cache_dir, config.replay.capacity)
        algorithm = _create_algorithm_with_cache(config, args.checkpoint, args.device)
    else:
        assert args.dataset_path, "--dataset-path is required when --cache-dir is not provided"
        algorithm = _create_algorithm_with_vla(config, args)
        train_buffer, val_buffer = _precompute_buffers(algorithm.policy, args, config)

    logger.info("Train buffer: %d transitions, Val buffer: %d transitions", len(train_buffer), len(val_buffer))
    logger.info(
        "Starting offline RL: %d gradient steps, batch_size=%d, beta=%.2f, actor_lr=%.2e, critic_lr=%.2e",
        config.offline_rl.num_gradient_steps, config.training.batch_size,
        config.training.beta, config.actor.lr, config.critic.lr,
    )

    actor_opt = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    start_time = time.time()
    metrics = offline_rl_loop(
        algorithm, config, train_buffer,
        val_buffer=val_buffer,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        save_dir=args.output_dir,
    )
    elapsed = time.time() - start_time

    logger.info(
        "Offline RL finished in %.1fs. Critic updates: %d, Actor updates: %d",
        elapsed, len(metrics.critic_losses), len(metrics.actor_losses),
    )

    # Final evaluation on validation buffer
    eval_metrics = evaluate_offline(algorithm, val_buffer, config, num_batches=10)
    logger.info(
        "Eval: expert_mse=%.4f, ref_mse=%.4f, Q_policy=%.4f, Q_expert=%.4f, Q_gap=%.4f, TD_err=%.4f",
        eval_metrics.expert_action_mse, eval_metrics.ref_action_mse,
        eval_metrics.mean_q_policy, eval_metrics.mean_q_expert,
        eval_metrics.q_gap, eval_metrics.mean_critic_td_error,
    )


if __name__ == "__main__":
    main()
