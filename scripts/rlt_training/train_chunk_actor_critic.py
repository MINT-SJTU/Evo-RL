#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import build_pi05_policy, configure_logging, load_training_config

logger = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chunk-level actor-critic from a transition cache or raw demo dataset.")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--transition-cache-dir", default=None, help="Directory containing chunk-transition cache files.")
    parser.add_argument("--demo-dataset-path", default=None, help="Raw demo dataset path. Required when no transition cache is provided.")
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config")
    parser.add_argument("--output-dir", default="outputs/rlt_actor_critic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rl-token-checkpoint", default=None, help="RL token checkpoint to initialize the policy encoder.")
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens before RL token encoding (0 disables pooling).")
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
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


def load_cached_replay_buffers(transition_cache_dir: str, capacity: int) -> tuple:
    from lerobot.rlt.offline_dataset import load_transition_cache

    train_buffer = load_transition_cache(transition_cache_dir, "train", capacity=capacity)
    val_buffer = load_transition_cache(transition_cache_dir, "val", capacity=capacity)
    return train_buffer, val_buffer


def build_live_replay_buffers(policy, args: argparse.Namespace, config) -> tuple:
    from lerobot.rlt.offline_dataset import build_transition_replay_buffer

    logger.info("Building train replay buffer from %s", args.demo_dataset_path)
    train_buffer = build_transition_replay_buffer(
        policy=policy,
        demo_dataset_path=args.demo_dataset_path,
        config=config,
        split="train",
        device=args.device,
    )
    logger.info("Building val replay buffer from %s", args.demo_dataset_path)
    val_buffer = build_transition_replay_buffer(
        policy=policy,
        demo_dataset_path=args.demo_dataset_path,
        config=config,
        split="val",
        device=args.device,
    )
    return train_buffer, val_buffer


def create_algorithm_with_cached_transitions(config, rl_token_checkpoint: str | None, device: str):
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
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
        logger.info("Loaded RL token checkpoint from %s", rl_token_checkpoint)

    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(device)
    return algorithm


def create_algorithm_with_pi05(config, args: argparse.Namespace):
    from lerobot.rlt.algorithm import RLTAlgorithm

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

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(args.device)
    return algorithm


def main() -> None:
    args = parse_args()

    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.trainer import offline_rl_loop

    config = load_training_config(args.config)
    apply_overrides(config, args)

    if args.transition_cache_dir is not None:
        logger.info("Loading transition cache from %s", args.transition_cache_dir)
        train_buffer, val_buffer = load_cached_replay_buffers(args.transition_cache_dir, config.replay.capacity)
        algorithm = create_algorithm_with_cached_transitions(config, args.rl_token_checkpoint, args.device)
    else:
        if args.demo_dataset_path is None:
            raise ValueError("--demo-dataset-path is required when --transition-cache-dir is not provided")
        algorithm = create_algorithm_with_pi05(config, args)
        train_buffer, val_buffer = build_live_replay_buffers(algorithm.policy, args, config)

    logger.info("Train transitions: %d, Val transitions: %d", len(train_buffer), len(val_buffer))
    logger.info(
        "Training chunk actor-critic: steps=%d batch_size=%d beta=%.2f actor_lr=%.2e critic_lr=%.2e",
        config.offline_rl.num_gradient_steps,
        config.training.batch_size,
        config.training.beta,
        config.actor.lr,
        config.critic.lr,
    )

    actor_optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    start_time = time.time()
    metrics = offline_rl_loop(
        algorithm=algorithm,
        config=config,
        replay_buffer=train_buffer,
        val_buffer=val_buffer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        save_dir=args.output_dir,
    )
    elapsed = time.time() - start_time
    logger.info(
        "Actor-critic training finished in %.1fs. critic_updates=%d actor_updates=%d",
        elapsed,
        len(metrics.critic_losses),
        len(metrics.actor_losses),
    )

    eval_metrics = evaluate_offline(algorithm, val_buffer, config, num_batches=10)
    logger.info(
        "Eval: expert_mse=%.4f ref_mse=%.4f q_policy=%.4f q_expert=%.4f q_gap=%.4f td_err=%.4f",
        eval_metrics.expert_action_mse,
        eval_metrics.ref_action_mse,
        eval_metrics.mean_q_policy,
        eval_metrics.mean_q_expert,
        eval_metrics.q_gap,
        eval_metrics.mean_critic_td_error,
    )


if __name__ == "__main__":
    main()
