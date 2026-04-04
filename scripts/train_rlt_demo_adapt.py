#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(description="RLT demo adaptation with pi0.5.")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config.")
    parser.add_argument("--output-dir", default="outputs/rlt_demo_adapt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vla-ft-weight", type=float, default=0.0)
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens (0=no pool)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.rlt.algorithm import RLTAlgorithm
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.demo_loader import make_demo_loader
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.policy import RLTPolicy
    from lerobot.rlt.trainer import demo_adaptation

    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]
    if args.steps is not None:
        config.demo_adaptation.steps = args.steps
    if args.batch_size is not None:
        config.demo_adaptation.batch_size = args.batch_size
    if args.lr is not None:
        config.demo_adaptation.lr = args.lr
    config.demo_adaptation.vla_ft_weight = args.vla_ft_weight

    logger.info("Loading pi0.5 from %s", args.model_path)
    vla = Pi05VLAAdapter(
        model_path=args.model_path,
        actual_action_dim=12,
        actual_proprio_dim=12,
        task_instruction=args.task_instruction,
        dtype="bfloat16",
        device=args.device,
        cache_dir=args.cache_dir,
        token_pool_size=args.token_pool_size,
    )
    logger.info("pi0.5 loaded")

    policy = RLTPolicy(config, vla).to(args.device)
    algorithm = RLTAlgorithm(policy, config)
    logger.info(
        "RLTPolicy created. RL token params: %.1fM",
        sum(param.numel() for param in policy.rl_token.parameters()) / 1e6,
    )

    # Resume from checkpoint if specified
    start_step = 0
    prior_losses = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        policy.rl_token.load_state_dict(ckpt["rl_token_state_dict"], strict=False)
        start_step = ckpt.get("step", 0)
        prior_losses = ckpt.get("losses", [])
        logger.info("Resumed from step %d (loss=%.4f)", start_step, prior_losses[-1] if prior_losses else 0)

    demo_loader = make_demo_loader(
        dataset_path=args.dataset_path,
        batch_size=config.demo_adaptation.batch_size,
        chunk_length=config.vla_horizon,
        num_workers=args.num_workers,
        device=args.device,
    )
    logger.info("Demo loader ready")

    # Build full rl_token for demo adaptation (needs decoder)
    rl_token_full = algorithm.build_rl_token_full(args.device)
    if args.vla_ft_weight > 0:
        params = list(rl_token_full.parameters()) + list(policy.vla.parameters())
    else:
        params = list(rl_token_full.parameters())
    optimizer = torch.optim.Adam(params, lr=config.demo_adaptation.lr)

    logger.info(
        "Starting demo adaptation: steps=%d, lr=%.2e, grad_clip=%.1f, warmup=%d",
        config.demo_adaptation.steps, config.demo_adaptation.lr,
        config.demo_adaptation.grad_clip_norm, config.demo_adaptation.warmup_steps,
    )
    start_time = time.time()
    losses = demo_adaptation(
        algorithm, config, demo_loader, optimizer, rl_token_full=rl_token_full,
        save_dir=args.output_dir, save_every=args.save_every,
        start_step=start_step, prior_losses=prior_losses,
    )
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else 0.0
    avg_last_100 = sum(losses[-100:]) / min(len(losses), 100) if losses else 0
    logger.info("Done in %.1fs. Final loss: %.4f, Avg last 100: %.4f", elapsed, final_loss, avg_last_100)


if __name__ == "__main__":
    main()
