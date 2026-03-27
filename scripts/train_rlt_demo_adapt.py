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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vla-ft-weight", type=float, default=0.0)
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.demo_loader import make_demo_loader
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.trainer import demo_adaptation

    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]
    config.demo_adaptation.steps = args.steps
    config.demo_adaptation.batch_size = args.batch_size
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
    )
    logger.info("pi0.5 loaded")

    agent = RLTAgent(config, vla).to(args.device)
    logger.info(
        "RLTAgent created. RL token params: %.1fM",
        sum(param.numel() for param in agent.rl_token.parameters()) / 1e6,
    )

    demo_loader = make_demo_loader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        chunk_length=config.vla_horizon,
        num_workers=args.num_workers,
        device=args.device,
    )
    logger.info("Demo loader ready")

    if args.vla_ft_weight > 0:
        params = list(agent.rl_token.parameters()) + list(agent.vla.parameters())
    else:
        params = list(agent.rl_token.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    logger.info("Starting demo adaptation for %d steps", args.steps)
    start_time = time.time()
    losses = demo_adaptation(agent, config, demo_loader, optimizer)
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else 0.0
    logger.info("Demo adaptation finished in %.1fs. Final loss: %.4f", elapsed, final_loss)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "rl_token_state_dict": agent.rl_token.state_dict(),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "losses": losses,
        "steps": args.steps,
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "demo_adapt_checkpoint.pt")

    with open(output_dir / "losses.json", "w") as handle:
        json.dump(losses, handle)

    logger.info("Saved checkpoint to %s", output_dir)


if __name__ == "__main__":
    main()
