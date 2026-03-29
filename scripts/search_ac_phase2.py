#!/usr/bin/env python
"""Phase 2 architecture search: combine winning factors from Phase 1.

Best individual factors found in Phase 1:
- Beta: 0.05-0.1 (10x improvement)
- Hidden dim: 512+ (12x improvement)
- Activation: silu (11x improvement)
- LayerNorm, Residual: TBD from Phase 1
- Layers: 2-3

This script runs combination experiments to find the optimal joint config.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    actor_hidden: int = 256
    actor_layers: int = 2
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False
    actor_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    critic_hidden: int = 256
    critic_layers: int = 2
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = False
    critic_lr: float = 3e-4
    beta: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_update_interval: int = 2
    gradient_steps: int = 100000


def build_phase2_experiments() -> list[ExperimentConfig]:
    """Combine best factors from Phase 1."""
    experiments = []

    # === Group A: SiLU + low beta (core combo) ===
    for beta in [0.05, 0.1, 0.15, 0.2]:
        experiments.append(ExperimentConfig(
            name=f"silu_b{beta}",
            actor_activation="silu", critic_activation="silu",
            beta=beta,
        ))

    # === Group B: SiLU + wide + low beta ===
    for beta in [0.05, 0.1, 0.15]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b{beta}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=beta,
        ))

    # === Group C: SiLU + wide + 3 layers + low beta ===
    for beta in [0.05, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_l3_b{beta}",
            actor_hidden=512, actor_layers=3,
            critic_hidden=512, critic_layers=3,
            actor_activation="silu", critic_activation="silu",
            beta=beta,
        ))

    # === Group D: SiLU + wide + LayerNorm + low beta ===
    for beta in [0.05, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_ln_b{beta}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            actor_layer_norm=True, critic_layer_norm=True,
            beta=beta,
        ))

    # === Group E: SiLU + wide + Residual + LayerNorm + low beta ===
    for beta in [0.05, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_res_ln_b{beta}",
            actor_hidden=512, actor_layers=3,
            critic_hidden=512, critic_layers=3,
            actor_activation="silu", critic_activation="silu",
            actor_layer_norm=True, critic_layer_norm=True,
            actor_residual=True, critic_residual=True,
            beta=beta,
        ))

    # === Group F: 1024 wide + SiLU + low beta ===
    for beta in [0.05, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"silu_w1024_b{beta}",
            actor_hidden=1024, critic_hidden=1024,
            actor_activation="silu", critic_activation="silu",
            beta=beta,
        ))

    # === Group G: Asymmetric (wide actor, moderate critic) ===
    experiments.append(ExperimentConfig(
        name="silu_a1024_c512_b0.1",
        actor_hidden=1024, actor_layers=3,
        critic_hidden=512, critic_layers=2,
        actor_activation="silu", critic_activation="silu",
        beta=0.1,
    ))
    experiments.append(ExperimentConfig(
        name="silu_a512_c1024_b0.1",
        actor_hidden=512, actor_layers=2,
        critic_hidden=1024, critic_layers=3,
        actor_activation="silu", critic_activation="silu",
        beta=0.1,
    ))

    # === Group H: Learning rate variations on best arch ===
    for lr in [1e-4, 5e-4, 1e-3]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b0.1_lr{lr}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=0.1, actor_lr=lr, critic_lr=lr,
        ))

    # === Group I: Ref dropout on best arch ===
    for p in [0.0, 0.3, 0.7]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b0.1_rd{p}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=0.1, ref_dropout_p=p,
        ))

    # === Group J: Actor update interval on best arch ===
    for interval in [1, 4]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b0.1_ai{interval}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=0.1, actor_update_interval=interval,
        ))

    # === Group K: Tau sweep on best arch ===
    for tau in [0.001, 0.01]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b0.1_tau{tau}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=0.1, tau=tau,
        ))

    # === Group L: Fixed std sweep on best arch ===
    for std in [0.01, 0.02, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"silu_w512_b0.1_std{std}",
            actor_hidden=512, critic_hidden=512,
            actor_activation="silu", critic_activation="silu",
            beta=0.1, fixed_std=std,
        ))

    # === Group M: Full kitchen sink (best everything) ===
    experiments.append(ExperimentConfig(
        name="best_combo_v1",
        actor_hidden=512, actor_layers=3,
        critic_hidden=512, critic_layers=3,
        actor_activation="silu", critic_activation="silu",
        actor_layer_norm=True, critic_layer_norm=True,
        actor_residual=True, critic_residual=True,
        beta=0.1, actor_lr=3e-4, critic_lr=3e-4,
        ref_dropout_p=0.5, tau=0.005,
    ))

    return experiments


def run_experiment(exp, cache_dir, checkpoint, device, results_file):
    """Run a single experiment and return metrics."""
    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_cached_buffer
    from lerobot.rlt.utils import soft_update
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    logger.info("=" * 60)
    logger.info("Running: %s", exp.name)

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

    config.actor.hidden_dim = exp.actor_hidden
    config.actor.num_layers = exp.actor_layers
    config.actor.activation = exp.actor_activation
    config.actor.layer_norm = exp.actor_layer_norm
    config.actor.residual = exp.actor_residual
    config.actor.lr = exp.actor_lr
    config.actor.ref_dropout_p = exp.ref_dropout_p
    config.actor.fixed_std = exp.fixed_std

    config.critic.hidden_dim = exp.critic_hidden
    config.critic.num_layers = exp.critic_layers
    config.critic.activation = exp.critic_activation
    config.critic.layer_norm = exp.critic_layer_norm
    config.critic.residual = exp.critic_residual
    config.critic.lr = exp.critic_lr

    config.training.beta = exp.beta
    config.training.gamma = exp.gamma
    config.training.tau = exp.tau
    config.training.batch_size = exp.batch_size
    config.training.actor_update_interval = exp.actor_update_interval

    vla = DummyVLAAdapter(token_dim=2048, action_dim=12, num_tokens=64, horizon=50)
    agent = RLTAgent(config, vla).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])

    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    train_buffer = load_cached_buffer(cache_dir, "train", capacity=config.replay.capacity)
    val_buffer = load_cached_buffer(cache_dir, "val", capacity=config.replay.capacity)

    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=config.critic.lr)

    C = config.chunk_length
    gamma, beta, tau = config.training.gamma, config.training.beta, config.training.tau
    actor_interval = config.training.actor_update_interval
    critic_update_count = 0
    actor_losses, critic_losses = [], []

    agent.train()
    start_time = time.time()

    for step in range(1, exp.gradient_steps + 1):
        batch = {k: v.to(device) for k, v in train_buffer.sample(exp.batch_size).items()}

        c_loss = critic_loss(agent.critic, agent.target_critic, agent.actor, batch, gamma, C)
        critic_opt.zero_grad()
        c_loss.backward()
        critic_opt.step()
        critic_losses.append(c_loss.item())
        critic_update_count += 1

        if critic_update_count % actor_interval == 0:
            a_loss = actor_loss(agent.actor, agent.critic, batch, beta)
            actor_opt.zero_grad()
            a_loss.backward()
            actor_opt.step()
            actor_losses.append(a_loss.item())

        soft_update(agent.target_critic, agent.critic, tau)

        if step % 10000 == 0:
            avg_a = sum(actor_losses[-5000:]) / max(len(actor_losses[-5000:]), 1)
            avg_c = sum(critic_losses[-10000:]) / max(len(critic_losses[-10000:]), 1)
            logger.info("[%s] Step %d/%d  critic=%.4f  actor=%.4f", exp.name, step, exp.gradient_steps, avg_c, avg_a)

    elapsed = time.time() - start_time

    agent.eval()
    eval_metrics = evaluate_offline(agent, val_buffer, config, num_batches=20)

    final_actor = sum(actor_losses[-2000:]) / max(len(actor_losses[-2000:]), 1)
    final_critic = sum(critic_losses[-2000:]) / max(len(critic_losses[-2000:]), 1)

    result = {
        "name": exp.name,
        "config": asdict(exp),
        "final_actor_loss": final_actor,
        "final_critic_loss": final_critic,
        "min_actor_loss": min(actor_losses) if actor_losses else float("inf"),
        "ref_mse": eval_metrics.ref_action_mse,
        "expert_mse": eval_metrics.expert_action_mse,
        "q_gap": eval_metrics.q_gap,
        "mean_q_policy": eval_metrics.mean_q_policy,
        "mean_q_expert": eval_metrics.mean_q_expert,
        "td_error": eval_metrics.mean_critic_td_error,
        "elapsed_sec": elapsed,
        "steps_per_sec": exp.gradient_steps / elapsed,
        "actor_params": sum(p.numel() for p in agent.actor.parameters()),
        "critic_params": sum(p.numel() for p in agent.critic.parameters()),
    }

    logger.info(
        "[%s] DONE: actor=%.4f critic=%.4f ref_mse=%.4f q_gap=%.4f time=%.1fs",
        exp.name, final_actor, final_critic, eval_metrics.ref_action_mse, eval_metrics.q_gap, elapsed,
    )

    # Save result
    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    # Save best checkpoint
    output_dir = results_file.parent
    best_file = output_dir / "best_result.json"
    if best_file.exists():
        best = json.loads(best_file.read_text())
        if result["ref_mse"] < best["ref_mse"]:
            best_file.write_text(json.dumps(result, indent=2))
            torch.save({
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "target_critic_state_dict": agent.target_critic.state_dict(),
                "config": asdict(exp),
            }, output_dir / "best_checkpoint.pt")
            logger.info("[%s] NEW BEST! ref_mse=%.4f", exp.name, result["ref_mse"])
    else:
        best_file.write_text(json.dumps(result, indent=2))
        torch.save({
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "target_critic_state_dict": agent.target_critic.state_dict(),
            "config": asdict(exp),
        }, output_dir / "best_checkpoint.pt")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 actor-critic architecture search")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/ac_search_p2")
    parser.add_argument("--gradient-steps", type=int, default=100000)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--only", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    experiments = build_phase2_experiments()
    if args.only:
        experiments = [e for e in experiments if e.name == args.only]

    for exp in experiments:
        exp.gradient_steps = args.gradient_steps

    logger.info("Phase 2: %d experiments", len(experiments))
    for i, exp in enumerate(experiments):
        if i < args.start_from:
            continue
        logger.info("Experiment %d/%d: %s", i + 1, len(experiments), exp.name)
        run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    # Summary
    if results_file.exists():
        results = json.loads(results_file.read_text())
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 SUMMARY (sorted by ref_mse)")
        logger.info("=" * 80)
        for r in sorted(results, key=lambda x: x["ref_mse"]):
            logger.info(
                "%-40s actor=%.4f ref_mse=%.4f q_gap=%.5f",
                r["name"], r["final_actor_loss"], r["ref_mse"], r["q_gap"],
            )


if __name__ == "__main__":
    main()
