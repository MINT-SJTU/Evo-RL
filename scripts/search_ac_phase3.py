#!/usr/bin/env python
"""Phase 3: Fine-tune around the best config from Phase 2.

Best config: residual MLP, 3 layers, 256 hidden, relu, beta=0.3
This script fine-tunes around that config with longer training and
micro-adjustments to find the absolute best.
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
    actor_layers: int = 3
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = True
    actor_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    critic_hidden: int = 256
    critic_layers: int = 3
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = True
    critic_lr: float = 3e-4
    beta: float = 0.3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_update_interval: int = 2
    gradient_steps: int = 200000


def build_phase3_experiments() -> list[ExperimentConfig]:
    """Fine-tune around res_b0.3 winner."""
    experiments = []

    # === Group A: Fine beta sweep around 0.3 ===
    for beta in [0.25, 0.35, 0.4]:
        experiments.append(ExperimentConfig(name=f"fine_b{beta}", beta=beta))

    # === Group B: Longer training at optimal beta ===
    experiments.append(ExperimentConfig(
        name="res_b0.3_300k", beta=0.3, gradient_steps=300000,
    ))

    # === Group C: LR sweep around winner ===
    for lr in [1e-4, 2e-4, 5e-4]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_lr{lr}", beta=0.3, actor_lr=lr, critic_lr=lr,
        ))

    # === Group D: Asymmetric LR (higher critic to stabilize Q) ===
    experiments.append(ExperimentConfig(
        name="res_b0.3_alr1e4_clr3e4", beta=0.3, actor_lr=1e-4, critic_lr=3e-4,
    ))
    experiments.append(ExperimentConfig(
        name="res_b0.3_alr3e4_clr1e3", beta=0.3, actor_lr=3e-4, critic_lr=1e-3,
    ))

    # === Group E: Tau fine-tune ===
    for tau in [0.001, 0.003, 0.01]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_tau{tau}", beta=0.3, tau=tau,
        ))

    # === Group F: Actor update interval ===
    for ai in [1, 3, 4]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_ai{ai}", beta=0.3, actor_update_interval=ai,
        ))

    # === Group G: Fixed std ===
    for std in [0.01, 0.02, 0.03]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_std{std}", beta=0.3, fixed_std=std,
        ))

    # === Group H: Ref dropout ===
    for rd in [0.0, 0.3, 0.7]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_rd{rd}", beta=0.3, ref_dropout_p=rd,
        ))

    # === Group I: Batch size ===
    for bs in [128, 512]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_bs{bs}", beta=0.3, batch_size=bs,
        ))

    # === Group J: Layer count ===
    for nl in [2, 4]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_l{nl}", beta=0.3,
            actor_layers=nl, critic_layers=nl,
        ))

    # === Group K: Hidden dim ===
    for hd in [384, 512]:
        experiments.append(ExperimentConfig(
            name=f"res_b0.3_h{hd}", beta=0.3,
            actor_hidden=hd, critic_hidden=hd,
        ))

    # === Group L: Best of best — longer training ===
    experiments.append(ExperimentConfig(
        name="res_b0.3_500k", beta=0.3, gradient_steps=500000,
    ))

    return experiments


def run_experiment(exp, cache_dir, checkpoint, device, results_file):
    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_cached_buffer
    from lerobot.rlt.utils import soft_update
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    logger.info("=" * 60)
    logger.info("Running: %s (%d steps)", exp.name, exp.gradient_steps)

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

        if step % 20000 == 0:
            avg_a = sum(actor_losses[-10000:]) / max(len(actor_losses[-10000:]), 1)
            avg_c = sum(critic_losses[-20000:]) / max(len(critic_losses[-20000:]), 1)
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

    logger.info("[%s] DONE: actor=%.4f ref_mse=%.4f q_gap=%.4f time=%.1fs",
        exp.name, final_actor, eval_metrics.ref_action_mse, eval_metrics.q_gap, elapsed)

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/ac_search_p3")
    parser.add_argument("--start-from", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    experiments = build_phase3_experiments()

    logger.info("Phase 3: %d experiments", len(experiments))
    for i, exp in enumerate(experiments):
        if i < args.start_from:
            continue
        logger.info("Experiment %d/%d: %s (%d steps)", i + 1, len(experiments), exp.name, exp.gradient_steps)
        run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    if results_file.exists():
        results = json.loads(results_file.read_text())
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3 SUMMARY (sorted by ref_mse)")
        for r in sorted(results, key=lambda x: x["ref_mse"]):
            logger.info("%-40s actor=%.4f ref_mse=%.5f q_gap=%.5f",
                r["name"], r["final_actor_loss"], r["ref_mse"], r["q_gap"])


if __name__ == "__main__":
    main()
