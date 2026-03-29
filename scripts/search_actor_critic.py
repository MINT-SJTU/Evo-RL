#!/usr/bin/env python
"""Architecture search for actor-critic on cached offline RL transitions.

Runs experiments with different architecture configurations and logs results
to a structured JSON file for comparison.
"""
from __future__ import annotations

import argparse
import copy
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
    """Single experiment configuration."""

    name: str
    # Actor
    actor_hidden: int = 256
    actor_layers: int = 2
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False
    actor_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    # Critic
    critic_hidden: int = 256
    critic_layers: int = 2
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = False
    critic_lr: float = 3e-4
    # Training
    beta: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_update_interval: int = 2
    gradient_steps: int = 50000


def build_experiment_list() -> list[ExperimentConfig]:
    """Build the full list of experiments to run."""
    experiments = []

    # === Phase 1: Beta sweep (most impactful hyperparameter) ===
    for beta in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        experiments.append(ExperimentConfig(
            name=f"beta_{beta}",
            beta=beta,
        ))

    # === Phase 2: Hidden dim sweep ===
    for h in [128, 512, 1024]:
        experiments.append(ExperimentConfig(
            name=f"hidden_{h}",
            actor_hidden=h, critic_hidden=h,
        ))

    # === Phase 3: Layer count sweep ===
    for n in [3, 4, 5]:
        experiments.append(ExperimentConfig(
            name=f"layers_{n}",
            actor_layers=n, critic_layers=n,
        ))

    # === Phase 4: Activation functions ===
    for act in ["gelu", "silu", "tanh"]:
        experiments.append(ExperimentConfig(
            name=f"act_{act}",
            actor_activation=act, critic_activation=act,
        ))

    # === Phase 5: Layer normalization ===
    experiments.append(ExperimentConfig(
        name="layernorm",
        actor_layer_norm=True, critic_layer_norm=True,
    ))

    # === Phase 6: Residual connections ===
    experiments.append(ExperimentConfig(
        name="residual",
        actor_residual=True, critic_residual=True,
        actor_layers=3, critic_layers=3,
    ))

    # === Phase 7: Residual + LayerNorm ===
    experiments.append(ExperimentConfig(
        name="residual_ln",
        actor_residual=True, critic_residual=True,
        actor_layer_norm=True, critic_layer_norm=True,
        actor_layers=3, critic_layers=3,
    ))

    # === Phase 8: Learning rate sweep ===
    for lr in [1e-4, 1e-3]:
        experiments.append(ExperimentConfig(
            name=f"lr_{lr}",
            actor_lr=lr, critic_lr=lr,
        ))

    # === Phase 9: Actor-specific LR (higher critic LR) ===
    experiments.append(ExperimentConfig(
        name="critic_lr_high",
        actor_lr=3e-4, critic_lr=1e-3,
    ))

    # === Phase 10: Actor update frequency ===
    for interval in [1, 4]:
        experiments.append(ExperimentConfig(
            name=f"actor_interval_{interval}",
            actor_update_interval=interval,
        ))

    # === Phase 11: Tau sweep ===
    for tau in [0.001, 0.01, 0.02]:
        experiments.append(ExperimentConfig(
            name=f"tau_{tau}",
            tau=tau,
        ))

    # === Phase 12: Ref dropout sweep ===
    for p in [0.0, 0.3, 0.7]:
        experiments.append(ExperimentConfig(
            name=f"ref_dropout_{p}",
            ref_dropout_p=p,
        ))

    # === Phase 13: Fixed std sweep ===
    for std in [0.01, 0.02, 0.1]:
        experiments.append(ExperimentConfig(
            name=f"fixed_std_{std}",
            fixed_std=std,
        ))

    # === Phase 14: Batch size ===
    for bs in [128, 512, 1024]:
        experiments.append(ExperimentConfig(
            name=f"batch_{bs}",
            batch_size=bs,
        ))

    # === Phase 15: Gamma ===
    for g in [0.95, 0.999]:
        experiments.append(ExperimentConfig(
            name=f"gamma_{g}",
            gamma=g,
        ))

    # === Phase 16: Wider actor, narrow critic ===
    experiments.append(ExperimentConfig(
        name="wide_actor_narrow_critic",
        actor_hidden=512, actor_layers=3,
        critic_hidden=256, critic_layers=2,
    ))

    # === Phase 17: Narrow actor, wider critic ===
    experiments.append(ExperimentConfig(
        name="narrow_actor_wide_critic",
        actor_hidden=256, actor_layers=2,
        critic_hidden=512, critic_layers=3,
    ))

    # === Phase 18: Big + all features ===
    experiments.append(ExperimentConfig(
        name="big_residual_ln_gelu",
        actor_hidden=512, actor_layers=4,
        critic_hidden=512, critic_layers=4,
        actor_activation="gelu", critic_activation="gelu",
        actor_layer_norm=True, critic_layer_norm=True,
        actor_residual=True, critic_residual=True,
    ))

    # === Phase 19: SiLU + residual + LN (modern architecture) ===
    experiments.append(ExperimentConfig(
        name="modern_silu",
        actor_hidden=512, actor_layers=3,
        critic_hidden=512, critic_layers=3,
        actor_activation="silu", critic_activation="silu",
        actor_layer_norm=True, critic_layer_norm=True,
        actor_residual=True, critic_residual=True,
    ))

    return experiments


def run_experiment(
    exp: ExperimentConfig,
    cache_dir: str,
    checkpoint: str | None,
    device: str,
    results_file: Path,
) -> dict:
    """Run a single experiment and return metrics."""
    from lerobot.rlt.agent import RLTAgent
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_cached_buffer
    from lerobot.rlt.replay_buffer import ReplayBuffer
    from lerobot.rlt.utils import soft_update
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    logger.info("=" * 60)
    logger.info("Running experiment: %s", exp.name)
    logger.info("Config: %s", json.dumps(asdict(exp), indent=2))

    # Build config
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

    # Create agent
    vla = DummyVLAAdapter(token_dim=2048, action_dim=12, num_tokens=64, horizon=50)
    agent = RLTAgent(config, vla).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        agent.rl_token.load_state_dict(ckpt["rl_token_state_dict"])

    agent.freeze_vla()
    agent.freeze_rl_token_encoder()

    # Load cached transitions
    train_buffer = load_cached_buffer(cache_dir, "train", capacity=config.replay.capacity)
    val_buffer = load_cached_buffer(cache_dir, "val", capacity=config.replay.capacity)
    logger.info("Buffers loaded: train=%d, val=%d", len(train_buffer), len(val_buffer))

    # Optimizers
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=config.critic.lr)

    # Training loop
    C = config.chunk_length
    gamma = config.training.gamma
    beta = config.training.beta
    tau = config.training.tau
    actor_interval = config.training.actor_update_interval
    critic_update_count = 0
    actor_losses = []
    critic_losses = []

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

        if step % 5000 == 0:
            avg_a = sum(actor_losses[-2500:]) / max(len(actor_losses[-2500:]), 1)
            avg_c = sum(critic_losses[-5000:]) / max(len(critic_losses[-5000:]), 1)
            logger.info(
                "[%s] Step %d/%d  critic=%.4f  actor=%.4f",
                exp.name, step, exp.gradient_steps, avg_c, avg_a,
            )

    elapsed = time.time() - start_time

    # Evaluation
    agent.eval()
    eval_metrics = evaluate_offline(agent, val_buffer, config, num_batches=20)

    # Compute summary stats
    final_actor_loss = sum(actor_losses[-1000:]) / max(len(actor_losses[-1000:]), 1)
    final_critic_loss = sum(critic_losses[-1000:]) / max(len(critic_losses[-1000:]), 1)
    min_actor_loss = min(actor_losses) if actor_losses else float("inf")

    result = {
        "name": exp.name,
        "config": asdict(exp),
        "final_actor_loss": final_actor_loss,
        "final_critic_loss": final_critic_loss,
        "min_actor_loss": min_actor_loss,
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
        exp.name, final_actor_loss, final_critic_loss,
        eval_metrics.ref_action_mse, eval_metrics.q_gap, elapsed,
    )

    # Append result to file
    _append_result(results_file, result)
    return result


def _append_result(results_file: Path, result: dict) -> None:
    """Append a result to the JSON results file (list of dicts)."""
    results = []
    if results_file.exists():
        results = json.loads(results_file.read_text())
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actor-critic architecture search")
    parser.add_argument("--cache-dir", required=True, help="Dir with cached transitions")
    parser.add_argument("--checkpoint", default=None, help="RL token checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/ac_search")
    parser.add_argument("--gradient-steps", type=int, default=50000)
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N experiments")
    parser.add_argument("--only", default=None, help="Run only experiment with this name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    experiments = build_experiment_list()

    if args.only:
        experiments = [e for e in experiments if e.name == args.only]
        if not experiments:
            logger.error("No experiment found with name: %s", args.only)
            return

    # Override gradient steps
    for exp in experiments:
        exp.gradient_steps = args.gradient_steps

    logger.info("Total experiments: %d", len(experiments))
    for i, exp in enumerate(experiments):
        if i < args.start_from:
            logger.info("Skipping %d: %s", i, exp.name)
            continue
        logger.info("Experiment %d/%d: %s", i + 1, len(experiments), exp.name)
        run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    # Print summary
    if results_file.exists():
        results = json.loads(results_file.read_text())
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY (sorted by final_actor_loss)")
        logger.info("=" * 80)
        results.sort(key=lambda r: r["final_actor_loss"])
        for r in results:
            logger.info(
                "%-35s actor=%.4f critic=%.4f ref_mse=%.4f q_gap=%.5f",
                r["name"], r["final_actor_loss"], r["final_critic_loss"],
                r["ref_mse"], r["q_gap"],
            )


if __name__ == "__main__":
    main()
