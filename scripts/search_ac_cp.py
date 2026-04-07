#!/usr/bin/env python
"""Autonomous AC architecture search on CP-only offline cache.

3-phase search:
  Phase 1: Factor isolation (20K steps, ~30 experiments)
  Phase 2: Combine winners (50K steps, ~15 experiments)
  Phase 3: Fine-tune best (100K-200K steps, ~10 experiments)

Designed to run autonomously for ~9 hours.
"""
from __future__ import annotations

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
    gradient_steps: int = 20000


def run_experiment(exp, cache_dir, checkpoint, device, results_file):
    from lerobot.rlt.algorithm import RLTAlgorithm
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_cached_buffer
    from lerobot.rlt.policy import RLTPolicy
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
    policy = RLTPolicy(config, vla).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(ckpt["rl_token_state_dict"], strict=False)

    policy.freeze_vla()
    policy.freeze_rl_token_encoder()

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(device)

    train_buffer = load_cached_buffer(cache_dir, "train", capacity=config.replay.capacity)
    val_buffer = load_cached_buffer(cache_dir, "val", capacity=config.replay.capacity)

    actor_opt = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    C = config.chunk_length
    algorithm.train()
    start_time = time.time()
    actor_losses = []
    critic_losses = []
    critic_count = 0

    for step in range(1, exp.gradient_steps + 1):
        batch = {k: v.to(device) for k, v in train_buffer.sample(exp.batch_size).items()}
        c_loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, exp.gamma, C)
        critic_opt.zero_grad()
        c_loss.backward()
        critic_opt.step()
        critic_losses.append(c_loss.item())
        critic_count += 1

        if critic_count % exp.actor_update_interval == 0:
            a_loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, exp.beta)
            actor_opt.zero_grad()
            a_loss.backward()
            actor_opt.step()
            actor_losses.append(a_loss.item())

        soft_update(algorithm.target_critic, algorithm.critic, exp.tau)

        if step % 10000 == 0:
            avg_a = sum(actor_losses[-5000:]) / max(len(actor_losses[-5000:]), 1)
            avg_c = sum(critic_losses[-10000:]) / max(len(critic_losses[-10000:]), 1)
            logger.info("[%s] %d/%d critic=%.4f actor=%.4f", exp.name, step, exp.gradient_steps, avg_c, avg_a)

    elapsed = time.time() - start_time
    algorithm.eval()
    eval_m = evaluate_offline(algorithm, val_buffer, config, num_batches=20)

    result = {
        "name": exp.name,
        "config": asdict(exp),
        "ref_mse": eval_m.ref_action_mse,
        "expert_mse": eval_m.expert_action_mse,
        "q_gap": eval_m.q_gap,
        "mean_q_policy": eval_m.mean_q_policy,
        "mean_q_expert": eval_m.mean_q_expert,
        "td_error": eval_m.mean_critic_td_error,
        "final_actor_loss": sum(actor_losses[-1000:]) / max(len(actor_losses[-1000:]), 1),
        "final_critic_loss": sum(critic_losses[-1000:]) / max(len(critic_losses[-1000:]), 1),
        "elapsed_sec": elapsed,
    }

    logger.info(
        "[%s] DONE ref_mse=%.4f q_gap=%.4f actor=%.4f time=%.1fs",
        exp.name, result["ref_mse"], result["q_gap"], result["final_actor_loss"], elapsed,
    )

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    # Free GPU memory
    del algorithm, policy, vla, train_buffer, val_buffer, actor_opt, critic_opt
    torch.cuda.empty_cache()

    return result


def phase1_experiments():
    """Factor isolation: vary one thing at a time."""
    exps = []

    # Baseline (previous winner from base model search)
    exps.append(ExperimentConfig(name="p1_baseline", gradient_steps=20000))
    exps.append(ExperimentConfig(
        name="p1_prev_winner_res_b0.3",
        actor_residual=True, critic_residual=True,
        actor_layers=3, critic_layers=3,
        beta=0.3, gradient_steps=20000,
    ))

    # Beta sweep (critical for offline RL)
    for beta in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        exps.append(ExperimentConfig(name=f"p1_beta_{beta}", beta=beta, gradient_steps=20000))

    # Residual connections
    for layers in [2, 3, 4]:
        exps.append(ExperimentConfig(
            name=f"p1_res_L{layers}",
            actor_residual=True, critic_residual=True,
            actor_layers=layers, critic_layers=layers,
            gradient_steps=20000,
        ))

    # Hidden dim
    for h in [128, 256, 512, 1024]:
        exps.append(ExperimentConfig(
            name=f"p1_hidden_{h}",
            actor_hidden=h, critic_hidden=h,
            gradient_steps=20000,
        ))

    # Activations
    for act in ["gelu", "silu", "tanh"]:
        exps.append(ExperimentConfig(
            name=f"p1_act_{act}",
            actor_activation=act, critic_activation=act,
            gradient_steps=20000,
        ))

    # LayerNorm
    exps.append(ExperimentConfig(
        name="p1_layernorm",
        actor_layer_norm=True, critic_layer_norm=True,
        gradient_steps=20000,
    ))

    # Batch size (important for small buffer ~5500 train transitions)
    for bs in [64, 128, 256, 512]:
        exps.append(ExperimentConfig(name=f"p1_batch_{bs}", batch_size=bs, gradient_steps=20000))

    # LR
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        exps.append(ExperimentConfig(
            name=f"p1_lr_{lr}",
            actor_lr=lr, critic_lr=lr,
            gradient_steps=20000,
        ))

    # Ref dropout
    for p in [0.0, 0.3, 0.5, 0.7, 1.0]:
        exps.append(ExperimentConfig(name=f"p1_refdrop_{p}", ref_dropout_p=p, gradient_steps=20000))

    # Fixed std
    for std in [0.01, 0.02, 0.05, 0.1, 0.2]:
        exps.append(ExperimentConfig(name=f"p1_std_{std}", fixed_std=std, gradient_steps=20000))

    # Tau
    for tau in [0.001, 0.005, 0.01, 0.02]:
        exps.append(ExperimentConfig(name=f"p1_tau_{tau}", tau=tau, gradient_steps=20000))

    # Gamma
    for g in [0.9, 0.95, 0.99, 0.999]:
        exps.append(ExperimentConfig(name=f"p1_gamma_{g}", gamma=g, gradient_steps=20000))

    return exps


def pick_phase2_experiments(results):
    """Combine top factors from Phase 1."""
    # Find best value for each factor
    best = {}
    for r in results:
        name = r["name"]
        mse = r["ref_mse"]
        if name.startswith("p1_beta_"):
            best.setdefault("beta", []).append((mse, r["config"]["beta"]))
        elif name.startswith("p1_res_"):
            best.setdefault("residual", []).append((mse, r["config"]["actor_layers"]))
        elif name.startswith("p1_hidden_"):
            best.setdefault("hidden", []).append((mse, r["config"]["actor_hidden"]))
        elif name.startswith("p1_act_"):
            best.setdefault("activation", []).append((mse, r["config"]["actor_activation"]))
        elif name.startswith("p1_batch_"):
            best.setdefault("batch", []).append((mse, r["config"]["batch_size"]))
        elif name.startswith("p1_lr_"):
            best.setdefault("lr", []).append((mse, r["config"]["actor_lr"]))
        elif name.startswith("p1_refdrop_"):
            best.setdefault("refdrop", []).append((mse, r["config"]["ref_dropout_p"]))
        elif name.startswith("p1_std_"):
            best.setdefault("std", []).append((mse, r["config"]["fixed_std"]))
        elif name.startswith("p1_tau_"):
            best.setdefault("tau", []).append((mse, r["config"]["tau"]))
        elif name.startswith("p1_gamma_"):
            best.setdefault("gamma", []).append((mse, r["config"]["gamma"]))

    winners = {}
    for factor, vals in best.items():
        vals.sort()
        winners[factor] = vals[0][1]  # best (lowest ref_mse)
        logger.info("Phase 1 winner for %s: %s (ref_mse=%.4f)", factor, vals[0][1], vals[0][0])

    # Also check if layernorm helped
    ln_results = [r for r in results if r["name"] == "p1_layernorm"]
    baseline_results = [r for r in results if r["name"] == "p1_baseline"]
    ln_helps = False
    if ln_results and baseline_results:
        ln_helps = ln_results[0]["ref_mse"] < baseline_results[0]["ref_mse"]
        logger.info("LayerNorm %s (%.4f vs %.4f baseline)",
                     "HELPS" if ln_helps else "hurts",
                     ln_results[0]["ref_mse"], baseline_results[0]["ref_mse"])

    # Build Phase 2 combinations
    exps = []
    base = ExperimentConfig(name="", gradient_steps=50000)

    # Apply best beta
    if "beta" in winners:
        base.beta = winners["beta"]

    # Combo 1: best beta + residual
    if "residual" in winners:
        e = copy.copy(base)
        e.name = "p2_res_bestbeta"
        e.actor_residual = True
        e.critic_residual = True
        e.actor_layers = winners["residual"]
        e.critic_layers = winners["residual"]
        exps.append(e)

    # Combo 2: best beta + best hidden
    if "hidden" in winners:
        e = copy.copy(base)
        e.name = "p2_hidden_bestbeta"
        e.actor_hidden = winners["hidden"]
        e.critic_hidden = winners["hidden"]
        exps.append(e)

    # Combo 3: residual + best hidden + best beta
    e = copy.copy(base)
    e.name = "p2_res_hidden_beta"
    e.actor_residual = True
    e.critic_residual = True
    e.actor_layers = winners.get("residual", 3)
    e.critic_layers = winners.get("residual", 3)
    e.actor_hidden = winners.get("hidden", 256)
    e.critic_hidden = winners.get("hidden", 256)
    exps.append(e)

    # Combo 4: + best activation
    if "activation" in winners and winners["activation"] != "relu":
        e2 = copy.copy(e)
        e2.name = "p2_res_hidden_beta_act"
        e2.actor_activation = winners["activation"]
        e2.critic_activation = winners["activation"]
        exps.append(e2)

    # Combo 5: + layernorm (if it helped)
    if ln_helps:
        e3 = copy.copy(e)
        e3.name = "p2_res_hidden_beta_ln"
        e3.actor_layer_norm = True
        e3.critic_layer_norm = True
        exps.append(e3)

    # Combo 6: all winners combined
    e_all = copy.copy(base)
    e_all.name = "p2_all_winners"
    e_all.actor_residual = True
    e_all.critic_residual = True
    e_all.actor_layers = winners.get("residual", 3)
    e_all.critic_layers = winners.get("residual", 3)
    e_all.actor_hidden = winners.get("hidden", 256)
    e_all.critic_hidden = winners.get("hidden", 256)
    if "activation" in winners:
        e_all.actor_activation = winners["activation"]
        e_all.critic_activation = winners["activation"]
    if ln_helps:
        e_all.actor_layer_norm = True
        e_all.critic_layer_norm = True
    if "lr" in winners:
        e_all.actor_lr = winners["lr"]
        e_all.critic_lr = winners["lr"]
    if "refdrop" in winners:
        e_all.ref_dropout_p = winners["refdrop"]
    if "std" in winners:
        e_all.fixed_std = winners["std"]
    if "tau" in winners:
        e_all.tau = winners["tau"]
    if "gamma" in winners:
        e_all.gamma = winners["gamma"]
    if "batch" in winners:
        e_all.batch_size = winners["batch"]
    exps.append(e_all)

    # Combo 7: previous winner (res 3L + β=0.3) at 50K for comparison
    e_prev = ExperimentConfig(
        name="p2_prev_winner_50k",
        actor_residual=True, critic_residual=True,
        actor_layers=3, critic_layers=3,
        beta=0.3, gradient_steps=50000,
    )
    exps.append(e_prev)

    # Combo 8-10: LR fine-tuning on best combo
    for lr_mult in [0.3, 1.0, 3.0]:
        e_lr = copy.copy(e_all)
        e_lr.name = f"p2_all_lr_x{lr_mult}"
        e_lr.actor_lr = e_all.actor_lr * lr_mult
        e_lr.critic_lr = e_all.critic_lr * lr_mult
        exps.append(e_lr)

    return exps, winners


def pick_phase3_experiments(results, winners):
    """Fine-tune around the Phase 2 winner."""
    p2_results = [r for r in results if r["name"].startswith("p2_")]
    if not p2_results:
        return []

    p2_results.sort(key=lambda r: r["ref_mse"])
    best = p2_results[0]
    cfg = best["config"]
    logger.info("Phase 2 winner: %s (ref_mse=%.4f)", best["name"], best["ref_mse"])

    exps = []
    base = ExperimentConfig(**{k: v for k, v in cfg.items() if k in ExperimentConfig.__dataclass_fields__})

    # Fine-tune with longer training
    for steps in [100000, 200000]:
        e = copy.copy(base)
        e.name = f"p3_winner_{steps // 1000}k"
        e.gradient_steps = steps
        exps.append(e)

    # LR decay variants
    for lr_mult in [0.5, 0.1]:
        e = copy.copy(base)
        e.name = f"p3_winner_lr_x{lr_mult}"
        e.gradient_steps = 100000
        e.actor_lr = base.actor_lr * lr_mult
        e.critic_lr = base.critic_lr * lr_mult
        exps.append(e)

    # Beta fine-tuning around winner
    winner_beta = base.beta
    for delta in [-0.5, -0.1, 0.1, 0.5]:
        new_beta = max(0.01, winner_beta + delta)
        e = copy.copy(base)
        e.name = f"p3_beta_{new_beta:.2f}"
        e.beta = new_beta
        e.gradient_steps = 100000
        exps.append(e)

    # Batch size variants (important for small buffer)
    for bs in [32, 64, 128]:
        if bs != base.batch_size:
            e = copy.copy(base)
            e.name = f"p3_batch_{bs}"
            e.batch_size = bs
            e.gradient_steps = 100000
            exps.append(e)

    return exps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/ac_search_cp")
    parser.add_argument("--skip-phase1", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    total_start = time.time()

    # Phase 1
    if not args.skip_phase1:
        logger.info("=" * 80)
        logger.info("PHASE 1: Factor isolation (20K steps each)")
        logger.info("=" * 80)
        p1_exps = phase1_experiments()
        logger.info("Phase 1: %d experiments", len(p1_exps))
        for i, exp in enumerate(p1_exps):
            logger.info("--- P1 experiment %d/%d: %s ---", i + 1, len(p1_exps), exp.name)
            run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    # Load all results
    results = json.loads(results_file.read_text()) if results_file.exists() else []
    p1_results = [r for r in results if r["name"].startswith("p1_")]

    # Phase 2
    logger.info("=" * 80)
    logger.info("PHASE 2: Combine winners (50K steps each)")
    logger.info("=" * 80)
    p2_exps, winners = pick_phase2_experiments(p1_results)
    logger.info("Phase 2: %d experiments", len(p2_exps))
    for i, exp in enumerate(p2_exps):
        logger.info("--- P2 experiment %d/%d: %s ---", i + 1, len(p2_exps), exp.name)
        run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    # Reload
    results = json.loads(results_file.read_text())

    # Phase 3
    logger.info("=" * 80)
    logger.info("PHASE 3: Fine-tune winner (100K-200K steps)")
    logger.info("=" * 80)
    p3_exps = pick_phase3_experiments(results, winners)
    logger.info("Phase 3: %d experiments", len(p3_exps))
    for i, exp in enumerate(p3_exps):
        logger.info("--- P3 experiment %d/%d: %s ---", i + 1, len(p3_exps), exp.name)
        run_experiment(exp, args.cache_dir, args.checkpoint, args.device, results_file)

    # Final summary
    results = json.loads(results_file.read_text())
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY (sorted by ref_mse, total time: %.1fh)", total_elapsed / 3600)
    logger.info("=" * 80)
    results.sort(key=lambda r: r["ref_mse"])
    for r in results[:20]:
        logger.info(
            "%-40s ref_mse=%.4f q_gap=%.5f actor=%.4f critic=%.4f time=%.0fs",
            r["name"], r["ref_mse"], r["q_gap"],
            r["final_actor_loss"], r["final_critic_loss"], r["elapsed_sec"],
        )

    logger.info("\nBest: %s (ref_mse=%.4f)", results[0]["name"], results[0]["ref_mse"])
    logger.info("Config: %s", json.dumps(results[0]["config"], indent=2))


if __name__ == "__main__":
    main()
