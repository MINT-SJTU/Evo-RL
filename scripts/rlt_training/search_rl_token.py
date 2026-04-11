#!/usr/bin/env python
"""RL Token architecture search on CP data.

Loads VLA once, then swaps RL Token configs and trains each for N steps.
Evaluates by reconstruction loss.

Key factors to search:
  - num_rl_tokens: 1, 2, 4, 8, 16
  - enc_layers / dec_layers: 1-5
  - ff_dim: 2048, 4096, 8192
  - nhead: 4, 8, 16
  - token_pool_size: 32, 64, 128
"""
from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging

logger = configure_logging(__name__)


@dataclass
class RLTokenExperiment:
    name: str
    num_rl_tokens: int = 4
    enc_layers: int = 3
    dec_layers: int = 3
    ff_dim: int = 4096
    nhead: int = 8
    token_pool_size: int = 64
    steps: int = 10000
    lr: float = 2e-4
    batch_size: int = 2


def build_experiments():
    exps = []

    # Baseline (current best config)
    exps.append(RLTokenExperiment(name="baseline"))

    # --- num_rl_tokens sweep ---
    for n in [1, 2, 4, 8, 16]:
        if n != 4:
            exps.append(RLTokenExperiment(name=f"tokens_{n}", num_rl_tokens=n))

    # --- Encoder layers ---
    for L in [1, 2, 4, 5]:
        exps.append(RLTokenExperiment(name=f"enc_{L}L", enc_layers=L))

    # --- Decoder layers ---
    for L in [1, 2, 4, 5]:
        exps.append(RLTokenExperiment(name=f"dec_{L}L", dec_layers=L))

    # --- Symmetric layer sweep ---
    for L in [1, 2, 4, 5, 6]:
        if L != 3:
            exps.append(RLTokenExperiment(name=f"sym_{L}L", enc_layers=L, dec_layers=L))

    # --- ff_dim ---
    for ff in [1024, 2048, 8192]:
        exps.append(RLTokenExperiment(name=f"ff_{ff}", ff_dim=ff))

    # --- nhead ---
    for nh in [4, 16]:
        exps.append(RLTokenExperiment(name=f"nhead_{nh}", nhead=nh))

    # --- token_pool_size (affects input token count) ---
    for ps in [16, 32, 128]:
        exps.append(RLTokenExperiment(name=f"pool_{ps}", token_pool_size=ps))

    # --- Asymmetric: deeper encoder, shallow decoder ---
    exps.append(RLTokenExperiment(name="enc5_dec1", enc_layers=5, dec_layers=1))
    exps.append(RLTokenExperiment(name="enc4_dec2", enc_layers=4, dec_layers=2))

    # --- Asymmetric: shallow encoder, deeper decoder ---
    exps.append(RLTokenExperiment(name="enc1_dec5", enc_layers=1, dec_layers=5))
    exps.append(RLTokenExperiment(name="enc2_dec4", enc_layers=2, dec_layers=4))

    # --- Combined: more tokens + fewer layers ---
    exps.append(RLTokenExperiment(name="tokens8_2L", num_rl_tokens=8, enc_layers=2, dec_layers=2))
    exps.append(RLTokenExperiment(name="tokens16_2L", num_rl_tokens=16, enc_layers=2, dec_layers=2))

    # --- Combined: fewer tokens + more layers ---
    exps.append(RLTokenExperiment(name="tokens2_5L", num_rl_tokens=2, enc_layers=5, dec_layers=5))
    exps.append(RLTokenExperiment(name="tokens1_5L", num_rl_tokens=1, enc_layers=5, dec_layers=5))

    # --- LR sweep ---
    for lr in [5e-5, 1e-4, 4e-4, 1e-3]:
        exps.append(RLTokenExperiment(name=f"lr_{lr}", lr=lr))

    return exps


def run_experiment(exp, vla, demo_loader_fn, device, results_file):
    """Run one RL Token experiment. VLA is shared across experiments."""
    from lerobot.rlt.rl_token import RLTokenModule

    logger.info("=" * 60)
    logger.info("Running: %s (tokens=%d, enc=%d, dec=%d, ff=%d, nh=%d, pool=%d)",
                exp.name, exp.num_rl_tokens, exp.enc_layers, exp.dec_layers,
                exp.ff_dim, exp.nhead, exp.token_pool_size)

    # Update VLA pool size if needed
    vla.token_pool_size = exp.token_pool_size

    # Build fresh RL Token module
    rl_token = RLTokenModule(
        token_dim=2048,
        nhead=exp.nhead,
        num_enc_layers=exp.enc_layers,
        num_dec_layers=exp.dec_layers,
        ff_dim=exp.ff_dim,
        num_rl_tokens=exp.num_rl_tokens,
    ).to(device)

    num_params = sum(p.numel() for p in rl_token.parameters())
    logger.info("RL Token params: %.2fM", num_params / 1e6)

    optimizer = torch.optim.Adam(rl_token.parameters(), lr=exp.lr)

    # Cosine LR with warmup
    import math
    warmup = min(500, exp.steps // 10)
    min_lr = 5e-6

    demo_iter = demo_loader_fn(exp.token_pool_size)
    rl_token.train()
    losses = []
    start_time = time.time()

    for step in range(1, exp.steps + 1):
        # LR schedule
        if step < warmup:
            lr = exp.lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(exp.steps - warmup, 1)
            lr = min_lr + 0.5 * (exp.lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        obs, expert_actions = next(demo_iter)
        with torch.no_grad():
            vla_out = vla.forward_vla(obs)

        loss = rl_token.reconstruction_loss(vla_out.final_tokens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rl_token.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 2000 == 0:
            avg = sum(losses[-500:]) / min(len(losses), 500)
            logger.info("[%s] Step %d/%d loss=%.4f avg500=%.4f lr=%.2e",
                        exp.name, step, exp.steps, loss.item(), avg, lr)

    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else float("inf")
    avg_last_500 = sum(losses[-500:]) / min(len(losses), 500)
    avg_last_100 = sum(losses[-100:]) / min(len(losses), 100)
    min_loss = min(losses) if losses else float("inf")

    result = {
        "name": exp.name,
        "config": asdict(exp),
        "final_loss": final_loss,
        "avg_last_500": avg_last_500,
        "avg_last_100": avg_last_100,
        "min_loss": min_loss,
        "num_params_M": num_params / 1e6,
        "elapsed_sec": elapsed,
        "steps_per_sec": exp.steps / elapsed,
    }

    logger.info("[%s] DONE loss=%.4f avg100=%.4f avg500=%.4f min=%.4f params=%.2fM time=%.1fs",
                exp.name, final_loss, avg_last_100, avg_last_500, min_loss,
                num_params / 1e6, elapsed)

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    # Cleanup
    del rl_token, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Elvinky/pi05_screw_271ep_sft_fp32")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/rl_token_search")
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    # Load VLA once (the expensive part)
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.demo_loader import make_demo_loader
    from lerobot.rlt.interfaces import Observation

    logger.info("Loading VLA from %s (dtype=%s)...", args.model_path, args.dtype)
    vla = Pi05VLAAdapter(
        model_path=args.model_path,
        actual_action_dim=12,
        actual_proprio_dim=12,
        task_instruction=args.task_instruction,
        dtype=args.dtype,
        device=args.device,
        token_pool_size=64,  # default, will be overridden per experiment
    )
    logger.info("VLA loaded")

    # Demo loader factory (recreates with different pool sizes if needed)
    def make_loader(pool_size):
        vla.token_pool_size = pool_size
        return make_demo_loader(
            dataset_path=args.dataset_path,
            batch_size=2,
            chunk_length=50,
            num_workers=0,
            device=args.device,
        )

    # Build and run experiments
    experiments = build_experiments()
    for exp in experiments:
        exp.steps = args.steps

    total_start = time.time()
    logger.info("Total experiments: %d", len(experiments))

    for i, exp in enumerate(experiments):
        if i < args.start_from:
            logger.info("Skipping %d: %s", i, exp.name)
            continue
        logger.info("--- Experiment %d/%d: %s ---", i + 1, len(experiments), exp.name)
        run_experiment(exp, vla, make_loader, args.device, results_file)

    # Final summary
    total_elapsed = time.time() - total_start
    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.sort(key=lambda r: r["avg_last_100"])

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY (sorted by avg_last_100, total: %.1fh)", total_elapsed / 3600)
    logger.info("=" * 80)
    for r in results[:20]:
        logger.info(
            "%-30s avg100=%.4f avg500=%.4f min=%.4f params=%.1fM time=%.0fs",
            r["name"], r["avg_last_100"], r["avg_last_500"],
            r["min_loss"], r["num_params_M"], r["elapsed_sec"],
        )
    logger.info("\nBest: %s (avg100=%.4f)", results[0]["name"], results[0]["avg_last_100"])
    logger.info("Config: %s", json.dumps(results[0]["config"], indent=2))


if __name__ == "__main__":
    main()
