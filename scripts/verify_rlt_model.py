#!/usr/bin/env python
"""Verify RLT model outputs: checkpoint metadata, action distributions, ref_mse.

Two modes:
  --cached-only: Load cached transitions + AC checkpoint, run actor forward, compute ref_mse.
                 Fast (no VLA needed), validates actor output quality.
  Full pipeline: Load VLA + RL token + Actor, run on real observations from dataset.
                 Slow (needs GPU for pi0.5), validates end-to-end consistency.

Usage:
  # Cached mode (fast, no VLA):
  python scripts/verify_rlt_model.py \
    --cache-dir outputs/cache_newdata_stride2 \
    --ac-ckpt outputs/exp_s2_b0.3/rl_checkpoint.pt

  # Full pipeline mode:
  python scripts/verify_rlt_model.py \
    --vla-model Elvinky/pi05_screw_271ep_sft_fp32 \
    --rl-token-ckpt outputs/rlt_demo_adapt_271ep_sft_fp32/demo_adapt_checkpoint.pt \
    --ac-ckpt outputs/exp_s2_b0.3/rl_checkpoint.pt \
    --dataset-path /home/coder/share/dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify RLT model outputs.")
    p.add_argument("--ac-ckpt", required=True, help="Offline RL checkpoint (.pt)")
    p.add_argument("--rl-token-ckpt", default=None, help="Demo adaptation checkpoint (.pt)")
    p.add_argument("--cache-dir", default=None, help="Cached transitions dir (skip VLA)")
    p.add_argument("--vla-model", default=None, help="VLA model path (for full pipeline)")
    p.add_argument("--dataset-path", default=None, help="LeRobot dataset path")
    p.add_argument("--config", default=None, help="RLT YAML config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=512, help="Samples for ref_mse check")
    p.add_argument("--token-pool-size", type=int, default=64)
    p.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve.")
    return p.parse_args()


def inspect_checkpoint(path: str, label: str) -> dict:
    """Print checkpoint metadata and return the loaded dict."""
    logger.info("=== Inspecting %s: %s ===", label, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    logger.info("Keys: %s", list(ckpt.keys()))

    if "step" in ckpt:
        logger.info("  step: %s", ckpt["step"])
    if "losses" in ckpt:
        losses = ckpt["losses"]
        logger.info("  final loss: %.6f (from %d steps)", losses[-1], len(losses))
    if "config" in ckpt:
        logger.info("  config: %s", ckpt["config"])
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        for k, v in m.items():
            if not isinstance(v, list):
                logger.info("  metrics.%s = %s", k, v)

    # Print state_dict shapes for key components
    for sd_key in ["actor_state_dict", "rl_token_state_dict", "critic_state_dict"]:
        if sd_key in ckpt:
            sd = ckpt[sd_key]
            num_params = sum(v.numel() for v in sd.values())
            logger.info("  %s: %d tensors, %.2fM params", sd_key, len(sd), num_params / 1e6)
            # Print first and last layer shapes
            keys = list(sd.keys())
            if keys:
                logger.info("    first: %s %s", keys[0], tuple(sd[keys[0]].shape))
                logger.info("    last:  %s %s", keys[-1], tuple(sd[keys[-1]].shape))

    return ckpt


def tensor_stats(t: torch.Tensor, name: str) -> None:
    """Print min/max/mean/std for a tensor."""
    logger.info("  %s: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                name, tuple(t.shape), t.min().item(), t.max().item(),
                t.mean().item(), t.std().item())


def verify_cached_mode(args: argparse.Namespace) -> None:
    """Fast verification using cached transitions (no VLA needed)."""
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.actor import ChunkActor
    from lerobot.rlt.offline_dataset import load_transition_cache
    from lerobot.rlt.utils import flatten_chunk, infer_actor_architecture, unflatten_chunk

    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.chunk_length = 10

    # Load AC checkpoint
    ac_ckpt = inspect_checkpoint(args.ac_ckpt, "AC checkpoint")

    ac_config = ac_ckpt.get("config", {})
    actor_kwargs = infer_actor_architecture(
        ac_ckpt["actor_state_dict"],
        default_activation=config.actor.activation,
        default_fixed_std=config.actor.fixed_std,
        default_ref_dropout_p=config.actor.ref_dropout_p,
    )
    actor_kwargs["hidden_dim"] = ac_config.get("actor_hidden", actor_kwargs["hidden_dim"])
    actor_kwargs["num_layers"] = ac_config.get("actor_layers", actor_kwargs["num_layers"])
    actor_kwargs["activation"] = ac_config.get("activation", actor_kwargs["activation"])
    actor_kwargs["residual"] = ac_config.get("architecture", "") == "residual_mlp" or actor_kwargs["residual"]
    if not ac_config:
        logger.warning("AC checkpoint has no config metadata; activation/fixed_std fall back to config defaults")
    beta = ac_config.get("beta", config.training.beta)

    logger.info("=== Actor config: hidden=%d, layers=%d, activation=%s, residual=%s, beta=%.2f ===",
                actor_kwargs["hidden_dim"], actor_kwargs["num_layers"],
                actor_kwargs["activation"], actor_kwargs["residual"], beta)

    token_dim = config.rl_token.token_dim  # 2048
    chunk_dim = config.chunk_length * config.action_dim  # 10 * 12 = 120
    state_dim = token_dim + config.proprio_dim  # 2048 + 12 = 2060

    actor = ChunkActor(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=actor_kwargs["hidden_dim"],
        num_layers=actor_kwargs["num_layers"],
        fixed_std=actor_kwargs["fixed_std"],
        ref_dropout_p=actor_kwargs["ref_dropout_p"],
        activation=actor_kwargs["activation"],
        layer_norm=actor_kwargs["layer_norm"],
        residual=actor_kwargs["residual"],
    )
    actor.load_state_dict(ac_ckpt["actor_state_dict"])
    actor.eval()
    actor.to(args.device)

    # Also inspect RL token checkpoint if provided
    if args.rl_token_ckpt:
        inspect_checkpoint(args.rl_token_ckpt, "RL Token checkpoint")

    # Load cached transitions
    logger.info("=== Loading cached transitions from %s ===", args.cache_dir)
    buf = load_transition_cache(args.cache_dir, "val", capacity=200_000)
    logger.info("Buffer size: %d transitions", len(buf))

    # Sample a batch
    n = min(args.num_samples, len(buf))
    batch = buf.sample(n)

    state_vec = batch["state_vec"].to(args.device)
    ref_chunk = batch["ref_chunk"].to(args.device)
    exec_chunk = batch["exec_chunk"].to(args.device)

    logger.info("=== Input distributions ===")
    tensor_stats(state_vec, "state_vec")
    tensor_stats(state_vec[:, :token_dim], "z_rl (state[:, :2048])")
    tensor_stats(state_vec[:, token_dim:], "proprio (state[:, 2048:])")
    tensor_stats(ref_chunk, "ref_chunk (VLA reference)")
    tensor_stats(exec_chunk, "exec_chunk (expert actions)")

    # Run actor forward
    ref_flat = flatten_chunk(ref_chunk)
    with torch.no_grad():
        mu, std = actor(state_vec, ref_flat, training=False)
    actor_chunk = unflatten_chunk(mu, config.chunk_length)

    logger.info("=== Actor output distributions ===")
    tensor_stats(actor_chunk, "actor_chunk (mu)")
    tensor_stats(std, "actor_std")

    # Compute ref_mse
    ref_mse = ((actor_chunk - ref_chunk) ** 2).mean().item()
    expert_mse = ((actor_chunk - exec_chunk) ** 2).mean().item()
    ref_vs_expert = ((ref_chunk - exec_chunk) ** 2).mean().item()

    logger.info("=== MSE metrics ===")
    logger.info("  ref_mse (actor vs VLA ref):    %.6f", ref_mse)
    logger.info("  expert_mse (actor vs expert):  %.6f", expert_mse)
    logger.info("  ref_vs_expert (VLA vs expert): %.6f", ref_vs_expert)

    # Per-dimension analysis
    per_dim_ref_mse = ((actor_chunk - ref_chunk) ** 2).mean(dim=(0, 1))
    logger.info("  per-dim ref_mse: %s", [f"{x:.4f}" for x in per_dim_ref_mse.tolist()])

    # Check if actor output is in reasonable range
    out_of_range = ((actor_chunk.abs() > 1.5).float().mean().item()) * 100
    logger.info("  actor outputs > |1.5|: %.1f%%", out_of_range)

    # Check correlation between actor and ref
    actor_flat = actor_chunk.reshape(-1)
    ref_flat_vals = ref_chunk.reshape(-1)
    correlation = torch.corrcoef(torch.stack([actor_flat, ref_flat_vals]))[0, 1].item()
    logger.info("  actor-ref correlation: %.4f", correlation)

    # Sample a few individual predictions for eyeballing
    logger.info("=== Sample predictions (first 3 transitions, step 0) ===")
    for i in range(min(3, n)):
        logger.info("  [%d] ref:   %s", i, [f"{x:.3f}" for x in ref_chunk[i, 0].tolist()])
        logger.info("  [%d] actor: %s", i, [f"{x:.3f}" for x in actor_chunk[i, 0].tolist()])
        logger.info("  [%d] expert:%s", i, [f"{x:.3f}" for x in exec_chunk[i, 0].tolist()])
        logger.info("  [%d] MSE: %.6f", i, ((actor_chunk[i] - ref_chunk[i]) ** 2).mean().item())

    # Verdict
    logger.info("=== VERDICT ===")
    if ref_mse < 0.05:
        logger.info("PASS: ref_mse=%.6f is in expected range (<0.05)", ref_mse)
    elif ref_mse < 0.1:
        logger.info("WARN: ref_mse=%.6f is elevated but maybe OK", ref_mse)
    else:
        logger.info("FAIL: ref_mse=%.6f is too high — actor is not tracking VLA reference", ref_mse)
        if state_vec[:, :token_dim].abs().max() > 100:
            logger.info("  HINT: z_rl values are very large — possible scale mismatch or VLA model mismatch")
        if ref_chunk.abs().max() > 2:
            logger.info("  HINT: ref_chunk values exceed [-1,1] — data may not be normalized")


def verify_full_pipeline(args: argparse.Namespace) -> None:
    """Full pipeline verification: load VLA + RL token + Actor, run on real observations."""
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.policy import RLTPolicy
    from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
    from lerobot.rlt.interfaces import Observation
    from lerobot.rlt.utils import flatten_chunk, infer_actor_architecture, unflatten_chunk
    from torch.utils.data import DataLoader, Subset

    config = RLTConfig.from_yaml(args.config) if args.config else RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.cameras = ["left_wrist", "right_wrist", "right_front"]

    # Inspect checkpoints
    ac_ckpt = inspect_checkpoint(args.ac_ckpt, "AC checkpoint")
    rl_ckpt = inspect_checkpoint(args.rl_token_ckpt, "RL Token checkpoint")

    ac_config = ac_ckpt.get("config", {})
    actor_kwargs = infer_actor_architecture(
        ac_ckpt["actor_state_dict"],
        default_activation=config.actor.activation,
        default_fixed_std=config.actor.fixed_std,
        default_ref_dropout_p=config.actor.ref_dropout_p,
    )
    actor_kwargs["hidden_dim"] = ac_config.get("actor_hidden", actor_kwargs["hidden_dim"])
    actor_kwargs["num_layers"] = ac_config.get("actor_layers", actor_kwargs["num_layers"])
    actor_kwargs["activation"] = ac_config.get("activation", actor_kwargs["activation"])
    actor_kwargs["residual"] = ac_config.get("architecture", "") == "residual_mlp" or actor_kwargs["residual"]
    if not ac_config:
        logger.warning("AC checkpoint has no config metadata; activation/fixed_std fall back to config defaults")
    config.actor.hidden_dim = actor_kwargs["hidden_dim"]
    config.actor.num_layers = actor_kwargs["num_layers"]
    config.actor.activation = actor_kwargs["activation"]
    config.actor.layer_norm = actor_kwargs["layer_norm"]
    config.actor.residual = actor_kwargs["residual"]
    config.actor.fixed_std = actor_kwargs["fixed_std"]
    config.actor.ref_dropout_p = actor_kwargs["ref_dropout_p"]

    # Load VLA
    logger.info("=== Loading VLA from %s ===", args.vla_model)
    vla = Pi05VLAAdapter(
        model_path=args.vla_model,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=args.task_instruction,
        dtype="bfloat16",
        device=args.device,
        token_pool_size=args.token_pool_size,
    )
    logger.info("VLA loaded. token_dim=%d, action_dim=%d", vla.token_dim, vla.action_dim)

    # Build policy
    policy = RLTPolicy(config, vla).to(args.device)
    policy.rl_token.load_state_dict(rl_ckpt["rl_token_state_dict"], strict=False)
    policy.actor.load_state_dict(ac_ckpt["actor_state_dict"])
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()

    # Load dataset
    dataset = RLTDemoDataset(
        dataset_path=args.dataset_path,
        chunk_length=config.vla_horizon,
        normalize_actions=True,
    )
    logger.info("Dataset loaded: %d samples", len(dataset))

    # Pick a few frames from the first episode
    indices = list(range(0, min(args.num_samples, len(dataset)), max(1, len(dataset) // args.num_samples)))
    loader = DataLoader(
        Subset(dataset, indices[:args.num_samples]),
        batch_size=4, shuffle=False,
        collate_fn=rlt_demo_collate, num_workers=0,
    )

    all_ref_chunks = []
    all_actor_chunks = []
    all_expert_chunks = []

    logger.info("=== Running full pipeline on %d frames ===", len(indices[:args.num_samples]))
    for batch_idx, (obs, expert_actions) in enumerate(loader):
        obs = Observation(
            images={k: v.to(args.device) for k, v in obs.images.items()},
            proprio=obs.proprio.to(args.device),
        )

        with torch.no_grad():
            action_chunk, mu_chunk, state_vec, ref_chunk = policy.select_action(obs, deterministic=True)

        # Subsample expert actions to match chunk_length
        from lerobot.rlt.utils import subsample_indices
        sub_idx = subsample_indices(expert_actions.shape[1], config.chunk_length)
        expert_chunk = expert_actions[:, sub_idx, :].to(args.device)

        all_ref_chunks.append(ref_chunk.cpu())
        all_actor_chunks.append(mu_chunk.cpu())
        all_expert_chunks.append(expert_chunk.cpu())

        if batch_idx == 0:
            logger.info("=== First batch details ===")
            tensor_stats(state_vec, "state_vec")
            tensor_stats(state_vec[:, :config.rl_token.token_dim], "z_rl")
            tensor_stats(state_vec[:, config.rl_token.token_dim:], "proprio")
            tensor_stats(ref_chunk, "ref_chunk (VLA)")
            tensor_stats(mu_chunk, "actor_chunk (mu)")
            tensor_stats(expert_chunk, "expert_chunk")

        if batch_idx >= 3:
            logger.info("(processed %d batches, stopping early for speed)", batch_idx + 1)
            break

    ref_all = torch.cat(all_ref_chunks)
    actor_all = torch.cat(all_actor_chunks)
    expert_all = torch.cat(all_expert_chunks)

    ref_mse = ((actor_all - ref_all) ** 2).mean().item()
    expert_mse = ((actor_all - expert_all) ** 2).mean().item()
    ref_vs_expert = ((ref_all - expert_all) ** 2).mean().item()

    logger.info("=== Full pipeline MSE (%d frames) ===", len(ref_all))
    logger.info("  ref_mse (actor vs VLA ref):    %.6f", ref_mse)
    logger.info("  expert_mse (actor vs expert):  %.6f", expert_mse)
    logger.info("  ref_vs_expert (VLA vs expert): %.6f", ref_vs_expert)

    logger.info("=== VERDICT ===")
    if ref_mse < 0.05:
        logger.info("PASS: ref_mse=%.6f — actor tracks VLA reference well", ref_mse)
    elif ref_mse < 0.1:
        logger.info("WARN: ref_mse=%.6f — elevated, check training", ref_mse)
    else:
        logger.info("FAIL: ref_mse=%.6f — actor diverged from VLA reference", ref_mse)


def main() -> None:
    args = parse_args()

    if args.cache_dir:
        verify_cached_mode(args)
    elif args.vla_model and args.dataset_path and args.rl_token_ckpt:
        verify_full_pipeline(args)
    else:
        logger.error("Provide either --cache-dir (cached mode) or --vla-model + --dataset-path + --rl-token-ckpt (full pipeline)")
        sys.exit(1)


if __name__ == "__main__":
    main()
