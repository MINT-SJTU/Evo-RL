#!/usr/bin/env python
"""Convert legacy .pt checkpoints to safetensors format for HuggingFace Hub.

Input:  --rl-token-ckpt (demo adapt .pt) + --ac-ckpt (actor-critic .pt)
Output: model.safetensors + config.json in --output-dir

The output only contains RL head weights (rl_token encoder + actor).
VLA backbone (~7 GB) is excluded.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import draccus
import torch
from safetensors.torch import save_file

from lerobot.policies.rlt.configuration_rlt import RLTPretrainedConfig
from lerobot.rlt.utils import filter_encoder_only


def convert(
    rl_token_ckpt: str | None,
    ac_ckpt: str | None,
    output_dir: str,
    config_overrides: dict | None = None,
) -> Path:
    """Merge .pt checkpoints into a single safetensors file + config.json.

    Returns the output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()

    # --- RL Token encoder weights ---
    if rl_token_ckpt:
        ckpt = torch.load(rl_token_ckpt, map_location="cpu", weights_only=True)
        enc_keys, skipped = filter_encoder_only(ckpt["rl_token_state_dict"])
        for k, v in enc_keys.items():
            merged[f"rl_token.{k}"] = v
        step = ckpt.get("step", -1)
        print(f"RL Token encoder: {len(enc_keys)} keys (trained {step} steps, skipped {len(skipped)} decoder keys)")
        del ckpt

    # --- Actor-Critic checkpoint (actor only + optionally overwrite rl_token) ---
    if ac_ckpt:
        ckpt = torch.load(ac_ckpt, map_location="cpu", weights_only=True)
        for k, v in ckpt["actor_state_dict"].items():
            merged[f"actor.{k}"] = v
        print(f"Actor: {len(ckpt['actor_state_dict'])} keys")

        # AC checkpoint may contain a more recent rl_token (trained during offline RL)
        if "rl_token_state_dict" in ckpt:
            enc_keys, skipped = filter_encoder_only(ckpt["rl_token_state_dict"])
            for k, v in enc_keys.items():
                merged[f"rl_token.{k}"] = v
            print(f"RL Token encoder overwritten from AC checkpoint ({len(enc_keys)} keys)")
        del ckpt

    if not merged:
        raise ValueError("At least one of --rl-token-ckpt or --ac-ckpt must be provided")

    # --- Save safetensors ---
    safetensors_path = out / "model.safetensors"
    save_file(merged, str(safetensors_path))
    total_mb = sum(v.numel() * v.element_size() for v in merged.values()) / 1e6
    print(f"Saved {safetensors_path} ({len(merged)} tensors, {total_mb:.1f} MB)")

    # --- Save config.json ---
    cfg = RLTPretrainedConfig(**(config_overrides or {}))
    config_path = out / "config.json"
    with open(config_path, "w") as f, draccus.config_type("json"):
        draccus.dump(cfg, f, indent=4)
    print(f"Saved {config_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-token-ckpt", type=str, default=None,
                        help="Path to demo_adapt_checkpoint.pt")
    parser.add_argument("--ac-ckpt", type=str, default=None,
                        help="Path to rl_checkpoint_best.pt")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write model.safetensors + config.json")
    parser.add_argument("--vla-path", type=str, default="lerobot/pi05_base",
                        help="VLA pretrained path to store in config")
    parser.add_argument("--task", type=str, default="",
                        help="Task instruction to store in config")
    args = parser.parse_args()

    overrides = {
        "vla_pretrained_path": args.vla_path,
        "task_instruction": args.task,
    }
    convert(args.rl_token_ckpt, args.ac_ckpt, args.output_dir, config_overrides=overrides)


if __name__ == "__main__":
    main()
