from __future__ import annotations

import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CAMERAS = ["left_wrist", "right_wrist", "right_front"]
DEFAULT_ACTION_DIM = 12
DEFAULT_PROPRIO_DIM = 12
DEFAULT_VLA_HORIZON = 50
DEFAULT_CHUNK_LENGTH = 10

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def load_training_config(config_path: str | None):
    from lerobot.rlt.config import RLTConfig

    config = RLTConfig.from_yaml(config_path) if config_path else RLTConfig()
    config.action_dim = DEFAULT_ACTION_DIM
    config.proprio_dim = DEFAULT_PROPRIO_DIM
    config.vla_horizon = DEFAULT_VLA_HORIZON
    config.chunk_length = DEFAULT_CHUNK_LENGTH
    config.cameras = list(DEFAULT_CAMERAS)
    return config


def build_pi05_policy(
    config,
    model_path: str,
    task_instruction: str,
    device: str,
    token_pool_size: int,
    dtype: str,
    rl_token_checkpoint: str | None = None,
    vla_cache_dir: str | None = None,
):
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.policy import RLTPolicy
    import torch

    vla = Pi05VLAAdapter(
        model_path=model_path,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=task_instruction,
        dtype=dtype,
        device=device,
        cache_dir=vla_cache_dir,
        token_pool_size=token_pool_size,
    )
    policy = RLTPolicy(config, vla).to(device)
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
    return policy
