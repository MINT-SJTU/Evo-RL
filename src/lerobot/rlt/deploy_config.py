from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeployConfig:
    """Configuration for deploying RLT policy on a real robot."""

    # Model paths
    vla_model_path: str = "lerobot/pi05_base"
    rl_token_checkpoint: str = ""
    ac_checkpoint: str = ""

    # Observation mapping
    camera_keys: list[str] = field(
        default_factory=lambda: ["left_wrist", "right_wrist", "right_front"]
    )
    proprio_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)

    # Phase control: "always_rl" uses RL actor always,
    # "always_vla" uses VLA reference always, "manual" uses PhaseController
    phase_mode: str = "always_rl"

    # Inference settings
    deterministic: bool = True
    device: str = "cuda"
    task_instruction: str = "pick up the object"
    token_pool_size: int = 64
    chunk_length: int = 10

    # Actor-critic architecture (must match checkpoint)
    actor_hidden_dim: int = 256
    actor_num_layers: int = 3
    actor_residual: bool = True
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    beta: float = 0.3

    def __post_init__(self):
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got '{self.phase_mode}'"
            )
