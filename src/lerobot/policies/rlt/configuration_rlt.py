from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# Default SO101 bilateral robot keys (6 DOF x 2 arms = 12)
_DEFAULT_PROPRIO_KEYS: list[str] = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]

_DEFAULT_CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]

_DEFAULT_ACTION_KEYS: list[str] = list(_DEFAULT_PROPRIO_KEYS)  # same as proprio


@PreTrainedConfig.register_subclass("rlt")
@dataclass
class RLTPretrainedConfig(PreTrainedConfig):
    """Configuration for RLT (RL Token) policy.

    RLT wraps a frozen VLA backbone (pi0.5) with a lightweight RL head
    (RL Token encoder + ChunkActor) trained via offline RL.
    """

    # --- VLA backbone ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    task_instruction: str = ""
    token_pool_size: int = 64

    # --- RL Token encoder architecture ---
    rl_token_dim: int = 2048
    rl_token_nhead: int = 8
    rl_token_enc_layers: int = 3
    rl_token_dec_layers: int = 3
    rl_token_ff_dim: int = 4096
    rl_token_num_rl_tokens: int = 4

    # --- Actor architecture ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 3
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = True

    # --- Deployment ---
    chunk_length: int = 10
    action_dim: int = 12
    proprio_dim: int = 12
    phase_mode: str = "always_rl"
    deterministic: bool = True

    # --- Observation mapping ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))
    proprio_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_PROPRIO_KEYS))
    action_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_ACTION_KEYS))

    # --- Normalization (RLT does not normalize; VLA handles its own) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got '{self.phase_mode}'"
            )

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.proprio_dim,),
            )
        for i, cam_key in enumerate(self.camera_keys):
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None
