from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RLTokenConfig:
    token_dim: int = 2048
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ff_dim: int | None = None  # defaults to 4 * token_dim if None
    num_rl_tokens: int = 1  # number of RL tokens (>1 reduces compression ratio)

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.token_dim


@dataclass
class ActorConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    fixed_std: float = 0.05
    lr: float = 3e-4
    ref_dropout_p: float = 0.5


@dataclass
class CriticConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    lr: float = 3e-4


@dataclass
class DemoAdaptConfig:
    steps: int = 5000
    batch_size: int = 32
    lr: float = 1e-4
    vla_ft_weight: float = 1.0
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    gamma: float = 0.99
    beta: float = 1.0
    tau: float = 0.005
    batch_size: int = 256
    utd_ratio: int = 5
    actor_update_interval: int = 2


@dataclass
class ReplayConfig:
    capacity: int = 200_000


@dataclass
class CollectorConfig:
    warmup_steps: int = 5000
    total_env_steps: int = 100_000
    chunk_subsample_stride: int = 2


@dataclass
class RLTConfig:
    seed: int = 0
    control_hz: int = 50
    action_dim: int = 14
    proprio_dim: int = 14
    vla_horizon: int = 50
    chunk_length: int = 10
    cameras: list[str] = field(default_factory=lambda: ["wrist_left", "wrist_right", "base"])

    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    demo_adaptation: DemoAdaptConfig = field(default_factory=DemoAdaptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLTConfig:
        """Load config from a YAML file, using defaults for missing fields."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        sub_configs = {}
        for key, subcls in [
            ("rl_token", RLTokenConfig),
            ("actor", ActorConfig),
            ("critic", CriticConfig),
            ("demo_adaptation", DemoAdaptConfig),
            ("training", TrainingConfig),
            ("replay", ReplayConfig),
            ("collector", CollectorConfig),
        ]:
            if key in raw:
                sub_configs[key] = subcls(**raw.pop(key))

        return cls(**raw, **sub_configs)

    @classmethod
    def default(cls) -> RLTConfig:
        """Load the shipped base.yaml defaults."""
        yaml_path = Path(__file__).parent / "configs" / "base.yaml"
        if yaml_path.exists():
            return cls.from_yaml(yaml_path)
        return cls()
