#!/usr/bin/env python

import builtins
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin
from lerobot.utils.recording_annotations import normalize_episode_success_label
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config

VALUE_TRAIN_CONFIG_NAME = "value_train_config.json"


@dataclass
class ValueTargetsConfig:
    success_field: str = "episode_success"
    default_success: str = "failure"
    length_scale_quantile: float = 0.8
    c_fail_coef: float = 1.0
    target_field: str = "observation.value_target"

    def validate(self) -> None:
        normalized_default = normalize_episode_success_label(self.default_success)
        if normalized_default is None:
            raise ValueError("'targets.default_success' must be either 'success' or 'failure'.")
        self.default_success = normalized_default

        if not self.success_field:
            raise ValueError("'targets.success_field' must be non-empty.")
        if not 0.0 < self.length_scale_quantile <= 1.0:
            raise ValueError("'targets.length_scale_quantile' must be within (0, 1].")
        if self.c_fail_coef < 0:
            raise ValueError("'targets.c_fail_coef' must be non-negative.")
        if not self.target_field.startswith("observation."):
            raise ValueError(
                "'targets.target_field' must start with 'observation.' to survive processor conversion."
            )


@dataclass
class ValueTrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    value: PreTrainedConfig | None = field(default_factory=Pistar06Config)
    env: Any | None = None

    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000

    num_workers: int = 4
    batch_size: int = 64
    steps: int = 8_000
    log_freq: int = 200
    tolerance_s: float = 1e-4

    save_checkpoint: bool = True
    save_freq: int = 4_000

    use_value_training_preset: bool = True
    use_policy_training_preset: bool = field(init=False, default=True)

    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    targets: ValueTargetsConfig = field(default_factory=ValueTargetsConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    rename_map: dict[str, str] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def policy(self) -> PreTrainedConfig | None:
        return self.value

    def validate(self) -> None:
        value_path = parser.get_path_arg("value")
        if value_path:
            cli_overrides = parser.get_cli_overrides("value")
            self.value = PreTrainedConfig.from_pretrained(value_path, cli_overrides=cli_overrides)
            self.value.pretrained_path = Path(value_path)
        elif self.resume:
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming. Please specify path to {VALUE_TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. Resuming from hub is not supported."
                )
            value_dir = Path(config_path).parent
            if self.value is not None:
                self.value.pretrained_path = value_dir
            self.checkpoint_path = value_dir.parent

        if self.value is None:
            raise ValueError("Value is not configured. Please specify a value config with `--value.type`.")
        if self.value.type != "pistar06":
            raise ValueError(
                f"Unsupported value type '{self.value.type}'. "
                "Current lerobot-value-train supports only '--value.type=pistar06'."
            )

        self.targets.validate()

        if hasattr(self.value, "target_key"):
            self.value.target_key = self.targets.target_field

        if not self.job_name:
            self.job_name = f"{self.value.type}_value"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/value_train") / train_dir

        self.use_policy_training_preset = self.use_value_training_preset
        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when value presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.value.get_optimizer_preset()
            self.scheduler = self.value.get_scheduler_preset()

        if self.value.push_to_hub and not self.value.repo_id:
            raise ValueError(
                "'value.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["value"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / VALUE_TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["ValueTrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "ValueTrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if VALUE_TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, VALUE_TRAIN_CONFIG_NAME)
            else:
                print(f"{VALUE_TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=VALUE_TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{VALUE_TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
