#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.processor_pistar06 import PISTAR06_IMAGE_MASK_KEY, PISTAR06_IMAGES_KEY

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM
else:
    AutoConfig = None
    AutoImageProcessor = None
    AutoModel = None
    AutoModelForCausalLM = None


PISTAR06_SAVE_INFO = "pistar06_save_info.json"


@dataclass
class EpisodeTargetInfo:
    episode_index: int
    task_index: int
    length: int
    success: bool


def build_bin_centers(
    num_bins: int,
    bin_min: float,
    bin_max: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.linspace(bin_min, bin_max, num_bins, dtype=torch.float32, device=device)


def project_values_to_bins(values: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if bin_centers.ndim != 1:
        raise ValueError(f"'bin_centers' must be rank-1, got shape={tuple(bin_centers.shape)}.")
    if bin_centers.shape[0] < 2:
        raise ValueError("At least 2 bins are required.")

    values = values.clamp(min=bin_centers[0], max=bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = torch.floor(scaled).long()
    high = torch.clamp(low + 1, max=bin_centers.shape[0] - 1)
    high_weight = (scaled - low.float()).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    target = torch.zeros(values.shape[0], bin_centers.shape[0], device=values.device, dtype=torch.float32)
    target.scatter_add_(1, low.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def expected_value_from_logits(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = functional.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def compute_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
    *,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    if episode_indices.shape != frame_indices.shape:
        raise ValueError("episode_indices and frame_indices must have the same shape.")
    if c_fail_coef < 0:
        raise ValueError("'c_fail_coef' must be non-negative.")

    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)
    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        if ep_idx not in episode_info:
            raise KeyError(f"Missing episode metadata for episode_index={ep_idx}.")
        ep = episode_info[ep_idx]
        task_max = task_max_lengths.get(ep.task_index)
        if task_max is None:
            raise KeyError(f"Missing task max length for task_index={ep.task_index}.")
        if task_max <= 0:
            raise ValueError(f"Invalid task max length {task_max} for task_index={ep.task_index}.")

        remaining_steps = ep.length - int(frame_indices[i]) - 1
        c_fail = float(task_max) * c_fail_coef
        g = -float(remaining_steps)
        if not ep.success:
            g -= c_fail

        denom = float(task_max) + c_fail
        g_norm = g / denom
        targets[i] = np.clip(g_norm, clip_min, clip_max)

    return targets



def _resolve_load_dtype(dtype_name: str) -> torch.dtype:
    requested_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    if requested_dtype == torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return requested_dtype


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = True


def _extract_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Cannot infer hidden size for model type {type(model)}: missing `.config`.")

    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError(f"Cannot infer hidden size for model config type {type(config)}.")


def _extract_vision_feature_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Cannot infer vision feature size for model type {type(model)}: missing `.config`.")

    if hasattr(config, "projection_dim"):
        return int(config.projection_dim)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "projection_dim"):
        return int(config.vision_config.projection_dim)
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
        return int(config.vision_config.hidden_size)
    raise ValueError(f"Cannot infer vision feature size for model config type {type(config)}.")


def _validate_loading_info(repo_id: str, model_label: str, loading_info: dict[str, list] | None) -> None:
    if loading_info is None:
        return
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    if not missing and not unexpected and not mismatched:
        return
    raise RuntimeError(
        f"Pretrained weights for {model_label} from '{repo_id}' did not load cleanly: "
        f"missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}. "
        "This usually indicates a model class/checkpoint mismatch."
    )


def _resolve_image_size(image_processor: Any) -> tuple[int, int]:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return 384, 384


def _resolve_norm_stats(
    image_processor: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean_raw = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    std_raw = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
    if len(mean_raw) != 3 or len(std_raw) != 3:
        raise ValueError(f"Expected RGB normalization stats of len=3, got mean={mean_raw} std={std_raw}.")
    mean = (float(mean_raw[0]), float(mean_raw[1]), float(mean_raw[2]))
    std = (float(std_raw[0]), float(std_raw[1]), float(std_raw[2]))
    if any(v <= 0 for v in std):
        raise ValueError(f"Invalid image std values: {std}.")
    return mean, std


def _load_language_model(
    repo_id: str,
    revision: str | None,
    dtype: torch.dtype,
) -> nn.Module:
    if AutoConfig is None or AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError("transformers is not installed. Install with `pip install 'lerobot[pi0]'`.")

    model_config = AutoConfig.from_pretrained(repo_id, revision=revision)
    architectures = getattr(model_config, "architectures", None) or []
    prefer_causal_lm = any(isinstance(arch, str) and arch.endswith("ForCausalLM") for arch in architectures)

    if prefer_causal_lm:
        lm_with_head, loading_info = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        _validate_loading_info(repo_id, "language_model(causal_lm)", loading_info)
        if not hasattr(lm_with_head, "model"):
            raise RuntimeError(
                f"AutoModelForCausalLM loaded from '{repo_id}' does not expose `.model` text backbone."
            )
        return lm_with_head.model

    language_model, loading_info = AutoModel.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        output_loading_info=True,
    )
    _validate_loading_info(repo_id, "language_model(auto_model)", loading_info)
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"AutoModel loaded from '{repo_id}' returned unexpected type: {type(language_model)}."
        )
    return language_model


class Pistar06Model(nn.Module):
    def __init__(self, cfg: Pistar06Config):
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError("transformers is not installed. Install with `pip install 'lerobot[pi0]'`.")

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            torch_dtype=self.model_dtype,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            use_fast=True,
        )
        image_height, image_width = _resolve_image_size(image_processor)
        image_mean, image_std = _resolve_norm_stats(image_processor)
        self.image_resolution = (image_height, image_width)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        vision_feature_size = _extract_vision_feature_size(self.vision_encoder)
        language_hidden_size = _extract_hidden_size(self.language_model)

        self.image_projector = nn.Sequential(
            nn.Linear(vision_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_hidden_dim * 2)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_bins),
        )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)

        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)

    def _encode_images(self, flat_images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)

        vision_outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported vision encoder output. Expected pooler_output or last_hidden_state.")

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Language model output does not contain `last_hidden_state`.")

        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def _preprocess_images(self, images: Tensor, image_attention_mask: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B,N,C,H,W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(
                f"'image_attention_mask' must have shape [B,N], got {tuple(image_attention_mask.shape)}."
            )

        bsize, num_cameras = images.shape[:2]
        if image_attention_mask.shape[0] != bsize or image_attention_mask.shape[1] != num_cameras:
            raise ValueError("Batch shape mismatch between images and image_attention_mask.")

        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32) / 255.0
        else:
            images = images.to(dtype=torch.float32)
            if bool(torch.max(images) > 1.0) or bool(torch.min(images) < 0.0):
                images = (images / 255.0).clamp(0.0, 1.0)

        flat_images = images.view(bsize * num_cameras, *images.shape[2:])
        if flat_images.shape[-2:] != self.image_resolution:
            flat_images = functional.interpolate(
                flat_images,
                size=self.image_resolution,
                mode="bilinear",
                align_corners=False,
            )

        mean = self.image_mean.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(device=flat_images.device, dtype=flat_images.dtype).view(1, 3, 1, 1)
        flat_images = (flat_images - mean) / std
        flat_images = flat_images.view(
            bsize,
            num_cameras,
            flat_images.shape[1],
            flat_images.shape[2],
            flat_images.shape[3],
        )

        camera_mask = image_attention_mask.to(device=flat_images.device, dtype=flat_images.dtype).view(
            bsize, num_cameras, 1, 1, 1
        )
        flat_images = flat_images * camera_mask
        return flat_images

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"'input_ids' must have shape [B, T], got {tuple(input_ids.shape)}.")
        if attention_mask.ndim != 2:
            raise ValueError(f"'attention_mask' must have shape [B, T], got {tuple(attention_mask.shape)}.")
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B, N, C, H, W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(
                f"'image_attention_mask' must have shape [B, N], got {tuple(image_attention_mask.shape)}."
            )

        bsize = input_ids.shape[0]
        if attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch between input_ids and attention_mask.")
        if images.shape[0] != bsize or image_attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch between language and image inputs.")
        if images.shape[1] == 0:
            raise ValueError("At least one camera is required for Pistar06Model.")

        image_attention_mask = image_attention_mask.to(dtype=torch.bool, device=images.device)
        if not torch.all(image_attention_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera input.")
        language_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if not torch.all(language_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token.")

        processed_images = self._preprocess_images(images, image_attention_mask)
        num_cameras = processed_images.shape[1]
        flat_images = processed_images.reshape(bsize * num_cameras, *processed_images.shape[2:])
        flat_images = flat_images.to(dtype=self.model_dtype)

        image_context = torch.no_grad() if self.cfg.freeze_vision_encoder else nullcontext()
        with image_context:
            image_features = self._encode_images(flat_images)

        language_context = torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        with language_context:
            language_features = self._encode_language(
                input_ids=input_ids, attention_mask=language_mask.long()
            )

        feature_dtype = torch.float32
        image_features = image_features.to(dtype=feature_dtype)
        language_features = language_features.to(dtype=feature_dtype)

        image_tokens = self.image_projector(image_features).view(bsize, num_cameras, -1)
        camera_token_mask = image_attention_mask.unsqueeze(-1).to(dtype=image_tokens.dtype)
        image_tokens = image_tokens * camera_token_mask

        camera_denominator = (
            image_attention_mask.sum(dim=1, keepdim=True).to(dtype=image_tokens.dtype).clamp_min(1.0)
        )
        image_pooled = image_tokens.sum(dim=1) / camera_denominator
        language_token = self.language_projector(language_features)

        joint_features = torch.cat([image_pooled, language_token], dim=-1)
        return self.value_head(self.final_norm(joint_features))


class Pistar06Policy(PreTrainedPolicy):
    config_class = Pistar06Config
    name = "pistar06"

    def __init__(
        self,
        config: Pistar06Config,
        dataset_meta=None,
        **kwargs: Any,
    ):
        del dataset_meta, kwargs
        super().__init__(config)
        self.config = config
        self.model = Pistar06Model(config)

        self.register_buffer(
            "bin_centers",
            build_bin_centers(config.num_bins, config.bin_min, config.bin_max),
            persistent=False,
        )

    def _frozen_checkpoint_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        if self.config.freeze_vision_encoder:
            prefixes.append("model.vision_encoder.")
        if self.config.freeze_language_model:
            prefixes.append("model.language_model.")
        return prefixes

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)

        model_to_save = self.module if hasattr(self, "module") else self
        state_dict = model_to_save.state_dict()
        excluded_prefixes = self._frozen_checkpoint_prefixes()

        if excluded_prefixes:
            state_dict = {
                key: tensor
                for key, tensor in state_dict.items()
                if not any(key.startswith(prefix) for prefix in excluded_prefixes)
            }

        save_file(state_dict, str(save_directory / SAFETENSORS_SINGLE_FILE))
        save_info = {
            "format_version": 1,
            "weights_mode": "partial" if excluded_prefixes else "full",
            "freeze_vision_encoder": bool(self.config.freeze_vision_encoder),
            "freeze_language_model": bool(self.config.freeze_language_model),
            "excluded_prefixes": excluded_prefixes,
            "saved_tensor_count": len(state_dict),
        }
        with open(save_directory / PISTAR06_SAVE_INFO, "w", encoding="utf-8") as f:
            json.dump(save_info, f, indent=2, sort_keys=True)

    @classmethod
    def _load_save_info(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> dict[str, Any] | None:
        del cls

        model_id = str(pretrained_name_or_path)
        save_info_path: Path | None = None
        if os.path.isdir(model_id):
            candidate = Path(model_id) / PISTAR06_SAVE_INFO
            if candidate.is_file():
                save_info_path = candidate
        else:
            try:
                resolved = hf_hub_download(
                    repo_id=model_id,
                    filename=PISTAR06_SAVE_INFO,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except (HfHubHTTPError, FileNotFoundError):
                resolved = None
            if resolved is not None:
                save_info_path = Path(resolved)

        if save_info_path is None:
            return None

        try:
            with open(save_info_path, encoding="utf-8") as f:
                parsed = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: Pistar06Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Pistar06Policy:
        save_info = cls._load_save_info(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        effective_strict = strict
        if save_info is not None and save_info.get("weights_mode") == "partial":
            if strict:
                logging.info(
                    "Detected partial Pistar06 checkpoint at '%s'; "
                    "forcing strict=False for automatic fallback.",
                    pretrained_name_or_path,
                )
            effective_strict = False

        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=effective_strict,
            **kwargs,
        )

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        return

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("Pistar06Policy is a value model and does not support action prediction.")

    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise RuntimeError("Pistar06Policy is a value model and does not support action selection.")

    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        images = batch[PISTAR06_IMAGES_KEY]
        image_attention_mask = batch[PISTAR06_IMAGE_MASK_KEY]

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )
        bin_centers = self.bin_centers.to(device=logits.device)
        return expected_value_from_logits(logits, bin_centers)

    def build_training_raw_batch_hook(self, dataset, targets_cfg):
        raw_frames = dataset.hf_dataset.with_format(None)
        frame_count = len(raw_frames)
        if frame_count == 0:
            raise ValueError("Dataset has no frames.")

        episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes = episodes_ds[:]
        n_episodes = len(episodes_ds)
        has_success = targets_cfg.success_field in episodes_ds.column_names

        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}
        for i in range(n_episodes):
            ep_idx = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks
            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved_success = resolve_episode_success_label(
                explicit_success,
                default_label=targets_cfg.default_success,
                require_label=True,
            )
            ep_success = resolved_success == EPISODE_SUCCESS

            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx,
                task_index=task_index,
                length=ep_length,
                success=ep_success,
            )
            task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=self.config.bin_min,
            clip_max=self.config.bin_max,
        )

        max_index = int(np.max(absolute_indices))
        value_target_lookup = np.zeros(max_index + 1, dtype=np.float32)
        value_target_lookup[absolute_indices] = value_targets.astype(np.float32, copy=False)

        target_key = targets_cfg.target_field

        def value_target_hook(batch: dict[str, Any], step: int) -> dict[str, Any]:
            del step
            batch_indices = batch.get("index")
            if batch_indices is None:
                raise KeyError("Missing 'index' in batch while building value targets.")
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)

            batch_indices_np = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            target_values = torch.from_numpy(value_target_lookup[batch_indices_np]).to(dtype=torch.float32)
            batch[target_key] = target_values
            return batch

        return value_target_hook

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.target_key not in batch:
            raise KeyError(
                f"Missing target key '{self.config.target_key}' in batch. "
                "Make sure lerobot-value-train target hook is enabled."
            )

        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        images = batch[PISTAR06_IMAGES_KEY]
        image_attention_mask = batch[PISTAR06_IMAGE_MASK_KEY]

        device = next(self.model.parameters()).device
        value_target = batch[self.config.target_key]
        if not isinstance(value_target, Tensor):
            value_target = torch.as_tensor(value_target)
        value_target = value_target.to(device=device, dtype=torch.float32, non_blocking=True)
        if value_target.ndim == 2 and value_target.shape[-1] == 1:
            value_target = value_target.squeeze(-1)
        if value_target.ndim != 1:
            raise ValueError(
                f"Value target must be rank-1 or [B,1], got shape={tuple(value_target.shape)} "
                f"for key '{self.config.target_key}'."
            )

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )

        bin_centers = self.bin_centers.to(device=device)
        soft_target = project_values_to_bins(value_target, bin_centers)
        log_probs = functional.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

        sample_weight = None
        if self.config.loss_weight_key in batch:
            sample_weight = batch[self.config.loss_weight_key]
            if not isinstance(sample_weight, Tensor):
                sample_weight = torch.as_tensor(sample_weight)
            sample_weight = sample_weight.to(device=device, dtype=torch.float32, non_blocking=True)
            if sample_weight.ndim == 2 and sample_weight.shape[-1] == 1:
                sample_weight = sample_weight.squeeze(-1)
            if sample_weight.ndim != 1:
                raise ValueError(
                    f"Loss weight must be rank-1 or [B,1], got shape={tuple(sample_weight.shape)} "
                    f"for key '{self.config.loss_weight_key}'."
                )
            if sample_weight.shape[0] != per_sample_loss.shape[0]:
                raise ValueError(
                    f"Loss weight batch size mismatch: expected {per_sample_loss.shape[0]}, "
                    f"got {sample_weight.shape[0]} for key '{self.config.loss_weight_key}'."
                )
            per_sample_loss = per_sample_loss * sample_weight

        pred_value = expected_value_from_logits(logits, bin_centers)
        value_mae = (pred_value - value_target).abs().mean()

        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()

        loss_dict = {
            "loss": float(loss.mean().detach().item())
            if reduction == "none"
            else float(loss.detach().item()),
            "value_mae": float(value_mae.detach().item()),
        }
        if sample_weight is not None:
            loss_dict["loss_weight_mean"] = float(sample_weight.mean().detach().item())
            loss_dict["loss_weight_min"] = float(sample_weight.min().detach().item())
            loss_dict["loss_weight_max"] = float(sample_weight.max().detach().item())
        return loss, loss_dict
