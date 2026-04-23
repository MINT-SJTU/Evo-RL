#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config

PISTAR06_IMAGES_KEY = "observation.pistar06.images"
PISTAR06_IMAGE_MASK_KEY = "observation.pistar06.image_attention_mask"


def _pad_last_dim(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return functional.pad(vector, (0, new_dim - vector.shape[-1]))


@ProcessorStepRegistry.register(name="pistar06_prepare_task_prompt")
@dataclass
class Pistar06PrepareTaskPromptProcessorStep(ProcessorStep):
    task_key: str = "task"
    include_state_in_prompt: bool = True
    state_feature: str = OBS_STATE
    max_state_dim: int = 32
    state_discretization_bins: int = 256

    def get_config(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "include_state_in_prompt": self.include_state_in_prompt,
            "state_feature": self.state_feature,
            "max_state_dim": self.max_state_dim,
            "state_discretization_bins": self.state_discretization_bins,
        }

    @staticmethod
    def _clean_prompt(task: str) -> str:
        return str(task).strip().replace("_", " ").replace("\n", " ").strip()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.task_key not in complementary_data:
            raise KeyError(f"Missing task field '{self.task_key}' in complementary data.")
        tasks_raw = complementary_data[self.task_key]
        if isinstance(tasks_raw, str):
            tasks = [tasks_raw]
        elif isinstance(tasks_raw, Sequence) and all(isinstance(task, str) for task in tasks_raw):
            tasks = list(tasks_raw)
        else:
            raise TypeError(
                f"Expected task field '{self.task_key}' as sequence of strings, got {type(tasks_raw)}."
            )

        prompts: list[str] = []
        if self.include_state_in_prompt:
            if self.state_feature not in observation:
                raise KeyError(
                    f"Missing state feature '{self.state_feature}' while include_state_in_prompt=True."
                )
            state = observation[self.state_feature]
            if not isinstance(state, Tensor):
                state = torch.as_tensor(state)

            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.ndim != 2:
                raise ValueError(
                    f"Expected state tensor with shape [B, D], got {tuple(state.shape)} "
                    f"for feature '{self.state_feature}'."
                )

            state = state.detach().to(dtype=torch.float32, device="cpu")
            state = _pad_last_dim(state, self.max_state_dim)
            state_np = state.numpy()
            bins = np.linspace(-1.0, 1.0, self.state_discretization_bins + 1, dtype=np.float32)[:-1]
            discretized_state = np.digitize(state_np, bins=bins) - 1

            if discretized_state.shape[0] != len(tasks):
                raise ValueError(
                    f"Task count ({len(tasks)}) does not match state batch size ({discretized_state.shape[0]})."
                )

            for i, task in enumerate(tasks):
                cleaned_task = self._clean_prompt(task)
                state_str = " ".join(map(str, discretized_state[i].tolist()))
                prompts.append(f"Task: {cleaned_task}, State: {state_str}\nValue: ")
        else:
            prompts = [f"Task: {self._clean_prompt(task)}\nValue: " for task in tasks]

        complementary_data[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pistar06_prepare_images")
@dataclass
class Pistar06PrepareImagesProcessorStep(ProcessorStep):
    camera_features: list[str]

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_features": self.camera_features,
        }

    @staticmethod
    def _to_bchw(img_batch: Tensor) -> Tensor:
        if img_batch.ndim != 4:
            raise ValueError(f"Expected image batch rank 4, got shape {tuple(img_batch.shape)}.")

        if img_batch.shape[1] in {1, 3}:  # [B,C,H,W]
            return img_batch
        if img_batch.shape[-1] in {1, 3}:  # [B,H,W,C]
            return img_batch.permute(0, 3, 1, 2)
        raise ValueError(
            "Camera tensor must be channels-first or channels-last. "
            f"Got camera batch with shape={tuple(img_batch.shape)}."
        )

    def _process_camera_batch(self, img_batch: Tensor) -> Tensor:
        return self._to_bchw(img_batch).detach().to(dtype=torch.float32)

    def _prepare_images(self, observation: dict[str, Any]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.camera_features if key in observation]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All configured cameras are missing in the input batch. "
                f"expected={self.camera_features} batch_keys={list(observation.keys())}"
            )

        reference_img = self._process_camera_batch(torch.as_tensor(observation[present_img_keys[0]]))
        bsize = reference_img.shape[0]
        image_tensors: list[Tensor] = []
        image_masks: list[Tensor] = []

        for key in self.camera_features:
            if key in observation:
                img = self._process_camera_batch(torch.as_tensor(observation[key]))
                if img.shape[0] != bsize:
                    raise ValueError(
                        f"Mismatched batch size across cameras. Camera '{key}' has {img.shape[0]}, expected {bsize}."
                    )
                if img.shape[1:] != reference_img.shape[1:]:
                    raise ValueError(
                        "Camera tensors must share the same [C,H,W] shape before model preprocessing. "
                        f"Camera '{key}' has {tuple(img.shape[1:])}, expected {tuple(reference_img.shape[1:])}."
                    )
                image_tensors.append(img)
                image_masks.append(torch.ones(bsize, dtype=torch.bool))
            else:
                image_tensors.append(torch.zeros_like(reference_img))
                image_masks.append(torch.zeros(bsize, dtype=torch.bool))

        images = torch.stack(image_tensors, dim=1)
        masks = torch.stack(image_masks, dim=1)
        return images, masks

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})

        images, image_attention_mask = self._prepare_images(observation)
        observation[PISTAR06_IMAGES_KEY] = images.to(dtype=torch.float32)
        observation[PISTAR06_IMAGE_MASK_KEY] = image_attention_mask.to(dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pistar06_pre_post_processors(
    config: Pistar06Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,  # noqa: ARG001
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    camera_features = list(config.camera_features)
    if not camera_features:
        camera_features = [k for k in (config.input_features or {}) if k.startswith(OBS_IMAGES)]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys={config.state_feature},
        ),
        Pistar06PrepareTaskPromptProcessorStep(
            task_key=config.task_field,
            include_state_in_prompt=config.include_state_in_prompt,
            state_feature=config.state_feature,
            max_state_dim=config.max_state_dim,
            state_discretization_bins=config.state_discretization_bins,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_repo_id,
            task_key=config.task_field,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        ),
        Pistar06PrepareImagesProcessorStep(camera_features=camera_features),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
