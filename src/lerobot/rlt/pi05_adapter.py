from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from lerobot.rlt.interfaces import Observation, VLAOutput
from lerobot.rlt.vla_adapter import VLAAdapter


class Pi05VLAAdapter(VLAAdapter):
    def __init__(
        self,
        model_path: str = "lerobot/pi05_base",
        actual_action_dim: int = 12,
        actual_proprio_dim: int = 12,
        camera_name_map: dict[str, str] | None = None,
        task_instruction: str = "pick up the object",
        dtype: str = "bfloat16",
        device: str = "cuda",
        num_inference_steps: int = 10,
        cache_dir: str | None = None,
        token_pool_size: int = 0,  # 0 = no pooling, >0 = pool prefix tokens to this size
        tokenizer_path: str | None = None,  # local path to tokenizer (avoids HF download)
    ):
        super().__init__()
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        self.actual_action_dim = actual_action_dim
        self.actual_proprio_dim = actual_proprio_dim
        self.task_instruction = self._clean_task(task_instruction)
        self.token_pool_size = token_pool_size

        self.camera_name_map = camera_name_map or {
            "left_wrist": "observation.images.left_wrist_0_rgb",
            "right_wrist": "observation.images.right_wrist_0_rgb",
            "right_front": "observation.images.base_0_rgb",
        }
        self.camera_order = [
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ]

        pi05_config = PI05Config(
            device=device,
            dtype=dtype,
            chunk_size=50,
            n_action_steps=50,
            max_action_dim=32,
            max_state_dim=32,
            image_resolution=(224, 224),
            num_inference_steps=num_inference_steps,
            gradient_checkpointing=False,
            compile_model=False,
        )
        pi05_config.input_features = {
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.right_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        }
        pi05_config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        }
        self.pi05_config = pi05_config
        self.num_inference_steps = pi05_config.num_inference_steps

        policy = PI05Policy.from_pretrained(model_path, config=pi05_config, cache_dir=cache_dir)
        self.pi05: PI05Pytorch = policy.model

        tokenizer_id = tokenizer_path or "leo009/paligemma-3b-pt-224"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=cache_dir)
        self.tokenizer.padding_side = "right"

        for param in self.pi05.parameters():
            param.requires_grad = False

    @staticmethod
    def _clean_task(task_instruction: str) -> str:
        return task_instruction.strip().replace("_", " ").replace("\n", " ")

    @property
    def token_dim(self) -> int:
        return self.pi05.paligemma_with_expert.paligemma.config.text_config.hidden_size

    @property
    def action_dim(self) -> int:
        return self.actual_action_dim

    @property
    def model_device(self) -> torch.device:
        return next(self.pi05.parameters()).device

    def _prepare_images(self, obs: Observation) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images: list[torch.Tensor] = []
        img_masks: list[torch.Tensor] = []
        device = self.model_device
        image_height, image_width = self.pi05_config.image_resolution
        batch_size = obs.proprio.shape[0]
        empty_image = torch.full((batch_size, 3, image_height, image_width), -1.0, device=device)

        obs_to_pi05 = {}
        for obs_key, pi05_key in self.camera_name_map.items():
            if obs_key in obs.images:
                obs_to_pi05[pi05_key] = obs.images[obs_key]

        for camera_key in self.camera_order:
            if camera_key not in obs_to_pi05:
                images.append(empty_image.clone())
                img_masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
                continue

            image = obs_to_pi05[camera_key].to(device=device, dtype=torch.float32)
            if image.ndim != 4:
                raise ValueError(f"Expected image tensor with 4 dims for {camera_key}, got shape {tuple(image.shape)}")

            is_channels_first = image.shape[1] == 3
            if is_channels_first:
                image = image.permute(0, 2, 3, 1)

            if image.shape[1:3] != self.pi05_config.image_resolution:
                image = resize_with_pad_torch(image, image_height, image_width)

            image = image * 2.0 - 1.0

            if is_channels_first:
                image = image.permute(0, 3, 1, 2)

            images.append(image)
            img_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))

        return images, img_masks

    def _prepare_language_tokens(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.model_device
        proprio = obs.proprio[..., : self.actual_proprio_dim].to(device=device, dtype=torch.float32)
        proprio = pad_vector(proprio, self.pi05_config.max_state_dim)

        state_np = proprio.detach().cpu().numpy()
        state_np = np.clip(state_np, -1.0, 1.0)
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        discretized = np.digitize(state_np, bins=bins) - 1
        discretized = np.clip(discretized, 0, 255)

        prompts = []
        for state_tokens in discretized:
            state_str = " ".join(map(str, state_tokens))
            prompts.append(f"Task: {self.task_instruction}, State: {state_str};\nAction: ")

        encoded = self.tokenizer(
            prompts,
            max_length=self.pi05_config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(device)
        masks = encoded["attention_mask"].to(device=device, dtype=torch.bool)
        return tokens, masks

    @torch.no_grad()
    def forward_vla(self, obs: Observation) -> VLAOutput:
        images, img_masks = self._prepare_images(obs)
        tokens, masks = self._prepare_language_tokens(obs)

        batch_size = tokens.shape[0]
        device = tokens.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi05.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.pi05._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)

        self.pi05.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"
        outputs, past_key_values = self.pi05.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_output = outputs[0]

        x_t = self.pi05.sample_noise(
            (batch_size, self.pi05_config.chunk_size, self.pi05_config.max_action_dim),
            device,
        )
        dt = -1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            time_value = 1.0 + step * dt
            time_tensor = torch.full((batch_size,), time_value, dtype=torch.float32, device=device)
            v_t = self.pi05.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        sampled_actions = x_t[:, :, : self.actual_action_dim]

        # Optional pooling: reduce ~968 tokens to pool_size for RL token encoder
        final_tokens = prefix_output.to(dtype=torch.float32)
        if self.token_pool_size > 0 and final_tokens.shape[1] > self.token_pool_size:
            # Adaptive average pooling along token dim: (B, M, D) -> (B, pool_size, D)
            final_tokens = final_tokens.permute(0, 2, 1)  # (B, D, M)
            final_tokens = torch.nn.functional.adaptive_avg_pool1d(final_tokens, self.token_pool_size)
            final_tokens = final_tokens.permute(0, 2, 1)  # (B, pool_size, D)

        return VLAOutput(
            final_tokens=final_tokens,
            sampled_action_chunk=sampled_actions.to(dtype=torch.float32),
        )

    def supervised_loss(self, obs: Observation, expert_actions: torch.Tensor) -> torch.Tensor:
        images, img_masks = self._prepare_images(obs)
        tokens, masks = self._prepare_language_tokens(obs)

        actions = expert_actions[..., : self.actual_action_dim].to(device=self.model_device, dtype=torch.float32)
        actions = pad_vector(actions, self.pi05_config.max_action_dim)

        losses = self.pi05.forward(images, img_masks, tokens, masks, actions)
        return losses[:, :, : self.actual_action_dim].mean()
