from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from lerobot.policies.evo1.flow_matching import FlowmatchingActionHead
from lerobot.policies.evo1.internvl3_embedder import InternVL3Embedder


class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            image_size=config.get("image_size", 448),
            device=self._device,
            num_language_layers=config.get("vlm_num_layers", 14),
            model_dtype=config.get("vlm_dtype", "bfloat16"),
            use_flash_attn=config.get("use_flash_attn", True),
        )

        action_head_type = config.get("action_head", "flowmatching").lower()
        if action_head_type != "flowmatching":
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        horizon = config.get("action_horizon", config.get("horizon", 16))
        per_action_dim = config.get("per_action_dim", 7)
        action_dim = horizon * per_action_dim

        config["horizon"] = horizon
        config["per_action_dim"] = per_action_dim
        config["action_dim"] = action_dim

        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_head = FlowmatchingActionHead(
            config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1),
            )
        ).to(self._device)

    def get_vl_embeddings(
        self,
        images: list[Image.Image | torch.Tensor],
        image_mask: torch.Tensor,
        prompt: str = "",
        return_cls_only: bool | None = None,
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        if not images:
            raise ValueError("EVO1 expects at least one image per sample.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: list | torch.Tensor) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError(f"Unsupported state input type: {type(state_input)}")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ):
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
            )
        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
        )

    @torch.no_grad()
    def run_inference(
        self,
        images: list[Image.Image | torch.Tensor],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: list | torch.Tensor,
        return_cls_only: bool | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_tokens = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
        )
        state_tensor = self.prepare_state(state_input)
        return self.predict_action(
            fused_tokens,
            state_tensor,
            action_mask=action_mask,
            embodiment_ids=embodiment_ids,
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_ids: torch.Tensor | None = None,
    ):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def set_finetune_flags(self):
        if not self.config.get("finetune_vlm", False):
            self._freeze_module(self.embedder)

        if not self.config.get("finetune_action_head", False):
            self._freeze_module(self.action_head)
