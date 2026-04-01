import functools
import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # nosec B105
IMG_START_TOKEN = "<img>"  # nosec B105
IMG_END_TOKEN = "</img>"  # nosec B105


# === Image Transformations ===
def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# === Aspect Ratio Handling ===
@functools.lru_cache(maxsize=10000)
def get_target_aspect_ratio(orig_width, orig_height, image_size, min_num, max_num):
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    target_aspect_ratio = get_target_aspect_ratio(orig_width, orig_height, image_size, min_num, max_num)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        num_language_layers: int | None = 14,
        model_dtype: str | torch.dtype = "bfloat16",
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self._requested_device = device
        self.image_size = image_size
        self.num_language_layers = num_language_layers
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        if isinstance(model_dtype, str):
            try:
                model_dtype = getattr(torch, model_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported EVO1 vlm_dtype '{model_dtype}'") from exc
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self._requested_device)

        if hasattr(self.model.language_model, "model"):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        if self.num_language_layers is not None:
            layers = layers[: self.num_language_layers]

        if hasattr(self.model.language_model, "model"):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _dynamic_preprocess_tensor(
        self, image_t, min_num=1, max_num=1, use_thumbnail=False
    ):
        # image_t shape expected: [C, H, W]
        C, orig_height, orig_width = image_t.shape

        # get ratio by cache
        target_aspect_ratio = get_target_aspect_ratio(
            orig_width, orig_height, self.image_size, min_num, max_num
        )

        ratio_w, ratio_h = target_aspect_ratio[0], target_aspect_ratio[1]
        target_width = self.image_size * ratio_w
        target_height = self.image_size * ratio_h
        blocks = ratio_w * ratio_h

        # Resize on GPU
        # image_t expected shape for interpolate is [C, H, W] mapping to size
        resized_img = TF.resize(
            image_t,
            size=[target_height, target_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        # Eliminate the for-loop using view and permute for zero-copy strided tensor tiling
        # resized_img shape: [C, ratio_h * image_size, ratio_w * image_size]
        # 1. view to -> [C, ratio_h, image_size, ratio_w, image_size]
        reshaped = resized_img.view(C, ratio_h, self.image_size, ratio_w, self.image_size)
        # 2. permute to -> [ratio_h, ratio_w, C, image_size, image_size]
        permuted = reshaped.permute(1, 3, 0, 2, 4)
        # 3. reshape to -> [blocks, C, image_size, image_size]
        stacked_tiles = permuted.reshape(blocks, C, self.image_size, self.image_size)

        if use_thumbnail and blocks != 1:
            thumbnail_img = TF.resize(
                image_t,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            # Concat the thumbnail directly along the batch dimension
            stacked_tiles = torch.cat([stacked_tiles, thumbnail_img.unsqueeze(0)], dim=0)

        return stacked_tiles

    def _preprocess_images(
        self, image_tensors_batch: list[list[Image.Image | torch.Tensor]]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        pixel_values_list = []
        batch_num_tiles_list = []
        model_dtype = next(self.model.parameters()).dtype
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )

        for image_tensors in image_tensors_batch:
            num_tiles_list = []
            for image in image_tensors:
                if isinstance(image, torch.Tensor):
                    if (
                        image.ndim == 3
                        and image.shape[0] == 3
                        and image.shape[1] == self.image_size
                        and image.shape[2] == self.image_size
                    ):
                        image_t = image.to(device=self.device)
                        if not torch.is_floating_point(image_t):
                            image_t = image_t.to(torch.float32) / 255.0
                        else:
                            image_t = image_t.to(torch.float32)
                            if float(image_t.max().item()) > 1.0:
                                image_t = image_t / 255.0
                        tile_tensors = (image_t.unsqueeze(0) - mean) / std
                    else:
                        image_t = image.to(device=self.device, dtype=torch.float32)
                        if not torch.is_floating_point(image):
                            image_t = image_t / 255.0
                        else:
                            if float(image_t.max().item()) > 1.0:
                                image_t = image_t / 255.0
                        tiles = self._dynamic_preprocess_tensor(image_t)
                        tile_tensors = (tiles - mean) / std
                else:
                    image_t = TF.to_tensor(image).to(device=self.device, dtype=torch.float32)
                    tiles = self._dynamic_preprocess_tensor(image_t)
                    tile_tensors = (tiles - mean) / std

                tile_tensors = tile_tensors.to(dtype=model_dtype)
                pixel_values_list.append(tile_tensors)
                num_tiles_list.append(tile_tensors.shape[0])
            batch_num_tiles_list.append(num_tiles_list)

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.empty(
                0,
                3,
                self.image_size,
                self.image_size,
                dtype=model_dtype,
                device=self.device,
            )

        return pixel_values, batch_num_tiles_list

    def _build_multimodal_prompt(
        self,
        batch_num_tiles_list: list[list[int]],
        text_prompts: list[str],
    ) -> list[str]:
        if len(batch_num_tiles_list) != len(text_prompts):
            raise ValueError(
                f"InternVL3 batch mismatch: num_image_batches={len(batch_num_tiles_list)} num_text_prompts={len(text_prompts)}"
            )

        prompts = []
        for num_tiles_list, text_prompt in zip(batch_num_tiles_list, text_prompts, strict=True):
            prompt_segments = []
            for i, tile_count in enumerate(num_tiles_list):
                token_count = self.model.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i + 1}: {image_tokens}\n")
            prompts.append("".join(prompt_segments) + text_prompt.strip())

        return prompts

    def _prepare_and_fuse_embeddings(
        self,
        prompts: list[str],
        vit_embeds: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        untruncated_ids = self.tokenizer(prompts, padding=False, truncation=False)["input_ids"]
        true_sequence_length = max((len(ids) for ids in untruncated_ids), default=0)

        if true_sequence_length > self.max_text_length:
            logging.warning(
                "InternVL3 prompt truncated in batch: max_length=%s actual_max_length=%s",
                self.max_text_length,
                true_sequence_length,
            )

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        img_token_mask = input_ids == self.img_context_token_id
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        batch_size, _, channels = input_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, channels).to(device=input_embeds.device, dtype=input_embeds.dtype)

        tokens_per_tile = self.model.num_image_token
        vit_idx = 0
        actual_vis_tokens_list = img_token_mask.sum(dim=1).tolist()

        for batch_idx in range(batch_size):
            expected_vis_tokens = sum(batch_num_tiles_list[batch_idx]) * tokens_per_tile
            actual_vis_tokens = int(actual_vis_tokens_list[batch_idx])
            if actual_vis_tokens > expected_vis_tokens:
                raise ValueError(
                    "InternVL3 detected more image placeholder tokens than expected in prompt construction: "
                    f"batch_idx={batch_idx}, actual={actual_vis_tokens}, expected={expected_vis_tokens}"
                )

            if vit_idx + expected_vis_tokens > vit_embeds.shape[0]:
                raise ValueError(
                    "InternVL3 produced fewer image tokens than expected for batch fusion: "
                    f"need_up_to={vit_idx + expected_vis_tokens}, got={vit_embeds.shape[0]}"
                )

            item_vit_embeds = vit_embeds[vit_idx : vit_idx + expected_vis_tokens]
            vit_idx += expected_vis_tokens

            if actual_vis_tokens > 0:
                input_embeds[batch_idx, img_token_mask[batch_idx]] = item_vit_embeds[:actual_vis_tokens]

            current_token_idx = 0
            img_token_locations = torch.where(img_token_mask[batch_idx])[0]
            for image_idx, num_tiles_for_this_image in enumerate(batch_num_tiles_list[batch_idx]):
                num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
                if not bool(image_masks[batch_idx, image_idx].item()):
                    start_offset = current_token_idx
                    end_offset = min(
                        current_token_idx + num_tokens_for_this_image,
                        int(img_token_locations.shape[0]),
                    )
                    if start_offset < end_offset:
                        masked_token_indices = img_token_locations[start_offset:end_offset]
                        attention_mask[batch_idx, masked_token_indices] = 0
                current_token_idx += num_tokens_for_this_image

        return input_embeds, attention_mask

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Image.Image | torch.Tensor] | list[list[Image.Image | torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str | list[str],
        return_cls_only: bool = True,
    ):
        is_batched_input = bool(image_tensors) and isinstance(image_tensors[0], list)
        if is_batched_input:
            image_tensors_batch = image_tensors
        else:
            image_tensors_batch = [image_tensors]

        batch_size = len(image_tensors_batch)
        if batch_size == 0:
            raise ValueError("InternVL3 expects at least one batch item.")

        if isinstance(text_prompt, str):
            text_prompts = [text_prompt] * batch_size
        else:
            text_prompts = text_prompt
            if len(text_prompts) != batch_size:
                raise ValueError(
                    f"InternVL3 batch mismatch: num_text_prompts={len(text_prompts)} num_batches={batch_size}"
                )

        if image_mask.ndim == 1:
            image_masks = image_mask.unsqueeze(0)
        else:
            image_masks = image_mask

        if image_masks.shape[0] != batch_size:
            raise ValueError(
                f"InternVL3 batch mismatch: image_mask_batch={image_masks.shape[0]} num_batches={batch_size}"
            )

        for batch_idx, image_list in enumerate(image_tensors_batch):
            if len(image_list) == 0:
                raise ValueError(f"InternVL3 expects at least one image per batch item, got empty at batch_idx={batch_idx}")
            if int(image_masks.shape[1]) != len(image_list):
                raise ValueError(
                    "InternVL3 mask/image count mismatch: "
                    f"batch_idx={batch_idx}, num_masks={image_masks.shape[1]}, num_images={len(image_list)}"
                )

        image_masks = image_masks.to(device=self.device, dtype=torch.bool)
        pixel_values, batch_num_tiles_list = self._preprocess_images(image_tensors_batch)

        if pixel_values.shape[0] == 0:
            logging.warning("InternVL3 received an empty image batch after preprocessing.")
            hidden_size = self.model.language_model.get_input_embeddings().embedding_dim
            vit_embeds = torch.empty(0, hidden_size, dtype=next(self.model.parameters()).dtype, device=self.device)
        else:
            vit_embeds = self.model.extract_feature(pixel_values)

        prompts = self._build_multimodal_prompt(batch_num_tiles_list, text_prompts)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(
            prompts,
            vit_embeds,
            image_masks,
            batch_num_tiles_list,
        )

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
