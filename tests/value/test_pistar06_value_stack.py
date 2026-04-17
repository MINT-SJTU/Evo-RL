#!/usr/bin/env python

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import lerobot.processor.tokenizer_processor as tokenizer_processor
import lerobot.values.pistar06.modeling_pistar06 as pistar06_modeling
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import (
    BackboneLoadResult,
    PISTAR06_SAVE_INFO,
    Pistar06Model,
    Pistar06Policy,
)
from lerobot.values.pistar06.processor_pistar06 import (
    PISTAR06_IMAGE_MASK_KEY,
    PISTAR06_IMAGES_KEY,
    Pistar06PrepareTaskPromptProcessorStep,
    make_pistar06_pre_post_processors,
)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, prompts, max_length, truncation, padding, return_tensors, padding_side="right"):
        del truncation, padding, return_tensors, padding_side
        bsize = len(prompts)
        input_ids = torch.zeros((bsize, max_length), dtype=torch.long)
        attention_mask = torch.zeros((bsize, max_length), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            token_count = min(max_length, max(1, len(prompt.split())))
            input_ids[i, :token_count] = torch.arange(1, token_count + 1, dtype=torch.long)
            attention_mask[i, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyImageProcessor:
    size = {"height": 32, "width": 32}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _DummyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(3, 16)
        self.config = SimpleNamespace(hidden_size=16)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        return self.image_proj(pooled)


class _DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embed = nn.Embedding(2048, 32)
        self.proj = nn.Linear(32, 32)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        hidden = self.proj(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class _DummyAutoModel:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        output_loading_info = bool(kwargs.pop("output_loading_info", False))
        model = _DummyVisionModel() if "siglip" in repo_id.lower() else _DummyLanguageModel()
        if output_loading_info:
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
        return model


class _DummyAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        del repo_id
        output_loading_info = bool(kwargs.pop("output_loading_info", False))
        model = SimpleNamespace(model=_DummyLanguageModel())
        if output_loading_info:
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
        return model


class _DummyAutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(architectures=[])


def _make_pi05_backbone_bundle() -> BackboneLoadResult:
    return BackboneLoadResult(
        vision_encoder=_DummyVisionModel(),
        language_model=_DummyLanguageModel(),
        image_resolution=(224, 224),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )


@pytest.fixture
def hf_stubs(monkeypatch):
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", _DummyTokenizer)
    monkeypatch.setattr(pistar06_modeling, "AutoImageProcessor", _DummyImageProcessor)
    monkeypatch.setattr(pistar06_modeling, "AutoModel", _DummyAutoModel)
    monkeypatch.setattr(pistar06_modeling, "AutoModelForCausalLM", _DummyAutoModelForCausalLM)
    monkeypatch.setattr(pistar06_modeling, "AutoConfig", _DummyAutoConfig)


class _FakeDatasetTable:
    def __init__(self, data: dict[str, list]):
        self._data = data
        self.column_names = list(data.keys())

    def with_format(self, _):
        return self

    def __len__(self):
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._data
        return self._data[key]


class _TaskLocAccessor:
    def __init__(self, mapping: dict[str, int]):
        self._mapping = mapping

    def __getitem__(self, task_name: str):
        return SimpleNamespace(task_index=self._mapping[task_name])


class _TaskIndexTable:
    def __init__(self, mapping: dict[str, int]):
        self.index = mapping.keys()
        self.loc = _TaskLocAccessor(mapping)


def test_pistar06_processor_pads_missing_cameras_and_tokenizes(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front", "observation.images.wrist"],
        tokenizer_max_length=16,
    )
    preprocessor, _ = make_pistar06_pre_post_processors(cfg)

    raw_batch = {
        "task": ["pick bottle", "place bottle"],
        OBS_STATE: torch.rand(2, 12),
        "observation.images.front": torch.rand(2, 3, 48, 40),
    }
    processed = preprocessor(raw_batch)

    assert processed[OBS_LANGUAGE_TOKENS].shape == (2, 16)
    assert processed[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool
    assert processed[PISTAR06_IMAGES_KEY].shape == (2, 2, 3, 48, 40)
    assert torch.equal(processed[PISTAR06_IMAGE_MASK_KEY][:, 0], torch.ones(2, dtype=torch.bool))
    assert torch.equal(processed[PISTAR06_IMAGE_MASK_KEY][:, 1], torch.zeros(2, dtype=torch.bool))


def test_pistar06_processor_requires_task_field(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(device="cpu", camera_features=["observation.images.front"])
    preprocessor, _ = make_pistar06_pre_post_processors(cfg)

    with pytest.raises(KeyError, match="Missing task field"):
        preprocessor({OBS_STATE: torch.rand(2, 12), "observation.images.front": torch.rand(2, 3, 48, 40)})


def test_pistar06_processor_can_disable_state_in_prompt(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        include_state_in_prompt=False,
    )
    preprocessor, _ = make_pistar06_pre_post_processors(cfg)

    raw_batch = {
        "task": ["pick bottle", "place bottle"],
        "observation.images.front": torch.rand(2, 3, 48, 40),
    }
    processed = preprocessor(raw_batch)
    assert processed[OBS_LANGUAGE_TOKENS].shape == (2, cfg.tokenizer_max_length)


def test_pistar06_model_forward(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        state_proj_dim=16,
        num_bins=17,
    )
    model = Pistar06Model(cfg)

    outputs = model(
        input_ids=torch.randint(0, 20, (3, 12), dtype=torch.long),
        attention_mask=torch.ones(3, 12, dtype=torch.bool),
        images=torch.rand(3, 1, 3, 32, 32),
        image_attention_mask=torch.ones(3, 1, dtype=torch.bool),
    )
    assert outputs.shape == (3, 17)


def test_pistar06_model_can_use_pi05_backbone_bundle(monkeypatch, hf_stubs):
    del hf_stubs
    monkeypatch.setattr(
        pistar06_modeling,
        "_load_pi05_backbone_bundle",
        lambda cfg: _make_pi05_backbone_bundle(),
    )

    cfg = Pistar06Config(
        device="cpu",
        backbone_source="pi05",
        camera_features=["observation.images.front"],
        freeze_vision_encoder=True,
        freeze_language_model=True,
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        num_bins=17,
    )
    model = Pistar06Model(cfg)

    assert model.image_resolution == (224, 224)
    assert not any(param.requires_grad for param in model.vision_encoder.parameters())
    assert not any(param.requires_grad for param in model.language_model.parameters())

    outputs = model(
        input_ids=torch.randint(0, 20, (2, 12), dtype=torch.long),
        attention_mask=torch.ones(2, 12, dtype=torch.bool),
        images=torch.rand(2, 1, 3, 32, 32),
        image_attention_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    assert outputs.shape == (2, 17)


def test_pistar06_policy_requires_valid_camera_mask(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        state_proj_dim=16,
        num_bins=17,
    )
    policy = Pistar06Policy(config=cfg)

    with pytest.raises(ValueError, match="at least one valid camera"):
        policy.model(
            input_ids=torch.randint(0, 10, (2, 8), dtype=torch.long),
            attention_mask=torch.ones(2, 8, dtype=torch.bool),
            images=torch.rand(2, 1, 3, 16, 16),
            image_attention_mask=torch.zeros(2, 1, dtype=torch.bool),
        )


def test_pistar06_policy_forward_computes_loss(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        target_key="observation.value_target",
        num_bins=17,
        fusion_hidden_dim=32,
        fusion_num_heads=8,
    )
    policy = Pistar06Policy(config=cfg)

    batch = {
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (4, 12), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 12, dtype=torch.bool),
        PISTAR06_IMAGES_KEY: torch.rand(4, 1, 3, 32, 32),
        PISTAR06_IMAGE_MASK_KEY: torch.ones(4, 1, dtype=torch.bool),
        cfg.target_key: torch.tensor([-1.0, -0.8, -0.4, 0.0], dtype=torch.float32),
    }

    loss, loss_dict = policy.forward(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert "loss" in loss_dict
    assert "value_mae" in loss_dict


def test_pistar06_policy_forward_applies_loss_weights(hf_stubs):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        target_key="observation.value_target",
        loss_weight_key="observation.value_loss_weight",
        num_bins=17,
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        dropout=0.0,
    )
    policy = Pistar06Policy(config=cfg)
    policy.eval()

    base_batch = {
        OBS_LANGUAGE_TOKENS: torch.randint(0, 100, (4, 12), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 12, dtype=torch.bool),
        PISTAR06_IMAGES_KEY: torch.rand(4, 1, 3, 32, 32),
        PISTAR06_IMAGE_MASK_KEY: torch.ones(4, 1, dtype=torch.bool),
        cfg.target_key: torch.tensor([-1.0, -0.8, -0.4, 0.0], dtype=torch.float32),
    }
    loss_weight = torch.tensor([2.0, 1.0, 0.5, 0.5], dtype=torch.float32)

    loss_none_unweighted, _ = policy.forward(base_batch, reduction="none")
    weighted_batch = {**base_batch, cfg.loss_weight_key: loss_weight}
    loss_none_weighted, _ = policy.forward(weighted_batch, reduction="none")
    assert torch.allclose(loss_none_weighted, loss_none_unweighted * loss_weight, atol=1e-6)

    loss_mean_weighted, loss_dict = policy.forward(weighted_batch, reduction="mean")
    assert torch.isclose(loss_mean_weighted, loss_none_weighted.mean(), atol=1e-6)
    assert "loss_weight_mean" in loss_dict
    assert "loss_weight_min" in loss_dict
    assert "loss_weight_max" in loss_dict


def test_pistar06_prompt_step_with_state_has_no_action_suffix():
    step = Pistar06PrepareTaskPromptProcessorStep(
        task_key="task",
        include_state_in_prompt=True,
        state_feature=OBS_STATE,
        max_state_dim=4,
        state_discretization_bins=8,
    )
    transition = {
        "observation": {OBS_STATE: torch.tensor([[0.0, 0.5], [-0.5, 0.2]], dtype=torch.float32)},
        "action": None,
        "next.reward": 0.0,
        "next.done": False,
        "next.truncated": False,
        "info": {},
        "complementary_data": {"task": ["pick bottle", "place bottle"]},
    }
    out = step(transition)
    prompts = out["complementary_data"]["task"]

    assert len(prompts) == 2
    assert prompts[0].startswith("Task: pick bottle, State:")
    assert prompts[1].startswith("Task: place bottle, State:")
    assert "Action:" not in prompts[0]
    assert "Action:" not in prompts[1]
    assert prompts[0].endswith("\nValue: ")
    assert prompts[1].endswith("\nValue: ")


def test_pistar06_prompt_step_without_state_uses_clean_task_only():
    step = Pistar06PrepareTaskPromptProcessorStep(
        task_key="task",
        include_state_in_prompt=False,
        state_feature=OBS_STATE,
    )
    transition = {
        "observation": {},
        "action": None,
        "next.reward": 0.0,
        "next.done": False,
        "next.truncated": False,
        "info": {},
        "complementary_data": {"task": ["pick_bottle\nnow"]},
    }
    out = step(transition)
    prompts = out["complementary_data"]["task"]

    assert prompts == ["Task: pick bottle now\nValue: "]


def test_pistar06_save_pretrained_skips_frozen_vision_weights_and_loads_partial(hf_stubs, tmp_path):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        freeze_vision_encoder=True,
        freeze_language_model=False,
    )
    policy = Pistar06Policy(config=cfg)

    with torch.no_grad():
        policy.model.vision_encoder.image_proj.weight.fill_(7.0)
        policy.model.language_model.proj.weight.fill_(8.0)
        policy.model.value_head[0].weight.fill_(9.0)

    policy.save_pretrained(tmp_path)

    saved_state = load_file(tmp_path / "model.safetensors")
    assert not any(key.startswith("model.vision_encoder.") for key in saved_state)
    assert any(key.startswith("model.language_model.") for key in saved_state)
    assert any(key.startswith("model.value_head.") for key in saved_state)

    with open(tmp_path / PISTAR06_SAVE_INFO, encoding="utf-8") as f:
        save_info = json.load(f)
    assert save_info["weights_mode"] == "partial"
    assert "model.vision_encoder." in save_info["excluded_prefixes"]

    loaded = Pistar06Policy.from_pretrained(tmp_path)
    assert torch.allclose(
        loaded.model.language_model.proj.weight,
        torch.full_like(loaded.model.language_model.proj.weight, 8.0),
    )
    assert torch.allclose(
        loaded.model.value_head[0].weight,
        torch.full_like(loaded.model.value_head[0].weight, 9.0),
    )
    assert not torch.allclose(
        loaded.model.vision_encoder.image_proj.weight,
        torch.full_like(loaded.model.vision_encoder.image_proj.weight, 7.0),
    )


def test_pistar06_save_pretrained_only_head_when_both_encoders_frozen_and_strict_load_works(
    hf_stubs, tmp_path
):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        freeze_vision_encoder=True,
        freeze_language_model=True,
    )
    policy = Pistar06Policy(config=cfg)

    with torch.no_grad():
        policy.model.vision_encoder.image_proj.weight.fill_(5.0)
        policy.model.language_model.proj.weight.fill_(6.0)
        policy.model.image_projector[0].weight.fill_(11.0)
        policy.model.value_head[0].weight.fill_(12.0)

    policy.save_pretrained(tmp_path)

    saved_state = load_file(tmp_path / "model.safetensors")
    assert not any(key.startswith("model.vision_encoder.") for key in saved_state)
    assert not any(key.startswith("model.language_model.") for key in saved_state)
    assert any(key.startswith("model.image_projector.") for key in saved_state)
    assert any(key.startswith("model.value_head.") for key in saved_state)

    loaded = Pistar06Policy.from_pretrained(tmp_path, strict=True)
    assert torch.allclose(
        loaded.model.image_projector[0].weight,
        torch.full_like(loaded.model.image_projector[0].weight, 11.0),
    )
    assert torch.allclose(
        loaded.model.value_head[0].weight,
        torch.full_like(loaded.model.value_head[0].weight, 12.0),
    )
    assert not torch.allclose(
        loaded.model.vision_encoder.image_proj.weight,
        torch.full_like(loaded.model.vision_encoder.image_proj.weight, 5.0),
    )
    assert not torch.allclose(
        loaded.model.language_model.proj.weight,
        torch.full_like(loaded.model.language_model.proj.weight, 6.0),
    )


def test_pistar06_save_pretrained_keeps_full_weights_when_encoders_not_frozen(hf_stubs, tmp_path):
    del hf_stubs
    cfg = Pistar06Config(
        device="cpu",
        camera_features=["observation.images.front"],
        freeze_vision_encoder=False,
        freeze_language_model=False,
    )
    policy = Pistar06Policy(config=cfg)

    with torch.no_grad():
        policy.model.vision_encoder.image_proj.weight.fill_(2.0)
        policy.model.language_model.proj.weight.fill_(3.0)

    policy.save_pretrained(tmp_path)

    saved_state = load_file(tmp_path / "model.safetensors")
    assert any(key.startswith("model.vision_encoder.") for key in saved_state)
    assert any(key.startswith("model.language_model.") for key in saved_state)

    with open(tmp_path / PISTAR06_SAVE_INFO, encoding="utf-8") as f:
        save_info = json.load(f)
    assert save_info["weights_mode"] == "full"
    assert save_info["excluded_prefixes"] == []

    loaded = Pistar06Policy.from_pretrained(tmp_path)
    assert torch.allclose(
        loaded.model.vision_encoder.image_proj.weight,
        torch.full_like(loaded.model.vision_encoder.image_proj.weight, 2.0),
    )
    assert torch.allclose(
        loaded.model.language_model.proj.weight,
        torch.full_like(loaded.model.language_model.proj.weight, 3.0),
    )
