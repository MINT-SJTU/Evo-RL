import math
import json

import pytest
import torch
import torch.nn.functional as F

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.modeling_evo1 import EVO1Policy
from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME


class DummyEmbedder(torch.nn.Module):
    def __init__(self, model_name="dummy", image_size=32, device="cpu"):
        super().__init__()
        self.device = device
        self.proj = torch.nn.Linear(4, 8)

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors,
        image_mask,
        text_prompt,
        return_cls_only=True,
    ):
        token = self.proj(torch.ones(1, 4))
        if return_cls_only:
            return token
        return token.unsqueeze(1).repeat(1, 3, 1)


def make_test_config() -> Evo1Config:
    return Evo1Config(
        device="cpu",
        use_amp=False,
        chunk_size=5,
        n_action_steps=2,
        max_state_dim=6,
        max_action_dim=6,
        max_views=2,
        embed_dim=8,
        hidden_dim=16,
        state_hidden_dim=8,
        num_heads=1,
        num_layers=1,
        num_inference_timesteps=2,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )


def test_evo1_policy_forward_and_inference_smoke(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)
    policy = EVO1Policy(make_test_config())

    batch = {
        "task": ["pick cube", "place cube"],
        "observation.state": torch.rand(2, 6),
        "observation.images.front": torch.rand(2, 3, 16, 16),
        "observation.images.wrist": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 5, 6),
    }

    loss, info = policy.forward(batch)
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert "loss" in info

    action_chunk = policy.predict_action_chunk(batch)
    assert action_chunk.shape == (2, 5, 6)

    action = policy.select_action(batch)
    assert action.shape == (2, 6)


def test_evo1_processors_keep_task_and_state_shape():
    preprocessor, postprocessor = make_evo1_pre_post_processors(
        make_test_config(),
        dataset_stats={
            "observation.state": {"min": torch.zeros(6), "max": torch.ones(6)},
            "action": {"min": torch.zeros(6), "max": torch.ones(6)},
        },
    )

    processed = preprocessor(
        {
            "task": "pick cube",
            "observation.state": torch.full((6,), 0.5),
            "observation.images.front": torch.ones(3, 16, 16),
            "observation.images.wrist": torch.ones(3, 16, 16),
        }
    )

    assert processed["task"] == ["pick cube"]
    assert processed["observation.state"].shape == (1, 6)

    action = postprocessor(torch.zeros(1, 6))
    assert action.shape == (1, 6)


def test_evo1_scheduler_matches_original_lambda():
    cfg = make_test_config()
    cfg.scheduler_warmup_steps = 3
    scheduler_cfg = cfg.get_scheduler_preset()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=cfg.optimizer_lr)
    scheduler = scheduler_cfg.build(optimizer, num_training_steps=20)

    def old_lr_lambda(current_step: int) -> float:
        if current_step < cfg.scheduler_warmup_steps:
            return current_step / max(1, cfg.scheduler_warmup_steps)
        progress = (current_step - cfg.scheduler_warmup_steps) / max(1, 20 - cfg.scheduler_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    for step in [0, 1, 2, 5, 10, 19, 20]:
        assert scheduler.lr_lambdas[0](step) == pytest.approx(old_lr_lambda(step))


def test_evo1_stage_freeze_semantics_match_original_training(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)

    stage1_cfg = make_test_config()
    stage1_cfg.training_stage = "stage1"
    stage1_cfg.__post_init__()
    stage1_policy = EVO1Policy(stage1_cfg)

    assert not any(param.requires_grad for param in stage1_policy.model.embedder.parameters())
    assert any(param.requires_grad for param in stage1_policy.model.action_head.parameters())

    stage2_cfg = make_test_config()
    stage2_cfg.training_stage = "stage2"
    stage2_cfg.__post_init__()
    stage2_policy = EVO1Policy(stage2_cfg)

    assert all(param.requires_grad for param in stage2_policy.model.embedder.parameters())
    assert all(param.requires_grad for param in stage2_policy.model.action_head.parameters())


def test_evo1_training_stage_sets_expected_flags():
    stage1_cfg = make_test_config()
    stage1_cfg.training_stage = "stage1"
    stage1_cfg.__post_init__()
    assert stage1_cfg.finetune_vlm is False
    assert stage1_cfg.finetune_action_head is True
    assert stage1_cfg.optimizer_betas == (0.9, 0.999)
    assert stage1_cfg.drop_last is True

    stage2_cfg = make_test_config()
    stage2_cfg.training_stage = "stage2"
    stage2_cfg.__post_init__()
    assert stage2_cfg.finetune_vlm is True
    assert stage2_cfg.finetune_action_head is True


def test_evo1_batch_preparation_matches_original_dataset_contract(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)

    cfg = make_test_config()
    cfg.max_views = 3
    policy = EVO1Policy(cfg)

    batch = {
        "task": ["fold towel", "stack cloth"],
        "observation.state": torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]], dtype=torch.float32
        ),
        "observation.images.front": torch.rand(2, 1, 3, 16, 16),
        "observation.images.wrist": torch.rand(2, 1, 3, 16, 16),
        "action": torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4),
    }

    prompts = policy._normalize_task_batch(batch)
    state, state_mask = policy._prepare_state(batch)
    actions, action_mask = policy._prepare_actions(batch)
    image_batches, image_masks = policy._collect_image_batches(batch)

    assert prompts == ["fold towel", "stack cloth"]

    assert state.shape == (2, 6)
    assert state_mask.shape == (2, 6)
    assert torch.allclose(state[:, :4].float(), torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
    assert torch.equal(state_mask[:, :4], torch.ones(2, 4, dtype=torch.bool))
    assert torch.equal(state_mask[:, 4:], torch.zeros(2, 2, dtype=torch.bool))

    assert actions.shape == (2, 5, 6)
    assert action_mask.shape == (2, 5, 6)
    assert torch.allclose(actions[:, :, :4].float(), batch["action"])
    assert torch.equal(action_mask[:, :, :4], torch.ones(2, 5, 4, dtype=torch.bool))
    assert torch.equal(action_mask[:, :, 4:], torch.zeros(2, 5, 2, dtype=torch.bool))

    assert len(image_batches) == 2
    assert all(len(sample) == 3 for sample in image_batches)
    assert image_masks.shape == (2, 3)
    assert torch.equal(image_masks, torch.tensor([[True, True, False], [True, True, False]]))
    assert torch.count_nonzero(image_batches[0][2]) == 0
    assert torch.count_nonzero(image_batches[1][2]) == 0


def test_evo1_explicit_masks_take_priority(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)
    policy = EVO1Policy(make_test_config())

    batch = {
        "observation.state": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        "state_mask": torch.tensor([[True, False, True, False]], dtype=torch.bool),
        "action": torch.arange(5 * 4, dtype=torch.float32).view(1, 5, 4),
        "action_mask": torch.tensor(
            [[[True, False, True, False]] * 5],
            dtype=torch.bool,
        ),
    }

    _state, state_mask = policy._prepare_state(batch)
    _action, action_mask = policy._prepare_actions(batch)

    assert torch.equal(state_mask[0, :4], torch.tensor([True, False, True, False]))
    assert torch.equal(action_mask[0, :, :4], batch["action_mask"][0])
    assert torch.equal(state_mask[0, 4:], torch.zeros(2, dtype=torch.bool))
    assert torch.equal(action_mask[0, :, 4:], torch.zeros(5, 2, dtype=torch.bool))


def test_evo1_masked_loss_matches_original_training_formula(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)
    policy = EVO1Policy(make_test_config())

    pred_velocity = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    target_velocity = torch.tensor([[0.5, 2.5, 1.0, 6.0]], dtype=torch.float32)
    action_mask = torch.tensor([[[True, True], [False, False]]], dtype=torch.bool)

    loss = policy._compute_masked_loss(pred_velocity, target_velocity, action_mask, reduction="mean")

    flat_mask = action_mask.view(action_mask.shape[0], -1)
    pred_masked = pred_velocity * flat_mask
    old_loss = F.mse_loss(pred_masked, target_velocity, reduction="mean") * (
        pred_velocity.numel() / flat_mask.sum().item()
    )

    assert loss.item() == pytest.approx(old_loss.item())


def test_resume_pretrain_uses_policy_weights_without_checkpoint_state(monkeypatch, tmp_path):
    from lerobot.configs import train as train_config_module

    pretrained_path = tmp_path / "stage1" / "pretrained_model"
    pretrained_path.mkdir(parents=True)
    loaded_cfg = make_test_config()
    loaded_cfg.push_to_hub = False

    monkeypatch.setattr(train_config_module.parser, "get_path_arg", lambda field_name: str(pretrained_path))
    monkeypatch.setattr(train_config_module.parser, "get_cli_overrides", lambda field_name: [])
    monkeypatch.setattr(
        train_config_module.PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, path, cli_overrides=None: loaded_cfg),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/repo"),
        policy=None,
        output_dir=tmp_path / "fresh_run",
        resume_pretrain=True,
    )

    cfg.validate()

    assert cfg.policy is loaded_cfg
    assert cfg.policy.pretrained_path == pretrained_path
    assert cfg.checkpoint_path is None
    assert cfg.resume is False


def test_resume_and_resume_pretrain_are_mutually_exclusive(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/repo"),
        policy=make_test_config(),
        output_dir=tmp_path / "fresh_run",
        resume=True,
        resume_pretrain=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_resume_pretrain_requires_policy_weights(tmp_path):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/repo"),
        policy=make_test_config(),
        output_dir=tmp_path / "fresh_run",
        resume_pretrain=True,
    )

    with pytest.raises(ValueError, match="expects weights"):
        cfg.validate()


def test_evo1_new_preprocessor_saves_device_processor(tmp_path):
    preprocessor, _ = make_evo1_pre_post_processors(
        make_test_config(),
        dataset_stats={
            "observation.state": {"min": torch.zeros(6), "max": torch.ones(6)},
            "action": {"min": torch.zeros(6), "max": torch.ones(6)},
        },
    )

    preprocessor.save_pretrained(tmp_path)

    with open(tmp_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json") as f:
        saved = json.load(f)

    saved_step_keys = {step["registry_name"] for step in saved["steps"]}
    assert "device_processor" in saved_step_keys


def test_evo1_training_forward_casts_core_tensors_to_bfloat16(monkeypatch):
    import lerobot.policies.evo1.evo1_model as evo1_model_module
    import lerobot.policies.evo1.modeling_evo1 as modeling_evo1_module

    monkeypatch.setattr(evo1_model_module, "InternVL3Embedder", DummyEmbedder)
    monkeypatch.setattr(
        modeling_evo1_module.EVO1Policy,
        "_training_compute_dtype",
        property(lambda self: torch.bfloat16),
    )

    policy = EVO1Policy(make_test_config())
    captured = {}

    def fake_prepare_state(_batch):
        return torch.randn(2, 6, dtype=torch.float32), torch.ones(2, 6, dtype=torch.bool)

    def fake_prepare_actions(_batch):
        return torch.randn(2, 5, 6, dtype=torch.float32), torch.ones(2, 5, 6, dtype=torch.bool)

    def fake_compute_fused_tokens(_prompts, _image_batches, _image_masks):
        return torch.randn(2, 3, 8, dtype=torch.float32)

    def fake_get_embodiment_ids(_batch, _batch_size):
        return torch.zeros(2, dtype=torch.long)

    def fake_collect_image_batches(_batch):
        return [[torch.zeros(3, 16, 16)] * 2 for _ in range(2)], torch.ones(2, 2, dtype=torch.bool)

    def fake_model_forward(fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
        captured["fused_tokens_dtype"] = fused_tokens.dtype
        captured["state_dtype"] = state.dtype
        captured["actions_gt_dtype"] = actions_gt.dtype
        captured["action_mask_dtype"] = action_mask.dtype
        batch_size = state.shape[0]
        flat_dim = actions_gt.shape[1] * actions_gt.shape[2]
        return torch.zeros(batch_size, flat_dim, dtype=torch.bfloat16), torch.zeros_like(actions_gt)

    monkeypatch.setattr(policy, "_prepare_state", fake_prepare_state)
    monkeypatch.setattr(policy, "_prepare_actions", fake_prepare_actions)
    monkeypatch.setattr(policy, "_compute_fused_tokens", fake_compute_fused_tokens)
    monkeypatch.setattr(policy, "_get_embodiment_ids", fake_get_embodiment_ids)
    monkeypatch.setattr(policy, "_collect_image_batches", fake_collect_image_batches)
    monkeypatch.setattr(policy.model, "forward", fake_model_forward)

    loss, info = policy.forward({"task": ["a", "b"]})

    assert captured["fused_tokens_dtype"] == torch.bfloat16
    assert captured["state_dtype"] == torch.bfloat16
    assert captured["actions_gt_dtype"] == torch.bfloat16
    assert captured["action_mask_dtype"] == policy._compute_dtype
    assert loss.dtype == torch.bfloat16
    assert "loss" in info
