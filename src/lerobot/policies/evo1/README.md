# EVO1 Notes

This directory contains the Evo1 policy integration for LeRobot/Evo-RL.

## Extra Dependencies

Compared with the default LeRobot policy stack, Evo1 usually needs these extras:

- `flash-attn`
- InternVL3 model weights
- `transformers` support for the InternVL3 remote code path

The most important compatibility constraint is:

- `flash-attn` must match your `torch` version, CUDA version, Python version, and wheel ABI.

## Known Working Combinations

Validated in this repo / recent experiments:

- Evo-RL A800 environment:
  - `torch==2.4.1+cu118`
  - `flash_attn==2.6.3`
- Original Evo1-style environment discussed during reproduction:
  - `python==3.10`
  - `torch==2.5.1`
  - `torchvision==0.20.1`
  - `transformers==4.39.0`
  - `flash-attn==2.8.3`
## Training Notes

- `training_stage=stage1` freezes the VLM and trains the action head.
- `training_stage=stage2` finetunes both the VLM and the action head.
- In this repo, `resume_pretrain=true` is the Evo1-style stage handoff: load weights only, but start a fresh optimizer/scheduler state.
- We currently recommend `dataset.image_transforms.enable=false` in the validated training commands because the image augmentation pipeline was not stable in the tested environment.
- Stage2 now defaults to `embedder_tensor_fastpath=true`, `enable_vlm_gradient_checkpointing=true`, and
  `gradient_checkpointing_use_reentrant=false`.
- When stage2 is launched from `--policy.path=<stage1_ckpt>`, the stage1 checkpoint config is loaded first.
  In practice, it is safer to explicitly pass the stage2 memory/speed flags on the CLI so they override any
  stage1-era values saved in `config.json`.

## Optimization Summary

The current Evo1 integration includes a small set of stage2-oriented training optimizations merged from the Evo_2 branch while preserving the LeRobot/Evo-RL structure:

- Runtime config switches were added for `embedder_tensor_fastpath`, `enable_vlm_gradient_checkpointing`, and `gradient_checkpointing_use_reentrant`.
- Evo1 image collection during training no longer forces a `GPU -> CPU -> GPU` round trip before the VLM embedder.
- InternVL3 stage2 now enables a more memory-friendly checkpointing path and disables `use_cache` when checkpointing is active.
- The main `lerobot_train` entrypoint was kept intact; only lightweight Evo1 stage2 runtime logging was added.
- These Evo1-specific optimizations apply to `stage2` training and to the final `RL policy train` stage when it uses
  `policy.type=evo1` with `training_stage=stage2`. They do not apply to `value-train` or `value-infer`, which use
  the separate value-model pipeline.
- In the current local environment, the default `pistar06` language backbone `google/gemma-3-270m` may fail with
  a gated-repo `403`. For runnable value/RL scripts, prefer explicitly setting
  `--value.language_repo_id=Qwen/Qwen2.5-0.5B`.

## Stage 1 Command

Example: 4 GPUs, effective batch size `16` (`4 x 4`), `5000` steps, save every `2500`.

```bash
cd /root/private_data/code/Evo-RL
PYTHONPATH=src /opt/conda/bin/accelerate launch \
  --config_file src/lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=Elvinky/bi_so101_fold_towel_d0 \
  --dataset.root=/root/private_data/data/bi_so101_fold_towel_d0 \
  --dataset.image_transforms.enable=false \
  --policy.type=evo1 \
  --policy.training_stage=stage1 \
  --policy.vlm_model_name=/root/private_data/models/InternVL3-1B \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=4 \
  --num_workers=4 \
  --steps=5000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=2500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=/root/private_data/code/Evo-RL/outputs/train/evo1_stage1
```

## Stage 2 Command

Example: start stage2 from the stage1 `5000`-step checkpoint, `80000` steps, save every `10000`.

```bash
cd /root/private_data/code/Evo-RL
PYTHONPATH=src /opt/conda/bin/accelerate launch \
  --config_file src/lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=Elvinky/bi_so101_fold_towel_d0 \
  --dataset.root=/root/private_data/data/bi_so101_fold_towel_d0 \
  --dataset.image_transforms.enable=false \
  --policy.path=/root/private_data/code/Evo-RL/outputs/train/evo1_stage1/checkpoints/005000/pretrained_model \
  --policy.training_stage=stage2 \
  --resume_pretrain=true \
  --policy.vlm_model_name=/root/private_data/models/InternVL3-1B \
  --policy.device=cuda \
  --policy.use_flash_attn=true \
  --policy.embedder_tensor_fastpath=true \
  --policy.enable_vlm_gradient_checkpointing=true \
  --policy.gradient_checkpointing_use_reentrant=false \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=4 \
  --num_workers=4 \
  --steps=80000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=/root/private_data/code/Evo-RL/outputs/train/evo1_stage2
```
