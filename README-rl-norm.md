# RL Value 归一化说明（q 分位数 + 尾部硬截断）

本文档说明本仓库中 value target 归一化逻辑的最新改动、配置方式与使用建议。

## 1. 为什么要改

原始逻辑按 task 内 `episode length` 的最大值（`max`）做归一化。  
当长度分布是长尾时，少数超长 episode 会把尺度拉得很大，导致大多数样本的 value 绝对值非常小（例如 `0.0005`），训练信号分辨率不足。

为了解决这个问题，当前改为：

- 使用 task 内长度分位数（默认 `q=0.8`）作为归一化尺度；
- 对尾部 `remaining_steps` 做硬截断（hard clipping）。

## 2. 当前归一化逻辑

对每个 task，先统计该 task 全部 episode 长度集合 `L_task`，计算：

- `S = quantile(L_task, q)`，默认 `q=0.8`
- `c_fail = S * c_fail_coef`
- `denom = S + c_fail`

对每个样本帧：

1. `remaining_steps = episode_length - frame_index - 1`
2. 尾部硬截断：`remaining_eff = min(remaining_steps, S)`
3. 成功轨迹：`g = -remaining_eff`
4. 失败轨迹：`g = -remaining_eff - c_fail`
5. `v = clip(g / denom, bin_min, bin_max)`

其中默认输出区间仍是 `[-1, 0]`（由 value model 配置中的 `bin_min/bin_max` 控制）。

## 3. 和旧版逻辑的关键差异

- 旧版分母主要受 `task_max_length` 控制，容易被极端长尾放大；
- 新版分母由 `task_scale = q 分位数` 控制，更鲁棒；
- 新版对 `remaining_steps` 超过 `task_scale` 的尾部直接硬截断，避免极端时长持续拉低 value。

## 4. 新增配置项

### 4.1 训练（`lerobot-value-train`）

位置：`targets.length_scale_quantile`

- 默认：`0.8`
- 合法范围：`(0, 1]`
- 含义：按 task 内 episode 长度的该分位数作为归一化尺度。

对应代码：

- `src/lerobot/configs/value_train.py`
- `src/lerobot/values/pistar06/modeling_pistar06.py`

### 4.2 推理/ACP（`lerobot-value-infer`）

位置：`acp.length_scale_quantile`

- 默认：`0.8`
- 合法范围：`(0, 1]`
- 含义：与训练一致，用于构造 ACP 中依赖的 value target（确保口径一致）。

对应代码：

- `src/lerobot/configs/value.py`
- `src/lerobot/scripts/lerobot_value_infer.py`

## 5. 使用方法

## 5.1 训练时指定 q（示例）

```bash
lerobot-value-train \
  --config_path <your_value_train_config.json> \
  --targets.length_scale_quantile 0.8
```

如果想更强抑制长尾影响，可尝试更小的 q（如 `0.7`）；  
如果想保留更多尾部差异，可增大 q（如 `0.9`）。

## 5.2 推理/ACP 时保持一致（示例）

```bash
lerobot-value-infer \
  --config_path <your_value_infer_config.json> \
  --acp.length_scale_quantile 0.8
```

建议训练与推理使用同一个 `q`，避免目标定义不一致带来的分布偏移。

## 6. 参数调优建议

- `q=0.8`：默认推荐，适合明显长尾任务；
- `q=0.9`：更接近旧逻辑，保留更多长时长差异；
- `q=0.7`：进一步提升主体样本分辨率，但尾部区分更粗；
- `c_fail_coef`：失败惩罚系数，保持原语义不变，仍可独立调节。

## 7. 注意事项

- `q` 必须在 `(0,1]`，否则配置校验会报错；
- task 内必须有有效正长度 episode，才能计算分位数；
- 当前尾部策略为“硬截断”，不是软压缩（如 `log1p`）；
- 该改动已同步训练与推理路径，避免出现“两套 target 口径”。

## 8. 相关代码入口

- 目标值计算：`src/lerobot/values/pistar06/modeling_pistar06.py`
  - `compute_normalized_value_targets`
  - `Pistar06Policy._compute_task_length_scales`
- 训练配置：`src/lerobot/configs/value_train.py`
  - `ValueTargetsConfig.length_scale_quantile`
- 推理/ACP 配置：`src/lerobot/configs/value.py`
  - `ValueInferenceACPConfig.length_scale_quantile`
- 推理流程：`src/lerobot/scripts/lerobot_value_infer.py`
  - `_build_episode_info` 中 task 尺度计算

## 9. 运行命令示例

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/mnt/data2/whs/Evo-RL/src:$PYTHONPATH python -m lerobot.scripts.lerobot_value_train \
  --dataset.root=/mnt/data/dataset/lerobot/arx_x5_single_demonstrations_slipper \
  --dataset.repo_id=arx_x5_single_demonstrations_slipper \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --targets.length_scale_quantile=0.8 \
  --output_dir=outputs/value_train/arx_slipper_run2_pistar06_frozen_q80 \
  --job_name=arx_slipper_v2_pistar06_frozen_q80 \
  --wandb.enable=true \
  --wandb.disable_artifact=true
```
