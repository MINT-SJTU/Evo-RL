# PI05 Backbone for Pistar06 Value

本文说明如何把 `lerobot/pi05_base` 中已经过机器人数据训练的 PaliGemma 骨干接入当前 `pistar06` value 训练框架。这里接入的是：

- `SigLIP` 视觉塔权重
- `Gemma` 语言模型权重

注意：这里使用的是 `pi05_base` 里的 `PaliGemma` 主干，不会把 `pi05` 的 action expert Gemma 接到 value head 上。

## 改动内容

当前代码新增了以下能力：

- `value.backbone_source=pi05` 时，从 `lerobot/pi05_base` 自动下载并加载机器人训练后的 `SigLIP + Gemma` 主干。
- 保留原有 `value.backbone_source=hf` 路径，默认行为不变。
- 新增 `value.tokenizer_repo_id`，在 `pi05` 模式下默认使用 `google/paligemma-3b-pt-224` tokenizer。
- 继续复用现有冻结开关：
  - `value.freeze_vision_encoder`
  - `value.freeze_language_model`

## 重要说明

你的 `evo-rl` 环境里默认解析到的 `lerobot` 不是当前工作区源码，而是另一个已安装包路径。为了确保命中本次修改，建议显式指定当前仓库的 `src`：

```bash
PYTHONPATH=/mnt/data3/whs/Evo-RL/src:$PYTHONPATH python -m lerobot.scripts.lerobot_value_train
```

不要继续直接依赖 `$(which lerobot-value-train)`，否则可能运行到旧包。

## 权重下载

第一次运行 `value.backbone_source=pi05` 时，代码会自动调用 `huggingface_hub.snapshot_download(...)` 下载 `lerobot/pi05_base` 到本机 Hugging Face cache。

如果你想提前下载，可以手动执行：

```bash
source ~/.bashrc
conda activate evo-rl
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/pi05_base')"
```

## 关键参数

新增或建议关注的参数如下：

- `--value.backbone_source=pi05`
  - 使用 `pi05_base` 中的机器人训练骨干。
- `--value.pi05_repo_id=lerobot/pi05_base`
  - 指定 PI05 权重来源，可以是 Hugging Face repo，也可以是本地目录。
- `--value.pi05_revision=<revision>`
  - 可选，指定具体 revision。
- `--value.tokenizer_repo_id=google/paligemma-3b-pt-224`
  - 可选。`pi05` 模式下默认就是这个值。
- `--value.freeze_vision_encoder=true|false`
  - 是否冻结 SigLIP。
- `--value.freeze_language_model=true|false`
  - 是否冻结 Gemma。

## 推荐运行方式

下面的命令都基于你给出的原始训练命令做最小改动。

### 1. 冻结 SigLIP 和 Gemma

适合先训练 value head / projector，显存更稳，收敛也更可控。

```bash
source ~/.bashrc
conda activate evo-rl

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data3/whs/Evo-RL/src:$PYTHONPATH \
python -m lerobot.scripts.lerobot_value_train \
  --dataset.root=/mnt/data/dataset/lerobot/arx_x5_single_demonstrations_slipper \
  --dataset.repo_id=arx_x5_single_demonstrations_slipper \
  --value.type=pistar06 \
  --value.backbone_source=pi05 \
  --value.pi05_repo_id=lerobot/pi05_base \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --output_dir=outputs/value_train/arx_slipper_run1_pi05_frozen \
  --job_name=arx_slipper_v1_pi05_frozen \
  --wandb.enable=true \
  --wandb.disable_artifact=true
```

### 2. 不冻结 SigLIP 和 Gemma

适合在数据量足够、显存允许时做全量端到端微调。

```bash
source ~/.bashrc
conda activate evo-rl

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data3/whs/Evo-RL/src:$PYTHONPATH \
python -m lerobot.scripts.lerobot_value_train \
  --dataset.root=/mnt/data/dataset/lerobot/arx_x5_single_demonstrations_slipper \
  --dataset.repo_id=arx_x5_single_demonstrations_slipper \
  --value.type=pistar06 \
  --value.backbone_source=pi05 \
  --value.pi05_repo_id=lerobot/pi05_base \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=false \
  --value.freeze_language_model=false \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --output_dir=outputs/value_train/arx_slipper_run1_pi05_unfrozen \
  --job_name=arx_slipper_v1_pi05_unfrozen \
  --wandb.enable=true \
  --wandb.disable_artifact=true
```

## 可选：继续使用 accelerate

如果你仍然希望保留 `accelerate launch` 形式，也可以这样写，核心是把入口改成当前工作区源码：

```bash
source ~/.bashrc
conda activate evo-rl

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/mnt/data3/whs/Evo-RL/src:$PYTHONPATH \
accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --num_machines=1 \
  -m lerobot.scripts.lerobot_value_train \
  --dataset.root=/mnt/data/dataset/lerobot/arx_x5_single_demonstrations_slipper \
  --dataset.repo_id=arx_x5_single_demonstrations_slipper \
  --value.type=pistar06 \
  --value.backbone_source=pi05 \
  --value.pi05_repo_id=lerobot/pi05_base \
  --value.dtype=bfloat16 \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --value.push_to_hub=false \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --output_dir=outputs/value_train/arx_slipper_run1_pi05_frozen \
  --job_name=arx_slipper_v1_pi05_frozen \
  --wandb.enable=true \
  --wandb.disable_artifact=true
```

把最后两行冻结参数改成 `false`，就是不冻结版本。

## 实现细节

`value.backbone_source=pi05` 时，代码会：

1. 下载或读取 `lerobot/pi05_base`。
2. 构建 `PI05Policy`。
3. 从 `pi05_base` 的 `paligemma` 主干中抽取：
   - vision: `SigLIP`
   - language: `Gemma`
4. 把它们适配到当前 `Pistar06Model` 的双塔 value 框架：
   - 图像分支仍然走当前 value 的 image pooling + projector
   - 文本分支仍然走当前 value 的 language pooling + projector
   - 最后的 `value_head` 结构保持不变

因此这次改动属于“把机器人训练过的 backbone 缝到现有 value head 上”，不是把整个 `pi05` policy 直接替换成 value model。

## 已验证内容

已通过以下定向测试：

```bash
source ~/.bashrc
conda activate evo-rl
PYTHONPATH=/mnt/data3/whs/Evo-RL/src:$PYTHONPATH python -m pytest \
  tests/value/test_pistar06_configuration.py \
  tests/value/test_pistar06_value_stack.py
```

测试覆盖了：

- `pi05` 模式下 tokenizer 默认值
- `pi05` 骨干接入后的前向路径
- 冻结开关仍然生效
- 原有 `pistar06` 保存/加载逻辑未被破坏
