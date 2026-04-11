# record_rlt_hil.py 调试修���记录

日期: 2026-04-11
机器: zhaobo-4090-1 (192.168.31.100), RTX 4090 24GB, conda env `evo-rl`

## 目标

在 machine 100 上跑通:
```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_rlt_hil.py \
    --vla-model /home/zhaobo-4090-1/models/pi05_screw_271ep_sft_fp32 \
    --rl-token-ckpt checkpoints/rlt_271ep_sft/demo_adapt_checkpoint.pt \
    --ac-ckpt checkpoints/rlt_278ep_sft/rl_checkpoint.pt \
    --task "Insert the copper screw into the black sleeve." \
    --num-episodes 1 --no-teleop
```

## 修复 1: dataset 命名加 `eval_` 前缀

**文件**: `scripts/record_rlt_hil.py:148`

**问题**: lerobot 的 `sanity_check_dataset_name()` 要求使用 policy 时 dataset 名必须以 `eval_` 开��，否则抛 `ValueError`。

**修复**: `rlt_hil_{time}` → `eval_rlt_hil_{time}`

## 修复 2: PI05Config 加载时 `type` 字段报错

**文件**: `src/lerobot/policies/rlt/modeling_rlt.py` — 新增 `_load_pi05_config()`

**问题**: SFT 模型的 `config.json` 包含 `"type": "pi05"` 字段，但 `PI05Config` 是具��子��（不是 `ChoiceRegistry` 基类），draccus 严格解析时拒绝未知字段：
```
DecodingError: The fields `type` are not valid for PI05Config
```

**修复**: 新增 `_load_pi05_config()` — 手动加载 config.json、移除 `type` 字段后写入临时文件，再交给 draccus 解析。替换��来的 `PI05Config.from_pretrained()` 调用。

## 修复 3: SFT checkpoint 缺少 `embed_tokens.weight`

**文件**: `src/lerobot/policies/rlt/modeling_rlt.py` — 新增 `_tie_embed_tokens()`, `_ensure_pi05()` 改用 `strict=False`

**问题**: SFT 模型的 safetensors 只存 `lm_head.weight`，不单独存 `embed_tokens.weight`（二者是 tied weight）。`load_state_dict(strict=True)` 报 missing key，异常被 PI05Policy.from_pretrained 的 `except Exception` 吞掉后，���个 state_dict 未加载。

**实际发现**: transformers 在模型初始化时已自动绑定 `embed_tokens.weight` 和 `lm_head.weight`（同一 data_ptr），所以 `strict=False` 时 missing key 不影响正确性。`_tie_embed_tokens()` 作���安全兜底保留。

## 修复 4: PrefixOutputCapture hook 不触发���核心问题）

**文件**: `src/lerobot/policies/rlt/action_modifier.py` — 重写 `PrefixOutputCapture`

**问题**: PI05 的 `sample_actions()` 直接调用 `self.paligemma_with_expert.forward(...)` 而非 `self.paligemma_with_expert(...)`。PyTorch 的 `register_forward_hook` 只在 `__call__` 时触发，直接调用 `.forward()` 不触发 hook。

**验证**:
```python
# .forward() — hook 不触发
pwm.forward(inputs_embeds=[prefix_embs, None])  # Captured: False

# __call__ — hook 触发
pwm(inputs_embeds=[prefix_embs, None])           # Captured: True
```

**修复**: 将 `register_forward_hook` 改为 monkey-patch `forward` 方法���在 patched forward 中调用原始 forward 后截取 prefix output �� pool 到 (B, 64, 2048)。

## 环境准备: checkpoint symlink

machine 100 上无 RLT checkpoint，通过 HF 镜像下载后建立 symlink:
```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Shiki42/rlt_pi0.5_screw', 'rl_token/271ep_pi0.5_screw_sft_rltoken/demo_adapt_checkpoint.pt')
hf_hub_download('Shiki42/rlt_pi0.5_screw', 'actor_critic/278ep0411/rl_checkpoint.pt')
"
# symlink: checkpoints/rlt_271ep_sft/ �� HF cache
# symlink: checkpoints/rlt_278ep_sft/ → HF cache
```

## 变更���件汇总

| 文件 | 改动 |
|------|------|
| `scripts/record_rlt_hil.py` | dataset 名加 `eval_` 前缀 |
| `src/lerobot/policies/rlt/modeling_rlt.py` | +`_load_pi05_config()`, +`_tie_embed_tokens()`, `_ensure_pi05()` 改用 strict=False |
| `src/lerobot/policies/rlt/action_modifier.py` | `PrefixOutputCapture` 从 forward_hook 改为 monkey-patch forward |
