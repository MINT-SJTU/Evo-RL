# Evo1 原始实现继承问题

这份文档记录的是：在 Evo-RL 接入 Evo1 过程中发现、但对照原始 Evo1 仓库后确认**并非迁移新引入**，而是原始 Evo1 本身已有的实现问题或工程限制。

## 1. `get_action()` 中 `action_mask` 判空顺序不安全

- 现象：先 `action_mask.view(...)`，后做 `None` 检查。
- 影响：如果外部传入 `None`，会先触发异常，后面的报错分支成为死代码。
- 归因：原始 Evo1 即如此实现。
- 建议：将 `None` 检查提前。

## 2. masked loss / target velocity 在一般 mask 场景下不严谨

- 当前写法：
  - `target_velocity = actions_gt - noise`
  - `loss = MSE(pred_velocity * mask, target_velocity)`
- 问题：
  - 如果 mask 不只是 padding mask，而是更一般的稀疏动作维度 mask，那么被 mask 掉的维度仍可能贡献非零 target。
- 归因：原始 Evo1 即如此实现。
- 备注：
  - 当前单 embodiment、padding-only 的常见场景下通常不明显触发。
- 建议：
  - 将 mask 同时应用到 `(pred_velocity - target_velocity)` 或显式 mask `target_velocity`。

## 3. forward 中动态创建 `single_action_proj`

- 现象：在 `forward/get_action` 中按需创建 `nn.Linear`。
- 影响：
  - 不利于 checkpoint / 分布式训练 / 参数集合稳定性。
- 归因：原始 Evo1 即如此实现。
- 建议：在 `__init__` 中定义完整模块。

## 4. InternVL3 层数硬编码为 14

- 现象：语言模型层被硬截断到 14 层。
- 影响：
  - 仅适配当前使用的特定 InternVL3 配置；
  - 切换不同规模模型时可迁移性较差。
- 归因：原始 Evo1 即如此实现。
- 建议：改为配置项。

## 5. VLM forward 为逐 sample 串行

- 现象：batch 内逐样本调用 VLM。
- 影响：
  - 训练吞吐较低，尤其 stage2 更明显。
- 归因：原始 Evo1 即如此实现。
- 建议：后续如有性能优化需求，可考虑 batched VLM forward。

## 6. `use_flash_attn=True` 与 `torch.bfloat16` 硬编码

- 现象：直接写死。
- 影响：
  - 环境兼容性受限。
- 归因：原始 Evo1 即如此实现。
- 建议：改为配置项。

## 7. 配置链较绕

- 现象：`dict -> SimpleNamespace` 形式在模型层继续传递配置。
- 影响：
  - 配置可读性和可追踪性一般。
- 归因：原始 Evo1 即如此实现。
- 建议：后续可考虑直接传更强约束的配置对象。

## 8. 说明

以上问题是 Evo-RL 接入时发现的“原始实现继承问题”，不等同于本次迁移引入的行为不一致。

当前 Evo-RL 接入侧会优先修复迁移层语义问题；原始 Evo1 本体问题建议由 Evo1 源实现维护侧统一评估和处理。
