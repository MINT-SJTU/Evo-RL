# Evo1 原始实现继承问题

这份文档记录的是：在 Evo-RL 接入 Evo1 过程中发现、但对照原始 Evo1 仓库后确认**并非迁移新引入**，而是原始 Evo1 本身已有的实现问题或工程限制。

## 已在当前 Evo-RL 接入层修复

下面这些问题最初来自原始 Evo1，但已经在当前 Evo-RL 端口里修掉，不再属于当前分支的残留问题：

### 1. `get_action()` 中 `action_mask` 判空顺序不安全

- 原始 Evo1：先 `action_mask.view(...)`，后做 `None` 检查。
- 当前端口：已改成先判空，再处理 shape/view。

### 2. masked loss / target velocity 在一般 mask 场景下不严谨

- 原始 Evo1：
  - `target_velocity = actions_gt - noise`
  - `loss = MSE(pred_velocity * mask, target_velocity)`
- 当前端口：
  - loss 改成先算 `(pred_velocity - target_velocity)` 再乘 mask；
  - `target_velocity` 和训练分支里的 `actions_gt_seq` 也会显式跟随 mask。

### 3. `forward/get_action` 中动态创建 `single_action_proj`

- 原始 Evo1：按需在前向里构造 `nn.Linear`。
- 当前端口：改成在 `__init__` 中一次性定义，避免参数集合在运行时变化。

### 4. InternVL3 层数硬编码为 14

- 原始 Evo1：语言模型层数硬编码截断到 14。
- 当前端口：改成配置项 `vlm_num_layers`，默认仍保持 14。

### 5. `use_flash_attn=True` 与 `torch.bfloat16` 硬编码

- 原始 Evo1：直接写死。
- 当前端口：改成配置项 `use_flash_attn` 和 `vlm_dtype`。

### 6. 其他已顺手修掉的工程性问题

- 删除了原始 Evo1 里永假的 `action_dim != horizon * per_action_dim` 检查。
- `InternVL3Embedder` 不再把 `device` 固定保存成字符串语义，而是运行时从模型参数读取实际 device。

## 当前仍保留的原始 Evo1 遗留问题

下面这些问题在当前 Evo-RL 端口里仍然基本沿用原始 Evo1 的实现，后续如果要继续打磨，建议由 Evo1 本体维护侧统一评估：

### 1. VLM forward 仍为逐 sample 串行

- 现象：batch 内逐样本调用 VLM。
- 影响：
  - 训练吞吐较低，尤其 stage2 更明显。
- 归因：原始 Evo1 即如此实现。
- 建议：后续如有性能优化需求，可考虑 batched VLM forward。

### 2. 配置链较绕

- 现象：`dict -> SimpleNamespace` 形式在模型层继续传递配置。
- 影响：
  - 配置可读性和可追踪性一般。
- 归因：原始 Evo1 即如此实现。
- 建议：后续可考虑直接传更强约束的配置对象。

## 说明

以上问题是 Evo-RL 接入时发现的“原始实现继承问题”，不等同于本次迁移引入的行为不一致。

当前 Evo-RL 接入侧已经优先修掉了会影响训练正确性或接口健壮性的部分；剩余项更多是性能与工程可维护性问题。
