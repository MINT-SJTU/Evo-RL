# AC 训练夜间迭代实验记录

> 日期: 2026-04-11 ~ 04-12
> 目标: 对照论文实现关键 trick（stride-2, chunk-level bootstrap discount），在新数据集上系统搜索 actor-critic 最优配置
> 约束: 不动 RL token 和 VLA 基座模型，只优化 AC 训练

## 背景

### 已知问题

1. **actual_steps 错误**: `_encoded_to_transitions()` 中 `actual_steps=chunk_length=10`，但论文要求 `actual_steps=stride`。当 stride=2 时，bootstrap discount 从 γ^10=0.90 变为 γ^2=0.98，Q 值传播效率大幅提升。
2. **数据集损坏**: 旧 478ep 数据集 53% 视频损坏，新版已上传 HuggingFace。
3. **β 过高**: 当前 β=5.0 是为了防止旧 bug 下的 Q 值爆炸。修复 bootstrap discount 后，Q-learning 信号增强，β 可能需要降低。

### 论文关键 trick

1. **Stride-2 Subsampling**: 每 2 步存一个 transition，next_state 间隔 2 步
2. **Chunk-level MDP Bootstrap**: γ^stride（非 γ^C）作为 bootstrap discount
3. **Reference Action Dropout**: 50% 概率将 ref_chunk 置零（已实现）
4. **BC Regularization**: β 控制 actor 偏离 VLA 参考的程度

### 代码修改

- `offline_dataset.py`: `_encoded_to_transitions()` 新增 `stride` 参数，`actual_steps=stride`
- `build_rlt_offline_cache.py`: `--frame-stride` 默认值从 1 改为 2，传递 stride 到 transition 构建

---

## 数据集

| 名称 | 来源 | Episodes | 说明 |
|------|------|----------|------|
| 新版 478ep | `Shiki42/271ep_sft_success_critical_phase_478ep` | 478 | 修复视频损坏问题 |

---

## 实验记录

### Cache 构建

| Cache ID | 数据集 | Stride | actual_steps | Transitions (train/val/test) | 耗时 |
|----------|--------|--------|-------------|------------------------------|------|
| (待填) | | | | | |

### AC 训练实验

| Exp ID | Cache | β | Steps | ref_mse | Q_policy | Q_expert | Q_gap | TD_err | critic_loss | 备注 |
|--------|-------|---|-------|---------|----------|----------|-------|--------|-------------|------|
| (待填) | | | | | | | | | | |

---

## 发现与结论

(迭代过程中持续更新)
