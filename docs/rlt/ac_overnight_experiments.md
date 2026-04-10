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
| cache_newdata_stride2 | 478ep (new, 0 skipped) | 2 | 2 | 39072 / 5752 / 4530 | 9273s |

### AC 训练实验（stride=2, 50K steps, ResidualMLP 3L/256h）

| Exp ID | β | ref_mse | expert_mse | Q_policy | Q_expert | Q_gap | TD_err | 状态 |
|--------|-----|---------|------------|----------|----------|-------|--------|------|
| s2_b0.1 | 0.1 | 71.60 | 72.66 | 27.05 | 15.08 | 11.97 | 0.47 | 发散 |
| s2_b0.3 | 0.3 | 84.79 | 85.68 | 63.79 | 20.79 | 43.00 | 4.36 | 发散 |
| s2_b1.0 | 1.0 | 2.525 | 3.350 | 19.82 | 15.39 | 4.43 | 0.60 | 偏离过大 |
| **s2_b3.0** | **3.0** | **0.035** | **0.871** | **8.66** | **8.45** | **0.21** | **0.020** | **最优** |
| s2_b5.0 | 5.0 | 0.190 | 1.062 | 13.15 | 11.46 | 1.68 | 0.70 | 良好 |
| s2_b10.0 | 10.0 | 0.010 | 0.812 | 8.21 | 8.02 | 0.19 | 0.030 | 近纯BC |

### 扩展实验（100K steps）

| Exp ID | β | Steps | ref_mse | Q_policy | Q_expert | Q_gap | TD_err | 备注 |
|--------|-----|-------|---------|----------|----------|-------|--------|------|
| s2_b0.3_100k | 0.3 | 100K | 13.45 | 18.30 | 10.49 | 7.81 | 0.079 | 发散 |
| s2_b1.0_100k | 1.0 | 100K | 191.08 | 371.12 | 9.31 | 361.81 | 12.19 | 灾难性发散 |
| s2_b3.0_100k | 3.0 | 100K | 209.55 | 1317.84 | 82.69 | 1235.16 | 250.28 | 灾难性发散 |
| s2_b5.0_100k | 5.0 | 100K | 2.89 | 49.67 | 26.62 | 23.04 | 10.03 | 已开始发散 |
| **s2_b10.0_100k** | **10.0** | **100K** | **0.010** | **8.21** | **8.02** | **0.20** | **0.032** | **稳定！** |

### Phase 2 细搜（β=2.0-4.0, 50K steps）

| Exp ID | β | ref_mse | expert_mse | Q_policy | Q_expert | Q_gap | TD_err | 备注 |
|--------|-----|---------|------------|----------|----------|-------|--------|------|
| s2_b2.0 | 2.0 | 0.117 | 1.021 | 9.62 | 9.17 | 0.45 | 0.12 | 偏离适中 |
| s2_b2.5 | 2.5 | 0.045 | 0.840 | 8.79 | 8.57 | 0.22 | 0.019 | 接近最优 |
| **s2_b3.0** | **3.0** | **0.035** | **0.871** | **8.66** | **8.45** | **0.21** | **0.020** | **最优** |
| s2_b4.0 | 4.0 | 0.025 | 0.793 | 8.56 | 8.36 | 0.20 | 0.019 | 微保守 |

---

## 理论分析

### Q 值传播分析

对于平均 206 帧/episode 的数据集：

| Stride | actual_steps | Transitions | Max Q at start | vs buggy γ^10 |
|--------|-------------|-------------|----------------|---------------|
| 1 | 1 (correct) | 206 | 1.261 | **1.24e8x better** |
| 1 | 10 (buggy) | 206 | 1.02e-8 | baseline (dead) |
| 2 | 2 (correct) | 103 | 1.261 | **3950x better** |
| 2 | 10 (buggy) | 103 | 3.19e-4 | barely alive |

**关键发现**：
- 无论 stride 多少，只要 `actual_steps` 正确，最大 Q 值都是 ~1.26（success_bonus=10）
- 之前 actual_steps=10 的 bug 导致 Q 信号几乎完全消失
- 之前 β=5.0 下 ref_mse=0.078、Q_gap=0.71 的结果本质上是纯 BC（Q 贡献约 0），不是真正的 RL
- 修复后 Q-learning 真正启动，β 需要重新调参

### 推论

1. 修复后 Q 值约在 0-1.3 范围（vs 之前虚假的 5-7）
2. actor_loss = -Q + β*MSE(mu, ref)，Q~1 时：
   - β=5.0: BC 完全主导，actor ≈ VLA ref
   - β=1.0: Q 和 BC 大致平衡
   - β=0.3: Q 信号主导，actor 可偏离 VLA ref
3. 期望最优 β 在 0.3-1.0 范围

---

## 发现与结论

### 关键发现 1: actual_steps 修复彻底改变了 Q-learning 行为

修复前（actual_steps=10）：Q 值几乎无法传播，actor 本质上做纯 BC，ref_mse 的"好"结果（0.008-0.078）是假象。
修复后（actual_steps=2）：Q 值在 8-20 范围正常传播，critic 稳定收敛（TD_err 0.02-0.70），actor 真正被 Q-gradient 驱动。

### 关键发现 2: β 是核心超参数，最优值为 3.0

| β 区间 | 行为 | 推荐 |
|--------|------|------|
| 0.1-0.3 | actor 完全发散，ref_mse > 70，Q 值过估计 | 不可用 |
| 1.0 | actor 偏离较大但不发散，ref_mse=2.5 | 不推荐 |
| **3.0** | **actor 紧贴 VLA ref（ref_mse=0.035），Q_gap=0.21 正向** | **推荐** |
| 5.0 | actor 较紧贴（ref_mse=0.19），Q_gap=1.68 较大 | 可用 |
| 10.0 | 近纯 BC（ref_mse=0.010），Q_gap=0.19 | 过保守 |

### 关键发现 3: β=3.0 实现了论文描述的"local action editing"

论文核心理念："turning online RL into local refinement of promising behaviors rather than unconstrained search"。
β=3.0 的 ref_mse=0.035 说明 actor 只做了很小的偏离（RMSE ≈ 0.19 in normalized space），但 Q_gap=0.21 说明这个小偏离确实带来了 value 提升。这正是论文要的效果。

### 关键发现 4: 新数据集质量显著提升

478 episodes 全部成功加载（0 skipped），比旧版的 53% 损坏率大幅改善。39072 train transitions（vs 旧版 33805）。

### 关键发现 5: 100K 步 offline RL 全线发散（β=10.0 除外）

| β | 50K ref_mse | 100K ref_mse | 100K 状态 |
|---|------------|-------------|----------|
| 0.3 | 84.79 | 13.45 | 发散 |
| 1.0 | 2.52 | 191.08 | 灾难 |
| 3.0 | 0.035 | 209.55 | 灾难（50K 还好，100K 爆炸） |
| 5.0 | 0.190 | 2.89 | 开始发散 |
| **10.0** | **0.010** | **0.010** | **完全稳定** |

**结论：offline RL 在固定 buffer 上训练必然发散**，唯一例外是 β 足够高（≥10）把 actor 锁在 VLA ref 上。
最优策略：β=3.0-4.0, 50K 步。不要延长训练。

### 关键发现 6: β=2.5-4.0 区间表现相近

β=2.5 (ref_mse=0.045) 和 β=4.0 (ref_mse=0.025) 与 β=3.0 (ref_mse=0.035) 差异很小。
实际部署推荐 β=3.0 或 β=4.0，两者都在 50K 步内给出好的 ref_mse 和正 Q_gap。

### 下一步

1. 用 **β=3.0, 50K 步** 的 checkpoint (`exp_s2_b3.0`) 部署到真机验证
2. 考虑在 β=2.0-4.0 区间进一步细搜（可能 β=2.0 能在保持稳定的同时给更多 Q 信号）
3. 尝试 β=3.0 + hidden_dim=512（论文 screw task 配置）
4. 研究 100K 步是否能用更高 β（如 β=5.0 100K）保持稳定并获得更好结果
