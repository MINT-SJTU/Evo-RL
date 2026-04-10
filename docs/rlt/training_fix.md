# Offline RL 训练 Bug：Action/State 空间不匹配

## 问题发现经过

### 起因：部署时机械臂剧烈运动

在 Machine 100 (4090) 上部署 RLT 策略时，发现两个现象：
- **不加 unnormalize**：机械臂固定在中位（action ~0 被当作角度 → ~0°）
- **加 unnormalize**：机械臂剧烈大幅运动

### 定位过程

1. **对比 VLA vs RLT 输出**：在 eval 数据集上跑 `visualize_vla_actions.py`，同时运行 VLA 和 RLT inference

   | 模型 | 原始输出范围 | unnormalize 后范围 |
   |------|------------|-------------------|
   | GT (ground truth) | — | [-107°, 103°] |
   | VLA (pi0.5) | [-1.03, 1.26] | [-117°, 102°] ✓ |
   | **RLT actor** | **[-2.08, 4.06]** | **[-282°, 188°]** ✗ |

   Actor 输出超出 [-1, 1] 范围，unnormalize 后放大到了合理角度范围的 2-3 倍。

2. **逐帧诊断**：对 5 个采样帧做 actor vs VLA 对比

   ```
   Frame     0: actor mu range [-1.96, 4.08], ref vs mu MSE = 1.18 (正常应 < 0.05)
   Frame   500: actor mu range [-1.25, 1.22], ref vs mu MSE = 0.05 (正常)
   Frame  1000: actor mu range [-2.35, 4.28], ref vs mu MSE = 0.47
   ```

   某些帧 actor 输出正常，某些帧偏差巨大。38% 的输出维度超出 [-1, 1]。

3. **追溯训练数据**：检查 `offline_dataset.py` 的 cache 构建流程，发现 `exec_chunk`（专家动作）和 `ref_chunk`（VLA 参考动作）的空间不一致。

## Bug 1：Action 空间不匹配

### 根因

训练时 `exec_chunk` 和 `ref_chunk` 在 **不同的空间**：

| 组件 | 来源 | 空间 | 典型范围 |
|------|------|------|---------|
| `exec_chunk` | LeRobot 数据集 `action` 字段 | 原始角度 | [-103°, 97°] |
| `ref_chunk` | VLA `sampled_action_chunk` | 归一化 | [-1, 1] |

影响链路：
1. **Critic** 从 `exec_chunk` 学到 `Q(state, 角度空间 action)`
2. **Actor** 被 BC 正则化拉向 `ref_chunk`（归一化空间），即 `actor_loss = -Q + β * MSE(mu, ref)`
3. **Target Q** 用 actor 输出（≈归一化）去评估，但 critic 期望角度空间 → out-of-distribution

结果：Q-gradient 对 actor 来说是噪声。Actor 仅靠 BC 项收敛，带有随机扰动。推理时输出大致在 [-1, 1] 但有频繁离群值。

### Bug 位置

`offline_dataset.py:build_transitions_from_demos()`:

```python
e = _subsample_chunk(expert_actions[i], chunk_length)  # expert_actions 在角度空间
encoded.append((s, r, e, rew))  # r (ref_chunk) 在归一化空间
```

`expert_actions` 来自 `RLTDemoDataset.__getitem__()`，直接读取 LeRobot 数据集的 `action` 字段（角度空间）。而 `ref_chunk` 来自 VLA 的 `sampled_action_chunk`（归一化空间）。

### 修复

在 `demo_loader.py` 的 `RLTDemoDataset` 中增加 `normalize_actions=True` 参数，使用数据集的 q01/q99 统计量将 `expert_actions` 归一化到 [-1, 1]：

```python
# 角度 → [-1, 1]
normalized = (degrees - q01) / (q99 - q01) * 2.0 - 1.0
```

所有 cache 构建脚本（`build_rlt_offline_cache.py`、`build_cp_offline_cache.py`、`train_best_ac_on_sft.py`）均传入 `normalize_actions=True`。

同时在 `deploy.py` 的 `_rl_action_chunk()` 中增加 `clamp(-1, 1)` 作为安全兜底。

## Bug 2：State/Action 尺度不匹配导致 Q 值爆炸

### 发现过程

修复 Bug 1 后重建 cache 并训练，结果 critic loss 从初始 ~12 爆炸到 1e34，actor loss 到 -5e16。最终 eval 指标全部 1e17+。

### 根因

修复 Bug 1 后，action 归一化到 [-1, 1]（norm ~5），但 proprio 仍然是原始角度（norm ~200）。Critic 的输入中 action 占比从修复前的 73%（550/750）降到 2.4%（5/205），critic 几乎看不到 action 的差异 → Q 值失去对 action 的区分能力 → TD bootstrap 无约束爆炸。

| 输入 | Bug 1 修复前（角度 action） | Bug 1 修复后（归一化 action） |
|------|--------------------------|---------------------------|
| z_rl | norm ~10 | norm ~10 |
| proprio | norm ~200 | norm ~200 |
| action | norm ~550 | **norm ~5** |
| action/state 比例 | 2.75 | **0.025** |

### 修复

在 `demo_loader.py` 中同时归一化 proprio（使用 `observation.state` 的 q01/q99）：

```python
if self._normalize_actions and self._state_q01 is not None:
    state = normalize_quantiles(state, self._state_q01, self._state_q99)
```

归一化后的尺度：
- state: z_rl (norm ~10) + proprio_normalized (norm ~1.7) → total ~10
- action: norm ~5
- action/state 比例: 0.5（健康）

同步修改了 `obs_bridge.py` 和 `deploy.py`，使推理时也传入 proprio_q01/q99 进行归一化，保持训练-推理一致。

## 完整修改清单

| 文件 | 修改内容 | Commit |
|------|---------|--------|
| `demo_loader.py` | 新增 `normalize_quantiles()` 函数；`RLTDemoDataset` 增加 `normalize_actions` 参数，同时归一化 action 和 proprio | 5f55475, bf23666 |
| `deploy.py` | `_rl_action_chunk()` 增加 `clamp(-1, 1)`；`_fill_action_queue()` 传入 proprio_q01/q99 | 5f55475, bf23666 |
| `deploy_config.py` | 增加 `proprio_q01`/`proprio_q99` 字段 | bf23666 |
| `obs_bridge.py` | `robot_obs_to_rlt_obs()` 增加 `proprio_q01`/`proprio_q99` 参数 | bf23666 |
| `build_rlt_offline_cache.py` | 传入 `normalize_actions=True`；增加 corrupt episode skip | 5f55475, 24d2cd7 |
| `build_cp_offline_cache.py` | 传入 `normalize_actions=True` | 5f55475 |
| `train_best_ac_on_sft.py` | 传入 `normalize_actions=True` | 5f55475 |
| `offline_dataset.py` | `precompute_offline_buffer()` 传入 `normalize_actions=True` | 5f55475 |
| `configs/ac_best_v3.yaml` | ResidualMLP 3L/256h, β=0.3, 50K steps | 5d0ca41 |

## 验证方法

### 1. Cache 验证

```python
data = torch.load("transitions_train.pt", weights_only=False)
t = data[0]
print("exec_chunk range:", t["exec_chunk"].min(), t["exec_chunk"].max())   # 应在 [-1, 1]
print("ref_chunk range:", t["ref_chunk"].min(), t["ref_chunk"].max())       # 应在 [-1, 1]
print("state_vec norm:", t["state_vec"].norm())                             # 应 ~10，不是 ~200
print("state_vec[-12:]:", t["state_vec"][-12:])                             # proprio 应在 [-1, 1]
```

### 2. 训练验证

训练过程中 critic loss 应稳定在 <100，不应爆炸到 1e10+。最终 eval 指标：
- `ref_mse` < 0.05（actor 输出接近 VLA 参考）
- `Q_gap` 在 [-0.1, 0] 范围（Q 值健康）

### 3. 部署验证

在 eval 数据集上运行 `visualize_vla_actions.py`，RLT 输出应在 [-1, 1] 内无需 clamp，unnormalize 后 MAE(VLA, RLT) < 10°。

## 影响

- 所有已有 actor-critic checkpoint（`v2_base`、`v2_sft`、`v1_baseline`）都在 bug 下训练，**不可复用**，必须用修复后的 pipeline 重新训练
- RL token checkpoint（`v8_base`、`v1_sft`）**不受影响**——只使用 VLA hidden states，不涉及 action 空间
- 478ep 数据集有 53% 的 episode 视频损坏，已通过 skip 机制跳过（24d2cd7）
