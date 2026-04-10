# Offline RL 训练 Bug 修复记录

> 本文档记录了部署 RLT 策略时发现的训练 pipeline bug、逐步定位与修复过程、以及尚未实现的论文关键 trick。用于清空上下文后继续迭代。

## 一、问题发现经过

### 起因：部署时机械臂剧烈运动

在 Machine 100 (4090) 上部署 RLT 策略时，发现两个现象：
- **不加 unnormalize**：机械臂固定在中位（action ~0 被当作角度 → ~0°）
- **加 unnormalize**：机械臂剧烈大幅运动

### 定位过程

1. **对比 VLA vs RLT 输出**：在 eval 数据集 `0408_271ep_sft/eval_autonomy_132556`（12043帧）上跑 `visualize_vla_actions.py`，同时运行 VLA 和 RLT inference，stride=10

   | 模型 | 原始输出范围 | unnormalize 后范围 |
   |------|------------|-------------------|
   | GT (ground truth) | — | [-107°, 103°] |
   | VLA (pi0.5) | [-1.03, 1.26] | [-117°, 102°] ✓ |
   | **RLT actor** | **[-2.08, 4.06]** | **[-282°, 188°]** ✗ |

   Actor 输出超出 [-1, 1] 范围，unnormalize 后放大到了合理角度范围的 2-3 倍。

2. **逐帧诊断**：对 5 个采样帧做 actor 的 `mu` 输出 vs VLA `ref_chunk` 对比

   ```
   Frame     0: actor mu range [-1.96, 4.08], ref vs mu MSE = 1.18 (正常应 < 0.05)
   Frame   500: actor mu range [-1.25, 1.22], ref vs mu MSE = 0.05 (正常)
   Frame  1000: actor mu range [-2.35, 4.28], ref vs mu MSE = 0.47
   ```

   某些帧 actor 输出正常，某些帧偏差巨大。统计 50 帧：38% 的输出维度超出 [-1, 1]。

3. **追溯训练数据**：检查 `offline_dataset.py` 的 cache 构建流程，发现 `exec_chunk`（专家动作）和 `ref_chunk`（VLA 参考动作）空间不一致。

---

## 二、Bug 1：Action 空间不匹配（已修复）

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

### 修复内容（commit 5f55475）

**`demo_loader.py`**：
- 新增 `normalize_quantiles(tensor, q01, q99)` 函数，使用 QUANTILES 公式 `(raw - q01) / (q99 - q01) * 2.0 - 1.0`
- `RLTDemoDataset.__init__()` 增加 `normalize_actions: bool = False` 参数
- 当 `normalize_actions=True` 时，从 `self._dataset.meta.stats["action"]` 读取 q01/q99
- `__getitem__()` 在返回 `expert_actions` 前应用归一化

**`deploy.py`**：
- `_rl_action_chunk()` 增加 `action_chunk.clamp(-1.0, 1.0)` 作为推理安全兜底

**所有 cache 构建脚本**：
- `build_rlt_offline_cache.py`：传入 `normalize_actions=True`
- `build_cp_offline_cache.py`：传入 `normalize_actions=True`
- `train_best_ac_on_sft.py`：传入 `normalize_actions=True`
- `offline_dataset.py` 的 `precompute_offline_buffer()`：传入 `normalize_actions=True`

---

## 三、Bug 2：State/Action 尺度不匹配（已修复）

### 发现过程

修复 Bug 1 后重建 cache 并训练，结果 critic loss 从初始 ~12 爆炸到 1e34，actor loss 到 -5e16。最终 eval 指标全部 1e17+。

### 根因

修复 Bug 1 后，action 归一化到 [-1, 1]（norm ~5），但 proprio 仍然是原始角度（norm ~200）。Critic 的输入中 action 占比从修复前的 73% 降到 2.4%，critic 几乎看不到 action 的差异 → Q 值失去对 action 的区分能力 → TD bootstrap 无约束爆炸。

| 输入 | Bug 1 修复前（角度 action） | Bug 1 修复后（归一化 action，原始 proprio） |
|------|--------------------------|--------------------------------------|
| z_rl | norm ~10 | norm ~10 |
| proprio | norm ~200 | norm ~200 |
| action | norm ~550 | **norm ~5** |
| action/state 比例 | 2.75 | **0.025** |

### 修复内容（commit bf23666）

**`demo_loader.py`**：
- `__init__()` 中增加 `_init_quantiles()` 方法，同时加载 action 和 `observation.state` 的 q01/q99
- `__getitem__()` 中对 proprio（`observation.state`）也应用 `normalize_quantiles()`
- 归一化后：state norm ~69（z_rl 主导），proprio 在 [-1.2, 1.7] 范围

**`obs_bridge.py`**：
- `robot_obs_to_rlt_obs()` 增加 `proprio_q01`/`proprio_q99` 可选参数
- 当传入 q01/q99 时，对 proprio 应用同样的 QUANTILES 归一化

**`deploy_config.py`**：
- 增加 `proprio_q01: list[float] | None` 和 `proprio_q99: list[float] | None` 字段

**`deploy.py`**：
- `__init__()` 中从 config 加载 `proprio_q01`/`proprio_q99` 为 tensor
- `_fill_action_queue()` 调用 `robot_obs_to_rlt_obs()` 时传入 proprio_q01/q99

---

## 四、Bug 3：Proprio 归一化后仍然训练发散（已修复）

### 发现过程

修复 Bug 2 后重建 cache 并训练。Cache 数据验证通过：
- `exec_chunk` 范围 [-1.08, 1.26] ✓
- `ref_chunk` 范围 [-1.00, 1.09] ✓
- `state_vec[-12:]`（proprio）范围 [-1.23, 1.73] ✓
- `state_vec` norm ~69（从 ~220 降下来） ✓

但训练仍然发散：

```
Critic loss 走势：
  step     0: 1.06
  step    50: 0.04
  step  1000: 0.09
  step  5000: 0.01   ← 很健康
  step 10000: 0.02   ← 仍然健康
  step 25000: 3.3e34  ← 爆炸
```

### 根因

TD3 的经典 Q 值过估计发散。33805 个 transition 中只有 177 个终端 transition 有 reward=10，其余全部 reward=0（因为 offline 模式下 `exec=expert`，action_matching reward `-(exec-expert)^2 = 0`）。

在 reward 几乎全零的情况下，TD target = γ^C × Q_target(next_state, mu_next)。当 actor 开始利用 critic 的外推误差（OOD actions），Q 值正反馈循环爆炸。没有梯度裁剪、没有 Q 值上限约束，发散不可避免。

### 修复内容（commit 102f243）

**`algorithm.py`**：
- `critic_update()` 和 `actor_update()` 增加 `grad_clip: float = 1.0` 参数
- 在 `loss.backward()` 后、`optimizer.step()` 前，调用 `torch.nn.utils.clip_grad_norm_(params, grad_clip)`

**`losses.py`**：
- `critic_loss()` 中，actor 的 target action 增加 `mu_next = mu_next.clamp(-1.0, 1.0)`
- target Q 值增加上下限 `q_next = q_next.clamp(-100.0, 100.0)`

**`configs/ac_best_v3.yaml`**：
- `beta` 从 0.3 调整到 5.0，增强 BC 正则化，限制 actor 偏离 VLA 参考

### 效果

修复后训练 50K 步完全稳定，最终结果：

```
Critic loss: 0.02-0.08 稳定（从未爆炸）
Eval: ref_mse=0.078, Q_policy=6.28, Q_expert=5.57, Q_gap=0.71, TD_err=0.34
```

- **Q_gap > 0**（0.71）：首次观测到 actor 的策略 Q 值高于专家 Q 值，说明 Q-learning 真正在工作
- **ref_mse = 0.078**：actor 与 VLA 参考偏差适中（不再是 Bug 下的虚假 0.008）
- **Q 值在合理范围**（5-7），不再爆炸

---

## 五、尚未实现的论文 Training Trick

### Trick 1：Stride-2 Subsampling（未实现）

**论文原文**：
> "Concretely we pick a stride of 2 and save transitions corresponding to `<x0, a0:C>, <x2, a2:C+2>, <x4, a4:C+4>, ...` to the replay buffer."

每 2 个控制步存一个 transition，action chunk 是从当前步开始的 C=10 步动作。这样每秒数据产生 ~25 个 RL 样本（50Hz / 2）。

**当前实现**：
- `build_rlt_offline_cache.py` 的 `frame_stride` 默认为 1（逐帧）
- 每帧产生一个 transition，比论文多 2 倍数据但 next_state 间隔不对

**需要改的地方**：
- `build_rlt_offline_cache.py`：CLI `--frame-stride` 默认改为 2
- 或者在 `_encoded_to_transitions()` 中按 stride 跳帧构建 transition

### Trick 2：Chunk-level MDP Bootstrap Discount（实现有误）

**论文原文**：chunk-level MDP 中每个 transition 的 bootstrap discount 应该是 γ^actual_steps，其中 actual_steps = stride（transition 间的实际步数），而不是 chunk_length。

**当前实现**（`offline_dataset.py:_encoded_to_transitions()`）：
```python
actual_steps=torch.tensor(chunk_length)  # 始终为 C=10
```

**问题**：next_state 间隔 1 帧（stride=1），但 discount = γ^10。相当于过度折扣了 Q 值传播，在 reward 稀疏时让 Q-learning 信号更加微弱。

**论文要求的对应关系**：
| stride | next_state 间隔 | actual_steps | bootstrap discount |
|--------|---------------|--------------|-------------------|
| 2（论文） | x_{t+2} | 2 | γ^2 = 0.98 |
| 10（chunk-level）| x_{t+10} | 10 | γ^10 = 0.90 |
| 1（当前） | x_{t+1} | **应为 1，但实际设为 10** | γ^10 = 0.90 ✗ |

**需要改的地方**：
- `_encoded_to_transitions()` 中 `actual_steps` 应等于实际 frame stride，不是 `chunk_length`
- `build_transitions_from_demos()` 或 `_encoded_to_transitions()` 需要接受 `stride` 参数
- reward_seq 的 `_action_matching_reward()` 已有 `actual_steps` 裁剪逻辑（步数外置零），无需额外改

### Trick 3：Reward 设计（部分实现）

**论文**：sparse +1 reward 由人类操作员在任务成功时给出。

**当前实现**：
- `reward_mode="hybrid"`，但 offline 模式下 `exec=expert` → action_matching reward 恒为 0
- 只有终端 transition 有 `success_bonus=10.0`
- 33805 个 transition 中只有 177 个有非零 reward

**影响**：reward 极度稀疏，Q-learning 信号微弱。配合错误的 γ^10 discount，Q 值传播几乎失效。修复 stride 和 discount 后应有改善。

---

## 六、完整代码修改清单

### 已完成的修改

| 文件 | 修改内容 | Commit |
|------|---------|--------|
| `src/lerobot/rlt/demo_loader.py` | 新增 `normalize_quantiles()` 函数；`RLTDemoDataset` 增加 `normalize_actions` 参数；同时归一化 action 和 proprio（`observation.state`）；增加 `_init_quantiles()` 方法 | 5f55475, bf23666 |
| `src/lerobot/rlt/deploy.py` | `_rl_action_chunk()` 增加 `clamp(-1, 1)`；`_fill_action_queue()` 传入 proprio_q01/q99；`__init__` 加载 proprio 归一化 tensor | 5f55475, bf23666 |
| `src/lerobot/rlt/deploy_config.py` | 增加 `proprio_q01: list[float]` 和 `proprio_q99: list[float]` 字段 | bf23666 |
| `src/lerobot/rlt/obs_bridge.py` | `robot_obs_to_rlt_obs()` 增加 `proprio_q01`/`proprio_q99` 可选参数，内部做 QUANTILES 归一化 | bf23666 |
| `src/lerobot/rlt/algorithm.py` | `critic_update()` 和 `actor_update()` 增加 `grad_clip=1.0` 梯度裁剪 | 102f243 |
| `src/lerobot/rlt/losses.py` | `critic_loss()` 中 target action 增加 `clamp(-1, 1)`，target Q 增加 `clamp(-100, 100)` | 102f243 |
| `scripts/build_rlt_offline_cache.py` | 传入 `normalize_actions=True`；增加 corrupt episode skip（try/except） | 5f55475, 24d2cd7 |
| `scripts/build_cp_offline_cache.py` | 传入 `normalize_actions=True` | 5f55475 |
| `scripts/train_best_ac_on_sft.py` | 传入 `normalize_actions=True` | 5f55475 |
| `src/lerobot/rlt/offline_dataset.py` | `precompute_offline_buffer()` 传入 `normalize_actions=True` | 5f55475 |
| `src/lerobot/rlt/configs/ac_best_v3.yaml` | ResidualMLP 3L/256h, β=5.0, 50K steps | 5d0ca41, 102f243 |

### 待完成的修改

| 文件 | 需要的修改 |
|------|----------|
| `src/lerobot/rlt/offline_dataset.py` | `_encoded_to_transitions()` 的 `actual_steps` 改为 stride 值而非 chunk_length；或增加 stride 参数按步跳帧构建 transition |
| `scripts/build_rlt_offline_cache.py` | `--frame-stride` 默认值从 1 改为 2（论文 stride=2） |
| `src/lerobot/rlt/configs/ac_best_v3.yaml` | 确认 `collector.chunk_subsample_stride: 2` 在 offline 训练中被正确使用 |

---

## 七、当前可用 Cache 和 Checkpoint

### Coder 服务器 (RTX 5090)

| 路径 | 内容 |
|------|------|
| `outputs/rlt_offline_cache_478ep_normalized/` | 478ep 数据集的归一化 cache（action + proprio 均已归一化），33805 train + 5551 val + 4185 test transitions。256 个 episode 因视频损坏被跳过 |
| `outputs/ac_478ep_normalized/rl_checkpoint.pt` | 上述 cache 训练的 AC checkpoint。β=5.0, grad_clip=1.0, Q_clamp=100, 50K steps。ref_mse=0.078, Q_gap=0.71 |
| `outputs/rlt_demo_adapt_271ep_sft_fp32/` | RL token checkpoint (271ep SFT VLA, 50K steps, loss=0.035) |

### VLA 模型

| 路径 | 模型 |
|------|------|
| `Elvinky/pi05_screw_271ep_sft_fp32`（HF cache） | 271ep SFT pi0.5（Coder 和 Machine 100 均有缓存） |

### 数据集

| 路径 | 内容 |
|------|------|
| `/home/coder/share/dataset_478ep` | HuggingFace `Shiki42/271ep_sft_success_critical_phase_478ep`，478 episodes, 98K frames, 53% 视频损坏 |

---

## 八、验证方法

### Cache 验证

```python
import torch
data = torch.load("transitions_train.pt", weights_only=False)
t = data[0]
print("exec_chunk range:", t["exec_chunk"].min(), t["exec_chunk"].max())   # 应在 [-1.1, 1.4]
print("ref_chunk range:", t["ref_chunk"].min(), t["ref_chunk"].max())       # 应在 [-1.1, 1.3]
print("state_vec norm:", t["state_vec"].norm())                             # 应 ~70，不是 ~220
print("proprio (last 12):", t["state_vec"][-12:])                           # 应在 [-1.5, 2.0]
print("reward_seq:", t["reward_seq"])                                       # 大部分应全零
```

### 训练验证

训练过程中 critic loss 应稳定在 < 1，不应爆炸。最终 eval 指标：
- `ref_mse` < 0.1（actor 输出接近 VLA 参考）
- `Q_gap` > 0（actor 学到了比专家更优的策略）
- `Q_policy` 和 `Q_expert` 在个位数范围

### 部署验证

在 eval 数据集上运行 `visualize_vla_actions.py`，RLT 输出应在 [-1, 1] 内无需 clamp，unnormalize 后 MAE(VLA, RLT) < 10°。

---

## 九、影响

- 所有已有 actor-critic checkpoint（`v2_base`、`v2_sft`、`v1_baseline`）都在 bug 下训练，**不可复用**
- RL token checkpoint（`v8_base`、`v1_sft`、`rlt_demo_adapt_271ep_sft_fp32`）**不受影响**
- 478ep 数据集有 53% 的 episode 视频损坏，需要检查数据集质量或重新上传
