# Offline RL Training Bug: Action Space Mismatch

## Root Cause

During offline RL training,`exec_chunk` (expert actions from dataset) and `ref_chunk` (VLA reference actions) are in **different action spaces**:

| Component | Source | Space | Typical Range |
|-----------|--------|-------|---------------|
| `exec_chunk` | LeRobot dataset `action` field | Raw degrees | [-103°, 97°] |
| `ref_chunk` | VLA `sampled_action_chunk` output | Normalized | [-1, 1] |

This mismatch propagates through the entire training pipeline:

1. **Critic** learns `Q(state, degree_action)` from `exec_chunk` in degree space
2. **Actor** is BC-regularized toward `ref_chunk` in normalized space (`actor_loss = -Q + β * MSE(mu, ref)`)
3. **Target Q** evaluates actor output (≈normalized) as if it were degrees → out-of-distribution

Result: Q-gradient to actor is noise. Actor converges only via BC term, with noisy perturbations from meaningless Q-gradients. At inference, actor outputs are approximately in [-1, 1] but with frequent outliers (38% of dims hit ±1 clamp). After unnormalization, outliers cause violent robot motion.

## Where the Bug Lives

### Cache building (`offline_dataset.py:build_transitions_from_demos`)

```python
e = _subsample_chunk(expert_actions[i], chunk_length)  # expert_actions in DEGREES
encoded.append((s, r, e, rew))  # r (ref_chunk) is NORMALIZED
```

The `expert_actions` come from `RLTDemoDataset.__getitem__()` which reads raw `item["action"]` from LeRobot dataset — these are in degree space. But `ref_chunk` comes from `policy.encode_observation()` → `vla.forward_vla()` → `sampled_action_chunk` which is the VLA's normalized output.

### Reward computation (`rewards.py`)

If using `action_matching` or `hybrid` reward mode, the reward compares `exec_chunk` (degrees) with `ref_chunk` (normalized) — comparing apples to oranges.

## How to Verify the Bug Exists

Run this diagnostic on any machine with the training dataset + VLA model:

```python
from lerobot.rlt.demo_loader import RLTDemoDataset, make_demo_loader
from lerobot.rlt.interfaces import Observation

dataset = RLTDemoDataset(dataset_path="<path>", chunk_length=50)
item = dataset[0]
print("expert_actions range:", item["expert_actions"].min(), item["expert_actions"].max())
# Expected: degree-scale values like [-103, 97]

# Then check ref_chunk from VLA:
# obs, expert = next(make_demo_loader(...))
# state_vec, ref_chunk = policy.encode_observation(obs)
# print("ref_chunk range:", ref_chunk.min(), ref_chunk.max())
# Expected: normalized [-1, 1]
```

If `expert_actions` range ≈ [-100, 100] and `ref_chunk` range ≈ [-1, 1], the bug is confirmed.

## The Fix

Normalize `exec_chunk` to [-1, 1] using the VLA model's q01/q99 quantiles **before** storing it in the transition. This ensures all action-space tensors (exec_chunk, ref_chunk, actor output) are in the same normalized [-1, 1] space.

### Changes required:

1. **`offline_dataset.py`**: Accept q01/q99 tensors; normalize `expert_actions` before creating transitions
2. **`build_rlt_offline_cache.py`**: Load q01/q99 from VLA model postprocessor; pass to cache builder
3. **`rewards.py`**: No change needed — once exec_chunk is normalized, comparisons with ref_chunk are correct
4. **`train_rlt_offline_rl.py`**: No change needed — consumes cached transitions

### Normalization formula (QUANTILES mode):

```python
# degrees -> [-1, 1]
normalized = (degrees - q01) / (q99 - q01) * 2.0 - 1.0
```

### What NOT to change:

- `ref_chunk`: already normalized (VLA output)
- `state_vec`: proprio stays in degrees (the actor input layer handles the scale)
- `deploy.py`: keep the `clamp(-1, 1)` as safety — but after retraining, outliers should be rare

## How to Verify the Fix Works

### 1. Pre-training check

After rebuilding the cache with normalized exec_chunk:

```python
# Load a cached transition
t = buf.sample(1)
print("exec_chunk range:", t["exec_chunk_flat"].min(), t["exec_chunk_flat"].max())
print("ref_chunk range:", t["ref_chunk_flat"].min(), t["ref_chunk_flat"].max())
# Both should be in [-1, 1]
```

### 2. Post-training check

After training actor-critic on the fixed cache:

```python
# Run actor on eval data
mu, _ = actor.forward(state_vec, ref_flat)
print("actor output range:", mu.min(), mu.max())
# Should be in [-1, 1] without clamping
# ref_mse should be < 0.01 (similar to pre-fix but now with meaningful Q-gradient)
```

### 3. Deployment check

Run `visualize_vla_actions.py` with the new checkpoint:

```bash
PYTHONPATH=src python scripts/visualize_vla_actions.py \
  --dataset-path <eval_dataset> \
  --vla-model <model_path> \
  --rl-token-ckpt <rl_token_ckpt> \
  --ac-ckpt <new_ac_ckpt> \
  --stride 10
```

Expected: RLT actions should be within [-1, 1] without needing clamp, and MAE(VLA, RLT) should be < 10° after unnormalization.

## Impact on Existing Checkpoints

All existing actor-critic checkpoints (`v2_base`, `v2_sft`, `v1_baseline`) were trained with this bug. They are NOT reusable with the fixed pipeline — must retrain from scratch with the new normalized cache.

RL token checkpoints (`v8_base`, `v1_sft`) are unaffected — they only use VLA hidden states, not action space.
