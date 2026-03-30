# RLT Project Context

## What this branch is

Branch `shuyuan/rlt` implements the **RLT (RL Token)** pipeline from the paper *"RL Token: Bootstrapping Online RL with Vision-Language-Action Models"* (Physical Intelligence, 2025).

## Development plan

- Final merged dev plan: `docs/rlt_final_dev_plan.md`
- Implementation dev plan: `dev_plan.md` (repo root)
- Full session log: `docs/session_log_rlt_implementation.md`

## Architecture (actual, post-iteration)

```
Frozen pi0.5 (PaLiGemma 2B + Gemma 300M, bf16)
  → prefix_output: (B, ~968, 2048) PaLiGemma hidden states
  → adaptive_avg_pool1d → (B, 64, 2048) pooled tokens
  → RL Token Encoder (3-layer transformer, 4 RL tokens) → (B, 4, 2048)
  → mean pool → z_rl (B, 2048)
  → RL Token Decoder (3-layer transformer) → reconstructed (B, 64, 2048)

z_rl + proprio = RL state x
Actor: πθ(a₁:C | x, ã₁:C) — ResidualMLP 3L/256h, ref dropout 50%, fixed_std=0.05
Critic: Qψ(x, a₁:C) — TD3 twin Q, ResidualMLP 3L/256h
```

Key hyperparams: C=10, H=50, action_dim=12, proprio_dim=12, token_dim=2048, γ=0.99, β=0.3, τ=0.005, UTD=5.

## Implementation status

### Milestone 1-7: COMPLETE (59/59 tests pass)

All core RLT modules implemented, reviewed (triple review: Claude + 2 Codex), and merged.

### Pi05 VLA Adapter: COMPLETE

Real pi0.5 integration via `Pi05VLAAdapter` — loads model, extracts prefix hidden states + action chunks via single forward with KV cache.

### Demo Adaptation Training: COMPLETE (loss = 0.015)

5 iterations of hyperparameter/architecture optimization on remote RTX 5090 server. Key breakthrough: token pooling (968→64 prefix tokens) + 4 RL tokens reduced compression ratio from 968:1 to 16:1.

### Offline RL Pipeline: COMPLETE (v2 optimized)

Full offline RL infrastructure implemented (24 parallel sub-agents: Claude + Codex dual-version per module, /simplify + /codex-review dual review):
- `rewards.py`: pluggable reward builder (terminal/action_matching/hybrid)
- `offline_dataset.py`: episode-level splitting, transition building, VLA forward caching
- `envs/reaching.py`: 12-DOF toy environment for pipeline validation
- `trainer.py`: `offline_rl_loop()` with periodic logging/eval/checkpoint
- `evaluator.py`: `evaluate_offline()` with offline-specific metrics (expert_mse, q_gap, TD error)
- `scripts/build_rlt_offline_cache.py`: one-time VLA forward precomputation
- `scripts/train_rlt_offline_rl.py`: offline RL training entry (cached + live modes)
- `scripts/search_actor_critic.py`: Phase 1 architecture search (factor isolation)
- `scripts/search_ac_phase2.py`: Phase 2 architecture search (combinations)
- `scripts/search_ac_phase3.py`: Phase 3 fine-tuning around winner

### Actor-Critic Architecture Search: COMPLETE (v2, 30x improvement)

3-phase search across 99 experiments on RTX 5090:
- **Phase 1** (32 experiments, 50K steps): isolated individual factors
- **Phase 2** (41 experiments, 100K steps): combined winning factors
- **Phase 3** (26 experiments, 200K-500K steps): fine-tuning around winner

**Winner**: ResidualMLP (3 layers, 256 hidden, ReLU) + β=0.3, 50K steps
- ref_mse: 0.008 (v1 baseline was 0.334 → **42x improvement**)
- q_gap: -0.011 (healthy)
- Key insight: residual connections alone give 34x improvement; longer training overfits

### SFT Model RL Token: TESTED (base model wins)

Trained RL tokens with the screw-task SFT model (`pi05_evorl_screw_147_sft`).
SFT final loss: 0.026 vs base model: 0.015. **Base model representations are more compressible.**

### Online RL Training: NOT STARTED

Blocked by lack of real robot / simulation environment. `online_rl_loop()` is implemented and tested with DummyEnvironment.

## Models & Checkpoints

### VLA Models (frozen, used as feature extractor)

| Model | Path on Coder Server | Description |
|-------|---------------------|-------------|
| **pi0.5 base (pretrained)** | `/home/coder/share/models/huggingface/hub/models--lerobot--pi05_base` | Original pretrained pi0.5, PaLiGemma 2B + Gemma 300M |
| **pi0.5 SFT (screw task)** | `/home/coder/share/models/pi05_evorl_screw_147_sft/` | Fine-tuned on 147 screw episodes (`Elvinky/pi05_evorl_screw_147_sft` on HF), 7.0GB |

### RL Token Checkpoint (demo adaptation)

| Checkpoint | Path on Coder Server | Description |
|-----------|---------------------|-------------|
| **v8 (best, base model)** | `outputs/rlt_demo_adapt_v8/demo_adapt_checkpoint.pt` | pi0.5 base, pool=64, 4 RL tokens, 3+3L, 50K steps, loss=0.015 |
| SFT model | `outputs/rlt_demo_adapt_sft_v1/demo_adapt_checkpoint.pt` | pi0.5 SFT, same config, 50K steps, loss=0.026 |

Contains: `rl_token_state_dict`, `step`, `losses`. Config: `src/lerobot/rlt/configs/pi05_rlt.yaml`.

### Offline RL Checkpoint

| Checkpoint | Path on Coder Server | Description |
|-----------|---------------------|-------------|
| **v2 (best)** | `outputs/ac_best_v2/rl_checkpoint_best.pt` | ResidualMLP 3L/256h, β=0.3, 50K steps, ref_mse=0.011 |
| v1 (baseline) | `outputs/rlt_offline_rl_v1/rl_checkpoint.pt` | Plain MLP 2L/256h, β=1.0, 100K steps, ref_mse=0.334 |

v2 contains: `actor_state_dict`, `critic_state_dict`, `target_critic_state_dict`, `rl_token_state_dict`, `config`, `metrics`.

### Architecture Search Results

| Search | Path on Coder Server | Description |
|--------|---------------------|-------------|
| Phase 1 | `outputs/ac_search_v1/results.json` | 32 experiments, factor isolation |
| Phase 2 | `outputs/ac_search_p2/results.json` | 41 experiments, combinations |
| Phase 3 | `outputs/ac_search_p3/results.json` | 26 experiments, fine-tuning |

### Offline RL Cache

| Cache | Path on Coder Server | Description |
|-------|---------------------|-------------|
| **transitions_train.pt** | `outputs/rlt_offline_cache/transitions_train.pt` (1.1GB) | 101,438 transitions from 117 episodes |
| **transitions_val.pt** | `outputs/rlt_offline_cache/transitions_val.pt` (110MB) | 10,391 transitions from 14 episodes |
| **transitions_test.pt** | `outputs/rlt_offline_cache/transitions_test.pt` (144MB) | 13,644 transitions from 16 episodes |

Built from: pi0.5 base + v8 RL token checkpoint, reward_mode=hybrid, token_pool_size=64. Build time: 121 min.

## Training Results

### Demo Adaptation (RL Token)

| Run | Config | Steps | Final Loss | Notes |
|-----|--------|-------|------------|-------|
| v1 | 2+2L, 1 tok, no clip | 5000 | 0.97 | Spike at step 3782 |
| v2 | +grad clip +cosine LR | 20000 | 0.29 | Crash at 8700 (memory leak) |
| v3 | 3+3L, +checkpoint +resume | 20000 | 0.29 | Fixed memory leak, stable |
| v7 | 2+2L, 1 tok, 50k steps | 50000 | 0.275 | 968:1 compression limit |
| **v8** | **pool=64, 4 tok, 3+3L** | **50000** | **0.015** | **Token pooling breakthrough** |

### Offline RL v1 (baseline, MLP)

| Metric | Start | End | Notes |
|--------|-------|-----|-------|
| Actor loss | 1.2563 | 0.7066 | Plain MLP 2L/256h, β=1.0 |
| ref_mse | — | 0.334 | Actor close to VLA reference |
| Q_gap | — | -0.0087 | Correct sign |

### Offline RL v2 (ResidualMLP, optimized)

| Metric | v1 | **v2** | Improvement |
|--------|-----|--------|-------------|
| ref_mse | 0.334 | **0.008-0.011** | **30-42x** |
| actor_loss | 0.707 | 0.242 | 3x lower |
| q_gap | -0.009 | -0.011 | Healthy |
| Architecture | MLP 2L | ResidualMLP 3L | +residual connections |
| Beta | 1.0 | 0.3 | Lower BC regularization |
| Steps | 100K | 50K | Shorter = better (overfitting) |

### Architecture Search Summary (99 total experiments)

| Factor | Best Ref MSE | vs Baseline | Phase |
|--------|-------------|-------------|-------|
| **Residual (3L)** | **0.008** | **42x** | P1 |
| LayerNorm | 0.014 | 24x | P1 |
| Hidden 512 | 0.023 | 15x | P1 |
| SiLU activation | 0.025 | 13x | P1 |
| Beta 0.1 | 0.029 | 12x | P1 |
| Residual + β=0.3 | 0.009 | 37x | P2 |

Key finding: factors do NOT compound. Residual alone is optimal. Longer training (>100K) overfits on fixed buffer.

### Demo Adaptation: SFT vs Base Model

| Model | Steps | Final Loss | Avg100 |
|-------|-------|------------|--------|
| **pi0.5 base** | **50K** | **0.015** | **0.015** |
| pi0.5 SFT | 50K | 0.028 | 0.026 |

Base model representations are more compressible for RL token encoding.

## Repository layout

```
src/lerobot/rlt/
  __init__.py              # Public API exports
  config.py                # RLTConfig dataclass tree with YAML loading + OfflineRLConfig
  interfaces.py            # Observation, VLAOutput, ChunkTransition, batch key constants
  vla_adapter.py           # VLAAdapter ABC + DummyVLAAdapter
  pi05_adapter.py          # Pi05VLAAdapter (real pi0.5 integration)
  rl_token.py              # RLTokenModule (multi-token encoder-decoder transformer)
  actor.py                 # ChunkActor with reference dropout
  critic.py                # ChunkCritic + TwinCritic
  losses.py                # discounted_chunk_return, critic_loss, actor_loss
  utils.py                 # soft_update, build_mlp, flatten/unflatten_chunk, subsample_indices
  replay_buffer.py         # Deque-based chunk replay with actual_steps tracking
  collector.py             # Environment ABC, warmup_collect, rl_collect_step
  agent.py                 # RLTAgent facade (encode_observation, deterministic action)
  phase_controller.py      # VLA→RL handoff state machine + HandoverClassifier
  trainer.py               # demo_adaptation + online_rl_loop + offline_rl_loop
  evaluator.py             # evaluate() + evaluate_offline() with OfflineEvalMetrics
  demo_loader.py           # RLTDemoDataset + make_demo_loader (gc-cleaning cycle)
  rewards.py               # build_reward_seq (terminal/action_matching/hybrid)
  offline_dataset.py       # split_episode_indices, build_transitions, cache save/load
  envs/
    __init__.py            # make_env factory
    reaching.py            # ReachingEnvironment (12-DOF toy env for validation)
  configs/
    base.yaml              # Default hyperparameters (action_dim=14)
    pi05_rlt.yaml          # Pi0.5 config (action_dim=12, 3+3 layers, 4 RL tokens, pool=64)

tests/rlt/
  test_interfaces.py, test_vla_adapter.py, test_rl_token.py,
  test_actor_critic.py, test_losses.py, test_replay.py,
  test_collector.py, test_agent.py, test_training_step.py,
  test_reaching_env.py, test_rewards.py,
  test_offline_dataset.py, test_offline_trainer.py, test_offline_eval.py

scripts/
  train_rlt_demo_adapt.py          # CLI for demo adaptation
  build_rlt_offline_cache.py       # CLI for VLA forward precomputation
  train_rlt_offline_rl.py          # CLI for offline RL training (cached + live modes)
  search_actor_critic.py           # Phase 1 architecture search (factor isolation, 50K steps)
  search_ac_phase2.py              # Phase 2 search (combinations, 100K steps)
  search_ac_phase3.py              # Phase 3 fine-tuning (200K-500K steps)
```

## Remote server environment

- **Coder workspace**: `shuyuan/evo-rl/main` on `https://coder.eulerai.au`
- **Working dir**: `/home/coder/share/Evo-RL-quick` (branch `shuyuan/rlt`)
- **GPU**: RTX 5090 32GB
- **PyTorch**: nightly 2.12.0.dev+cu128 (required for sm_120)
- **Python venv**: `/home/coder/share/venv-lerobot`
- **Dataset**: `/home/coder/share/dataset` (147 episodes, 125K frames, 12-DOF, 3 cameras)
- **SSH**: use `/coder-evo-rl-ssh` skill; connection is unstable under GPU load, use tmux

## Code rules

- `from __future__ import annotations` in every file
- No try/except for error swallowing
- Max 3 levels of nesting
- Each file under 1000 lines
- Reuse existing code; no backward compat shims

## What to do next

1. ~~**Use SFT model**~~: DONE — base model wins (loss 0.015 vs SFT 0.026)
2. ~~**Tune offline RL**~~: DONE — 99-experiment architecture search, ResidualMLP + β=0.3 gives 42x improvement
3. **Investigate expert_mse**: The expert_mse=2715 is still high — likely a scale/normalization issue in how expert actions are stored vs computed
4. **Online RL**: Integrate a real or simulated environment, run `online_rl_loop()` with the v2 actor/critic as initialization
5. **Phase controller**: Train the `HandoverClassifier` on collected intervention data
6. **Rebuild offline cache with SFT model**: Even though SFT RL tokens are worse, the SFT model's action chunks may be better for the screw task. Consider rebuilding cache with SFT VLA actions + base RL tokens
