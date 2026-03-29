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
Actor: πθ(a₁:C | x, ã₁:C) — Gaussian, ref dropout 50%, fixed_std=0.05
Critic: Qψ(x, a₁:C) — TD3 twin Q
```

Key hyperparams: C=10, H=50, action_dim=12, proprio_dim=12, token_dim=2048, γ=0.99, β=1.0, τ=0.005, UTD=5.

## Implementation status

### Milestone 1-7: COMPLETE (59/59 tests pass)

All core RLT modules implemented, reviewed (triple review: Claude + 2 Codex), and merged.

### Pi05 VLA Adapter: COMPLETE

Real pi0.5 integration via `Pi05VLAAdapter` — loads model, extracts prefix hidden states + action chunks via single forward with KV cache.

### Demo Adaptation Training: COMPLETE (loss = 0.015)

5 iterations of hyperparameter/architecture optimization on remote RTX 5090 server. Key breakthrough: token pooling (968→64 prefix tokens) + 4 RL tokens reduced compression ratio from 968:1 to 16:1.

### Offline RL Pipeline: COMPLETE (v1 trained)

Full offline RL infrastructure implemented (24 parallel sub-agents: Claude + Codex dual-version per module, /simplify + /codex-review dual review):
- `rewards.py`: pluggable reward builder (terminal/action_matching/hybrid)
- `offline_dataset.py`: episode-level splitting, transition building, VLA forward caching
- `envs/reaching.py`: 12-DOF toy environment for pipeline validation
- `trainer.py`: `offline_rl_loop()` with periodic logging/eval/checkpoint
- `evaluator.py`: `evaluate_offline()` with offline-specific metrics (expert_mse, q_gap, TD error)
- `scripts/build_rlt_offline_cache.py`: one-time VLA forward precomputation
- `scripts/train_rlt_offline_rl.py`: offline RL training entry (cached + live modes)
- 84/84 tests pass (59 original + 25 new)

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
| **v8 (best)** | `/home/coder/share/Evo-RL-quick/outputs/rlt_demo_adapt_v8/demo_adapt_checkpoint.pt` | Trained with pi0.5 base, pool=64, 4 RL tokens, 3+3 layers, 50K steps, final loss=0.015 |

Contains: `rl_token_state_dict`, `step`, `losses`. Config: `src/lerobot/rlt/configs/pi05_rlt.yaml`.

### Offline RL Checkpoint

| Checkpoint | Path on Coder Server | Description |
|-----------|---------------------|-------------|
| **v1** | `/home/coder/share/Evo-RL-quick/outputs/rlt_offline_rl_v1/rl_checkpoint.pt` | Actor + critic trained on cached demo transitions, 100K gradient steps |

Contains: `actor_state_dict`, `critic_state_dict`, `target_critic_state_dict`, optimizer states, `step`, loss histories.

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

### Offline RL v1

| Metric | Start | End | Notes |
|--------|-------|-----|-------|
| Critic loss | 0.0136 | 0.0016 | TD error converged |
| Actor loss | 1.2563 | 0.7066 | 44% reduction, BC reg dominant |
| Val critic | — | 0.003-0.008 | No overfitting |
| ref_mse | — | 0.334 | Actor close to VLA reference |
| expert_mse | — | 2715.75 | High — scale issue, needs investigation |
| Q_gap | — | -0.0087 | Correct sign (policy < expert) |
| TD_error | — | 0.128 | Moderate |
| Speed | — | 355 steps/sec | 100K steps in 282s on RTX 5090 |

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

1. **Use SFT model**: Re-run demo adaptation with the screw-task SFT model (`pi05_evorl_screw_147_sft`) instead of base pi0.5 — should give better RL tokens since the VLA already knows the task
2. **Investigate expert_mse**: The offline RL v1 expert_mse=2715 is very high — likely a scale/normalization issue. Check if action dimensions are being correctly handled
3. **Tune offline RL**: Try lower β (0.1-0.5) to let actor explore beyond VLA reference; try more gradient steps; try different reward modes
4. **Online RL**: Integrate a real or simulated environment, run `online_rl_loop()` with the trained actor/critic as initialization
5. **Phase controller**: Train the `HandoverClassifier` on collected intervention data
6. **Scale**: Try larger RL token encoder (4+4 layers) or more RL tokens (8+) if GPU allows
