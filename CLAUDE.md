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

Real pi0.5 integration via `Pi05VLAAdapter` — loads `lerobot/pi05_base`, extracts prefix hidden states + action chunks via single forward with KV cache.

### Demo Adaptation Training: COMPLETE (loss = 0.015)

5 iterations of hyperparameter/architecture optimization on remote RTX 5090 server. Key breakthrough: token pooling (968→64 prefix tokens) + 4 RL tokens reduced compression ratio from 968:1 to 16:1.

### Online RL Training: NOT STARTED

Next step: run `online_rl_loop()` with a real or simulated environment.

## Repository layout

```
src/lerobot/rlt/
  __init__.py              # Public API exports
  config.py                # RLTConfig dataclass tree with YAML loading
  interfaces.py            # Observation, VLAOutput, ChunkTransition, batch key constants
  vla_adapter.py           # VLAAdapter ABC + DummyVLAAdapter
  pi05_adapter.py          # Pi05VLAAdapter (real pi0.5 integration)
  rl_token.py              # RLTokenModule (multi-token encoder-decoder transformer)
  actor.py                 # ChunkActor with reference dropout
  critic.py                # ChunkCritic + TwinCritic
  losses.py                # discounted_chunk_return, critic_loss, actor_loss
  utils.py                 # soft_update, build_mlp, flatten/unflatten_chunk
  replay_buffer.py         # Deque-based chunk replay with actual_steps tracking
  collector.py             # Environment ABC, warmup_collect, rl_collect_step
  agent.py                 # RLTAgent facade (single VLA forward guarantee)
  phase_controller.py      # VLA→RL handoff state machine + HandoverClassifier
  trainer.py               # demo_adaptation (grad clip, cosine LR, checkpointing) + online_rl_loop
  evaluator.py             # evaluate() with EvalMetrics
  demo_loader.py           # RLTDemoDataset + make_demo_loader (gc-cleaning cycle)
  configs/
    base.yaml              # Default hyperparameters (action_dim=14)
    pi05_rlt.yaml           # Pi0.5 config (action_dim=12, 3+3 layers, 4 RL tokens, pool=64)

tests/rlt/
  test_interfaces.py, test_vla_adapter.py, test_rl_token.py,
  test_actor_critic.py, test_losses.py, test_replay.py,
  test_collector.py, test_agent.py, test_training_step.py

scripts/
  train_rlt_demo_adapt.py  # CLI for demo adaptation (supports --resume, --token-pool-size)

src/lerobot/datasets/
  video_utils.py           # Modified: added decode_video_frames_pure_pyav() fallback
```

## Remote server environment

- **Coder workspace**: `shuyuan/evo-rl/main` on `https://coder.eulerai.au`
- **Working dir**: `/home/coder/share/Evo-RL-quick` (branch `shuyuan/rlt`)
- **GPU**: RTX 5090 32GB
- **PyTorch**: nightly 2.12.0.dev+cu128 (required for sm_120)
- **Python venv**: `/home/coder/share/venv-lerobot`
- **Dataset**: `/home/coder/share/dataset` (147 episodes, 125K frames, 12-DOF, 3 cameras)
- **pi0.5 model**: `/home/coder/share/models/huggingface/hub/models--lerobot--pi05_base`
- **Best checkpoint**: `/home/coder/share/Evo-RL-quick/outputs/rlt_demo_adapt_v8/demo_adapt_checkpoint.pt`
- **Loss curve**: `outputs/rlt_demo_adapt_v8/losses.json` (50000 data points, final=0.015)
- **SSH**: use `/coder-evo-rl-ssh` skill; connection is unstable under GPU load, use tmux

## Training iterations log

| Run | Config | Steps | Final Loss | Notes |
|-----|--------|-------|------------|-------|
| v1 | 2+2L, 1 tok, no clip | 5000 | 0.97 | Spike at step 3782 |
| v2 | +grad clip +cosine LR | 20000 | 0.29 | Crash at 8700 (memory leak) |
| v3 | 3+3L, +checkpoint +resume | 20000 | 0.29 | Fixed memory leak, stable |
| v7 | 2+2L, 1 tok, 50k steps | 50000 | 0.275 | 968:1 compression limit |
| **v8** | **pool=64, 4 tok, 3+3L** | **50000** | **0.015** | **Token pooling breakthrough** |

## Code rules

- `from __future__ import annotations` in every file
- No try/except for error swallowing
- Max 3 levels of nesting
- Each file under 1000 lines
- Reuse existing code; no backward compat shims

## What to do next

1. **Online RL**: Implement a real or simulated environment, run `online_rl_loop()` from `trainer.py`
2. **Evaluation**: Use `evaluator.py` to measure success rate, episode length, Q-values
3. **Phase controller**: Train the `HandoverClassifier` on collected intervention data
4. **Policy registration**: Optionally integrate into `lerobot.policies.*` via factory pattern
5. **Scale**: Try larger RL token encoder (4+4 layers) or more RL tokens (8+) if GPU allows
