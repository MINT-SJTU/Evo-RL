# RLT Project Context

## What this branch is

Branch `shuyuan/rlt` implements the **RLT (RL Token)** pipeline from the paper *"RL Token: Bootstrapping Online RL with Vision-Language-Action Models"* (Physical Intelligence, 2025).

## Development plan

The final merged dev plan is at `docs/rlt_final_dev_plan.md`. It was synthesized from two independent analyses:
- `docs/rlt_implementation_guide.md` — code-heavy version with full PyTorch implementations
- The user also had `/Users/shuyuan/Downloads/rlt_pipeline_implementation_guide.md` — an engineering-first version with clean interfaces, failure modes, testing strategy

The final plan merges the best of both: complete code + clean software engineering.

## Architecture summary

RLT = freeze a pretrained VLA, extract a compact "RL token" from its embeddings, and train a lightweight actor-critic on top for sample-efficient online RL.

```
Frozen VLA → token embeddings z₁:M → RL Token Encoder → z_rl (1×2048)
                                   → Action Expert → reference chunk ã (H=50)

z_rl + proprio = RL state x
Actor: πθ(a₁:C | x, ã₁:C) — Gaussian, ref dropout 50%
Critic: Qψ(x, a₁:C) — TD3 twin Q
```

Key hyperparams: C=10, H=50, action_dim=14, 50Hz, γ=0.99, β=1.0, τ=0.005, UTD=5.

## Repository layout to create

```
src/lerobot/rlt/
  __init__.py, configs/base.yaml, interfaces.py, vla_adapter.py,
  rl_token.py, actor.py, critic.py, losses.py, replay_buffer.py,
  collector.py, agent.py, phase_controller.py, trainer.py, evaluator.py, utils.py
tests/test_rlt/
  test_shapes.py, test_losses.py, test_replay.py, test_training_step.py
```

## Build order (7 milestones)

1. Interfaces & Config
2. VLA Adapter (ABC + DummyVLAAdapter)
3. RL Token Module (encoder-decoder transformer)
4. Actor & Critic (MLP heads)
5. Losses (reconstruction, critic TD, actor BC-reg)
6. Replay Buffer & Collector
7. Agent, Trainer, Evaluator, Phase Controller

## What has been done

- Two parallel sub-agent implementations were completed and tested (39/39 tests passed)
- Both implementations have been cleaned up (worktrees removed)
- No code has been committed to this branch yet — ready for fresh implementation

## Code rules

- `from __future__ import annotations` in every file (Python 3.9 compat)
- No try/except for error swallowing
- Max 3 levels of nesting
- Each file under 1000 lines
- Reuse existing code; no backward compat shims
