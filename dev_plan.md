# RLT Implementation Plan

> Merged from Claude Plan (codebase-convention-aware) and Codex Plan (spec-first engineering analysis).
> Source requirements: `docs/rlt_final_dev_plan.md`

---

## Strategy

Implement RLT as a standalone subpackage at `src/lerobot/rlt/` with pure PyTorch modules. Do NOT integrate into `lerobot.policies.*` / factory / PreTrainedPolicy on day one -- the algorithm must be validated first.

Reuse repo conventions where they help:
- Dataclass config tree (matching `SACConfig` / draccus pattern in `src/lerobot/configs/policies.py`)
- Actor/critic MLP patterns from `src/lerobot/policies/sac/modeling_sac.py`
- pytest style from `tests/policies/`, `tests/rl/`

No existing files need modification -- setuptools auto-discovers packages under `src/`.

**Python version**: Repo declares `requires-python = ">=3.10"`. Use `from __future__ import annotations` for style consistency (some files like `processor/core.py` already do), but use 3.10+ syntax (`dict[str, T]`, `X | None`) matching the rest of the codebase.

---

## File Map

### Source (`src/lerobot/rlt/`)

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `__init__.py` | Public API exports | 30 |
| `config.py` | `RLTConfig` dataclass tree with nested sub-configs | 150 |
| `configs/base.yaml` | Default hyperparameters (loadable by config.py) | 40 |
| `interfaces.py` | `Observation`, `VLAOutput`, `ChunkTransition` dataclasses + batch key constants | 80 |
| `vla_adapter.py` | `VLAAdapter` ABC + `DummyVLAAdapter` | 100 |
| `rl_token.py` | `RLTokenModule`: encoder, decoder, reconstruction loss | 120 |
| `actor.py` | `ChunkActor` with reference dropout, `sample()` | 80 |
| `critic.py` | `ChunkCritic`, `TwinCritic`, `min_q` | 70 |
| `losses.py` | `discounted_chunk_return`, `critic_loss`, `actor_loss` | 80 |
| `utils.py` | `soft_update`, `flatten_chunk`, `compute_discount_vector` | 60 |
| `replay_buffer.py` | Deque-based chunk-level replay, collation | 100 |
| `collector.py` | Rollout, warmup, stride-2 subsampling, intervention handling | 250 |
| `agent.py` | `RLTAgent` facade: VLA + RL token + actor/critic | 200 |
| `phase_controller.py` | VLA->RL handoff state machine + optional classifier | 150 |
| `trainer.py` | Demo adaptation loop + online RL loop | 400 |
| `evaluator.py` | Evaluation loop + metrics | 150 |
| **Total** | | **~2060** |

### Tests (`tests/rlt/`)

| File | Covers | Milestone |
|------|--------|-----------|
| `__init__.py` | | M1 |
| `test_interfaces.py` | Dataclass creation, shape validation | M1 |
| `test_vla_adapter.py` | DummyVLA shapes, properties | M2 |
| `test_rl_token.py` | Encode/decode shapes, reconstruction loss, gradient flow, causal mask | M3 |
| `test_actor_critic.py` | Forward shapes, ref dropout stats, twin-Q min | M4 |
| `test_losses.py` | Discount computation, critic/actor loss correctness, gradient targets | M5 |
| `test_replay.py` | Add/sample, capacity limit, batch keys | M6 |
| `test_collector.py` | Warmup, rollout with dummy env, stride-2 subsampling | M6 |
| `test_agent.py` | End-to-end forward, action selection | M7 |
| `test_training_step.py` | Single gradient step, loss not NaN, frozen params stay frozen | M7 |

---

## Dependency Graph

```
M1: interfaces.py, config.py, __init__.py
 │
 ├── M2: vla_adapter.py
 │    │
 │    └── M3: rl_token.py
 │              │
 │              ├── M4: actor.py, critic.py
 │              │    │
 │              │    └── M5: losses.py
 │              │
 │              └── M6: replay_buffer.py, collector.py
 │
 └── M7: utils.py, agent.py, phase_controller.py, trainer.py, evaluator.py
```

Every milestone must be green (all tests pass) before the next begins.

---

## Milestone Details

### M1 -- Interfaces & Config

**Create**: `__init__.py`, `interfaces.py`, `config.py`, `configs/base.yaml`

**Config approach**: Use a Python `@dataclass` tree rather than YAML-only. This matches the existing `SACConfig` / draccus pattern and enables type checking + IDE support. Provide a `classmethod` to load defaults from `base.yaml`.

```
RLTConfig
  ├── rl_token: RLTokenConfig (token_dim, nhead, enc_layers, dec_layers, ff_dim)
  ├── actor: ActorConfig (hidden_dim, num_layers, fixed_std, lr, ref_dropout_p)
  ├── critic: CriticConfig (hidden_dim, num_layers, lr)
  ├── demo_adaptation: DemoAdaptConfig (steps, batch_size, lr, vla_ft_weight)
  ├── training: TrainingConfig (gamma, beta, tau, batch_size, utd_ratio, actor_update_interval)
  ├── replay: ReplayConfig (capacity)
  └── collector: CollectorConfig (warmup_steps, total_env_steps, chunk_subsample_stride)
```

**Design decisions**:
- `ChunkTransition` stores single (unbatched) transitions; batching at sample time
- Batch dict key names as string constants in `interfaces.py` to avoid typos
- Use `dict[str, torch.Tensor]` (3.10+ style) matching rest of codebase

**Tests**: Instantiate every dataclass with dummy tensors; config YAML round-trip.

---

### M2 -- VLA Adapter

**Create**: `vla_adapter.py`

- `VLAAdapter(ABC, nn.Module)` with `forward_vla`, `supervised_loss`, `token_dim`, `action_dim`
- `DummyVLAAdapter` in same file (~30 lines), returns random tensors with correct shapes
- Single forward pass returns both `final_tokens` AND `sampled_action_chunk` (no double inference)

**Tests**: DummyVLA output shapes, property values, supervised_loss returns scalar.

---

### M3 -- RL Token Module

**Create**: `rl_token.py`

- Learnable `<rl>` parameter appended to VLA tokens
- Transformer encoder reads out last position -> `z_rl (B, D)`
- Transformer decoder reconstructs with causal masking + teacher forcing
- `norm_first=True` (pre-norm, more stable)
- `ff_dim` default `4 * token_dim` but configurable

**Pitfalls**:
- Causal mask: PyTorch >=2.0 returns bool masks by default; ensure device + dtype compat
- Stop-gradient contract: `encode` receives already-detached tokens; `reconstruction_loss` handles detach internally
- Memory: use small dims (token_dim=64, nhead=4, layers=1) in unit tests

**Tests**: Encode shape `(B, D)`, decode shape `(B, M, D)`, reconstruction loss decreases on fixed data, gradients flow only to encoder/decoder params (not input tokens), causal masking verified.

---

### M4 -- Actor & Critic

**Create**: `actor.py`, `critic.py`

**Actor**:
- Input: `state_vec (B, state_dim)` + `ref_chunk_flat (B, chunk_dim)` where `state_dim = token_dim + proprio_dim`, `chunk_dim = C * action_dim`
- Reference dropout: binary mask per batch element (not `nn.Dropout`), zeros entire ref chunk with prob 0.5
- Fixed std (0.05), `sample()` adds Gaussian noise
- Output: `mu, std` both shape `(B, chunk_dim)`

**Critic**:
- `ChunkCritic`: MLP, input `state_vec + action_flat`, output `(B, 1)`
- `TwinCritic`: two ChunkCritics, `min_q` returns element-wise minimum

**Tests**: Output shapes, ref dropout zeroes ~50% of batch (test with large batch), gradient flow, twin-Q min is correct.

---

### M5 -- Losses

**Create**: `losses.py`

Three standalone functions (not class methods):
1. `discounted_chunk_return(reward_seq, gamma)` -> `(B, 1)` vectorized discount
2. `critic_loss(critic, target_critic, actor, batch, gamma, C)` -> TD3-style with chunk-level discounting, `torch.no_grad()` for targets
3. `actor_loss(actor, critic, batch, beta)` -> Q-maximization + BC regularization, critic NOT detached (actor needs gradients through Q)

**Pitfalls**:
- Reward alignment: `reward_seq` must correspond to the C timesteps of executed chunk; off-by-one common
- `done=1` must zero out the bootstrap term
- Critic params should not be in actor optimizer (handled at trainer level)

**Tests**: Hand-computed discount, done masking, BC term scales with beta, target is stop-gradiented.

---

### M6 -- Replay Buffer & Collector

**Create**: `replay_buffer.py`, `collector.py`

**Buffer**: Deque-based for simplicity. Known perf issue: `random.sample(list(...))` copies full list. Add TODO for tensor-backed circular buffer optimization later.

**Collector**: Most complex file. Orchestrates:
1. **Warmup**: Run VLA policy without RL, fill buffer
2. **RL rollout**: VLA forward -> RL token encode -> actor chunk -> execute -> store transition
3. **Intervention**: Set flag on transitions when human takes over

**Key design decisions**:
- Stride-2 subsampling belongs in the collector, not the buffer (H=50, C=10 -> 5 chunks per VLA call, store at positions 0, 2, 4)
- Collector receives VLA adapter + agent as dependencies (dependency injection, testable with dummies)
- Abstract `Environment` protocol/ABC so collector is testable with dummy env
- Start single-threaded; add threading in M7 if needed

**Tests**: Buffer add/sample shapes, capacity limit, batch keys; collector warmup + rollout with dummy env, stride-2 verification.

---

### M7 -- Agent, Trainer, Phase Controller, Evaluator

**Create**: `utils.py`, `agent.py`, `trainer.py`, `phase_controller.py`, `evaluator.py`

**RLTAgent** (`agent.py`, ~200 lines):
- Facade owning VLA adapter, RL token, actor, critic, target critic
- `get_rl_state(obs)` -> VLA forward + RL token encode -> concat with proprio
- `select_action(obs)` -> full pipeline: state + ref -> actor sample -> reshape `(C, action_dim)`
- `get_reference_chunk(obs)` -> VLA forward -> subsample to length C

**Trainer** (`trainer.py`, ~400 lines):
- **Demo adaptation**: `L_ro + alpha * L_vla` for N steps, then freeze VLA + RL token encoder
- **Online RL**: Collection loop + UTD=5 updates per env step. Every `actor_update_interval` critic updates, one actor update. Soft-update targets.
- Split into methods; if >1000 lines, factor update logic into helpers

**Phase Controller** (`phase_controller.py`, ~150 lines):
- State machine: `VLA_PHASE -> CRITICAL_PHASE`
- Manual mode: human triggers transition (training)
- Learned mode: binary classifier on `z_rl` predicts critical state (deployment)

**Evaluator** (`evaluator.py`, ~150 lines):
- Metrics: success rate, episode length, throughput (successes/10min), mean Q, action deviation from reference

**Utils** (`utils.py`, ~60 lines):
- `soft_update(target, source, tau)`
- `flatten_chunk(chunk)`: `(B, C, action_dim)` -> `(B, C*action_dim)`
- `compute_discount_vector(gamma, length)`

**Tests**: Agent shapes, single training step (loss not NaN, frozen params stay frozen), soft update correctness, phase controller transitions.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gradient freeze/unfreeze bugs | RL token or VLA accidentally trained/frozen in wrong phase | Test `requires_grad` and `param.grad is None` after each phase's training step |
| Replay chunk boundary misalignment | TD targets poisoned by off-by-one | Unit test with hand-constructed 2-chunk episode, verify reward_seq indices |
| VLA abstraction mismatch | Real VLA may not expose both tokens + action chunks in one pass | `VLAAdapter` ABC enforces the contract; add concrete adapter only when VLA is available |
| Causal mask dtype (PyTorch >=2.0) | Silent shape/device errors | Explicit `.to(device=..., dtype=torch.float32)` on mask |
| trainer.py / collector.py bloat | Exceed 1000-line limit | Factor update logic and rollout helpers into separate functions early |
| Integration creep | Premature `lerobot.policies.factory` integration adds config/processor burden | Stay under `lerobot.rlt` until algorithm validated; no policy registry until M7+ |
| Buffer perf (deque + list copy) | Slow sampling with large buffer | Start simple, add tensor-backed circular buffer as follow-up |

---

## Integration Notes

- `src/lerobot/rlt/` is auto-discovered by setuptools. `from lerobot.rlt.agent import RLTAgent` works immediately.
- Can import shared utilities: `lerobot.utils.random_utils` (seeding), `lerobot.utils.constants` (standard keys).
- Do NOT reuse `lerobot.rl.buffer` (HILSerl-specific transition format, different from `ChunkTransition`).
- Do NOT reuse `lerobot.datasets.online_buffer` (numpy.memmap-backed, different data contract).
- Future policy-registry integration (if needed) would add `RLTConfig` to `lerobot.policies` and register in `factory.py` -- separate follow-up work.
