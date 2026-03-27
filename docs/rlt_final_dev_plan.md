# RLT Final Development Plan (Merged)

> Merged from two independent analyses of *"RL Token: Bootstrapping Online RL with Vision-Language-Action Models"* (Physical Intelligence, 2025).
> Combines complete PyTorch implementations with clean software engineering practices.

---

## Repository Layout

```text
src/lerobot/rlt/
  __init__.py
  configs/
    base.yaml              # default hyperparameters
  interfaces.py            # Observation, VLAOutput, ChunkTransition dataclasses
  vla_adapter.py           # VLAAdapter ABC — wraps any chunked VLA
  rl_token.py              # RLTokenEncoder + RLTokenDecoder
  actor.py                 # ChunkActor with reference dropout
  critic.py                # ChunkCritic + TwinCritic
  replay_buffer.py         # Chunk-level replay with stride-2 subsampling
  agent.py                 # RLTAgent — ties everything together
  losses.py                # RL token reconstruction, critic TD, actor BC-reg losses
  collector.py             # Rollout loop, warmup, human intervention handling
  phase_controller.py      # Critical-phase handoff + optional handover classifier
  trainer.py               # Demo adaptation trainer + online RL trainer
  evaluator.py             # Evaluation loop with metrics
  utils.py                 # Soft update, discount computation, safety utils
tests/
  test_rlt/
    test_shapes.py         # Shape smoke tests for all modules
    test_losses.py         # Loss computation correctness
    test_replay.py         # Replay buffer + subsampling
    test_training_step.py  # Single gradient step end-to-end
```

---

## Build Order (7 Milestones)

### Milestone 1 — Interfaces & Config

**Files**: `interfaces.py`, `configs/base.yaml`

Define all data containers first. Every downstream module depends on these.

```python
# interfaces.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class Observation:
    images: Dict[str, torch.Tensor]        # camera_name -> (B, C, H, W)
    proprio: torch.Tensor                   # (B, proprio_dim)
    instruction_ids: Optional[torch.Tensor] = None
    timestamp: Optional[float] = None

@dataclass
class VLAOutput:
    final_tokens: torch.Tensor             # (B, M, token_dim)
    sampled_action_chunk: torch.Tensor     # (B, H, action_dim)
    extra: dict = field(default_factory=dict)

@dataclass
class ChunkTransition:
    state_vec: torch.Tensor                # (B, state_dim) = concat(z_rl, proprio)
    exec_chunk: torch.Tensor               # (B, C, action_dim)
    ref_chunk: torch.Tensor                # (B, C, action_dim)
    reward_seq: torch.Tensor               # (B, C)
    next_state_vec: torch.Tensor           # (B, state_dim)
    next_ref_chunk: torch.Tensor           # (B, C, action_dim)
    done: torch.Tensor                     # (B,)
    intervention: torch.Tensor             # (B,) — 0/1 flag
```

```yaml
# configs/base.yaml
seed: 0
control_hz: 50
action_dim: 14
vla_horizon: 50       # H [Paper]
chunk_length: 10      # C [Paper]
chunk_subsample_stride: 2  # [Paper]

cameras:
  - wrist_left
  - wrist_right
  - base

rl_token:
  token_dim: 2048     # match VLA hidden size
  nhead: 8            # [Engineering choice]
  enc_layers: 4       # [Engineering choice]
  dec_layers: 4       # [Engineering choice]

demo_adaptation:
  steps: 5000         # [Paper: 2000-10000]
  batch_size: 32
  lr: 1.0e-4
  vla_ft_weight: 1.0  # alpha

actor:
  hidden_dim: 256     # [Paper: 256 standard, 512 for screw]
  num_layers: 2       # [Paper: 2 standard, 3 for screw]
  fixed_std: 0.05
  lr: 3.0e-4

critic:
  hidden_dim: 256
  num_layers: 2
  lr: 3.0e-4

gamma: 0.99
beta: 1.0             # BC regularization weight
ref_dropout_p: 0.5    # [Paper]
tau: 0.005            # target network soft update
batch_size: 256
utd_ratio: 5          # [Paper]
actor_update_interval: 2  # [Paper: 2 critic updates per 1 actor update]

warmup_steps: 5000
replay_capacity: 200000
total_env_steps: 100000
critical_phase_only: true
allow_human_intervention: true
```

---

### Milestone 2 — VLA Adapter

**File**: `vla_adapter.py`

Abstract away the VLA so RL code never depends on proprietary internals. One forward pass returns both embeddings and action chunk (avoid redundant VLA inference).

```python
# vla_adapter.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .interfaces import Observation, VLAOutput

class VLAAdapter(ABC, nn.Module):
    """Abstract adapter for any chunked VLA model."""

    @abstractmethod
    def forward_vla(self, obs: Observation) -> VLAOutput:
        """Single forward pass returning final token embeddings + sampled action chunk."""
        ...

    @abstractmethod
    def supervised_loss(self, obs: Observation, expert_actions: torch.Tensor) -> torch.Tensor:
        """Action prediction loss for optional VLA fine-tuning on demos."""
        ...

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Dimension of each token in final_tokens."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Per-timestep action dimension."""
        ...
```

For testing without a real VLA, also implement:

```python
class DummyVLAAdapter(VLAAdapter):
    """Random-output adapter for shape testing and development."""
    def __init__(self, token_dim=2048, num_tokens=64, action_dim=14, horizon=50):
        super().__init__()
        self._token_dim = token_dim
        self._action_dim = action_dim
        self._num_tokens = num_tokens
        self._horizon = horizon

    def forward_vla(self, obs: Observation) -> VLAOutput:
        B = obs.proprio.shape[0]
        return VLAOutput(
            final_tokens=torch.randn(B, self._num_tokens, self._token_dim),
            sampled_action_chunk=torch.randn(B, self._horizon, self._action_dim),
        )

    def supervised_loss(self, obs, expert_actions):
        return torch.tensor(0.0)

    @property
    def token_dim(self): return self._token_dim
    @property
    def action_dim(self): return self._action_dim
```

---

### Milestone 3 — RL Token Module

**File**: `rl_token.py`

The RL token is the core architectural contribution. Encoder appends a learned `<rl>` embedding, runs a transformer, and reads out the last position. Decoder autoregressively reconstructs VLA embeddings from this bottleneck.

```python
# rl_token.py
import torch
import torch.nn as nn

class RLTokenModule(nn.Module):
    def __init__(self, token_dim: int = 2048, nhead: int = 8,
                 num_enc_layers: int = 4, num_dec_layers: int = 4,
                 ff_dim: int = 4096):
        super().__init__()
        self.rl_token_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out_proj = nn.Linear(token_dim, token_dim)

    def encode(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vla_tokens: (B, M, D) — final VLA token embeddings (stop-gradiented)
        Returns:
            z_rl: (B, D) — the RL token
        """
        B = vla_tokens.shape[0]
        rl = self.rl_token_embed.expand(B, -1, -1)
        x = torch.cat([vla_tokens, rl], dim=1)   # (B, M+1, D)
        out = self.encoder(x)
        return out[:, -1, :]                       # (B, D)

    def decode(self, z_rl: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive reconstruction with teacher forcing.
        Args:
            z_rl: (B, D)
            teacher_tokens: (B, M, D) — stop-gradiented VLA embeddings
        Returns:
            pred: (B, M, D)
        """
        B, M, D = teacher_tokens.shape
        memory = z_rl.unsqueeze(1)                                        # (B, 1, D)
        # Shifted input: [z_rl, z_1, ..., z_{M-1}]
        dec_input = torch.cat([memory, teacher_tokens[:, :-1]], dim=1)     # (B, M, D)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(M).to(z_rl.device)
        out = self.decoder(tgt=dec_input, memory=memory, tgt_mask=causal_mask)
        return self.out_proj(out)

    def reconstruction_loss(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """L_ro = E[Σ_i || pred_i - z̄_i ||²]"""
        z_bar = vla_tokens.detach()
        z_rl = self.encode(z_bar)
        pred = self.decode(z_rl, z_bar)
        return ((pred - z_bar) ** 2).mean()
```

---

### Milestone 4 — Actor & Critic

**Files**: `actor.py`, `critic.py`

Both are lightweight MLPs. The actor is conditioned on both the RL state and the VLA reference chunk, with 50% reference dropout during training.

```python
# actor.py
import torch
import torch.nn as nn

class ChunkActor(nn.Module):
    def __init__(self, state_dim: int, chunk_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, fixed_std: float = 0.05,
                 ref_dropout_p: float = 0.5):
        super().__init__()
        layers = []
        in_dim = state_dim + chunk_dim  # state + flattened reference chunk
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, chunk_dim))
        self.net = nn.Sequential(*layers)
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p

    def forward(self, state_vec: torch.Tensor, ref_chunk_flat: torch.Tensor,
                training: bool = False):
        if training:
            mask = (torch.rand(state_vec.shape[0], 1, device=state_vec.device)
                    > self.ref_dropout_p).float()
            ref_chunk_flat = ref_chunk_flat * mask
        x = torch.cat([state_vec, ref_chunk_flat], dim=-1)
        mu = self.net(x)
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def sample(self, state_vec, ref_chunk_flat, training=False):
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        return mu + std * torch.randn_like(std), mu
```

```python
# critic.py
import torch
import torch.nn as nn

class ChunkCritic(nn.Module):
    def __init__(self, state_dim: int, chunk_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim + chunk_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_vec: torch.Tensor, action_flat: torch.Tensor):
        return self.net(torch.cat([state_vec, action_flat], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, chunk_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        self.q1 = ChunkCritic(state_dim, chunk_dim, hidden_dim, num_layers)
        self.q2 = ChunkCritic(state_dim, chunk_dim, hidden_dim, num_layers)

    def forward(self, state_vec, action_flat):
        return self.q1(state_vec, action_flat), self.q2(state_vec, action_flat)

    def min_q(self, state_vec, action_flat):
        q1, q2 = self.forward(state_vec, action_flat)
        return torch.minimum(q1, q2)
```

---

### Milestone 5 — Losses

**File**: `losses.py`

Three losses, all from the paper.

```python
# losses.py
import torch
import torch.nn.functional as F

def discounted_chunk_return(reward_seq: torch.Tensor, gamma: float) -> torch.Tensor:
    """reward_seq: (B, C) -> (B, 1)"""
    B, C = reward_seq.shape
    discounts = gamma ** torch.arange(C, device=reward_seq.device, dtype=reward_seq.dtype)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def critic_loss(critic, target_critic, actor, batch: dict, gamma: float, C: int):
    """TD3-style chunk-level TD loss."""
    x, a = batch["state_vec"], batch["exec_chunk_flat"]
    x_next, ref_next = batch["next_state_vec"], batch["next_ref_flat"]
    reward_seq, done = batch["reward_seq"], batch["done"]

    with torch.no_grad():
        a_next, _ = actor.sample(x_next, ref_next)
        q_next = target_critic.min_q(x_next, a_next)
        r = discounted_chunk_return(reward_seq, gamma)
        target = r + (gamma ** C) * (1.0 - done.unsqueeze(-1)) * q_next

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(actor, critic, batch: dict, beta: float):
    """Q-maximization + BC regularization toward VLA reference."""
    x, ref = batch["state_vec"], batch["ref_chunk_flat"]
    a, _ = actor.sample(x, ref, training=True)
    q = critic.min_q(x, a)
    bc_reg = F.mse_loss(a, ref)
    return -q.mean() + beta * bc_reg
```

---

### Milestone 6 — Replay Buffer & Collector

**Files**: `replay_buffer.py`, `collector.py`

```python
# replay_buffer.py
import torch
import random
from collections import deque
from .interfaces import ChunkTransition

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, transition: ChunkTransition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        return {
            "state_vec": torch.stack([t.state_vec for t in batch]),
            "exec_chunk_flat": torch.stack([t.exec_chunk.flatten(-2) for t in batch]),
            "ref_chunk_flat": torch.stack([t.ref_chunk.flatten(-2) for t in batch]),
            "reward_seq": torch.stack([t.reward_seq for t in batch]),
            "next_state_vec": torch.stack([t.next_state_vec for t in batch]),
            "next_ref_flat": torch.stack([t.next_ref_chunk.flatten(-2) for t in batch]),
            "done": torch.stack([t.done for t in batch]),
        }
```

Collector handles warmup, RL rollout, human intervention, and stride-2 subsampling.

---

### Milestone 7 — Agent, Trainer & Evaluator

**Files**: `agent.py`, `trainer.py`, `evaluator.py`, `phase_controller.py`

The `RLTAgent` ties VLA adapter + RL token + actor/critic into a single interface.

`trainer.py` implements:
1. **Demo adaptation**: `L_ro(φ) + α·L_vla(θ_vla)` for 2k-10k steps, then freeze.
2. **Online RL**: Async collection + off-policy updates at UTD=5.

`phase_controller.py` handles VLA→RL handoff (manual during training, learned for deployment).

`evaluator.py` tracks: success rate, episode length, throughput (successes/10min), mean Q, action deviation from reference.

---

## Testing Strategy (Before Robot)

1. **Unit tests**: Shape correctness for every module (rl_token encode/decode, actor, critic, replay collation).
2. **Loss tests**: Verify reconstruction loss decreases, TD target computation, BC regularization gradient direction.
3. **Smoke test**: Random-tensor forward/backward through full pipeline, confirm gradients only flow through actor/critic (not frozen VLA/RL-token).
4. **Sim test**: Toy chunk-control environment with sparse reward, verify learning curve.

---

## Failure Modes to Watch

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Actor copies reference exactly | Dropout not applied / β too high | Verify 50% dropout; lower β |
| Q-values explode | Target network / done masking bug | Check τ, done flags, reward alignment |
| No improvement over VLA | C=1 or no warmup | Use C=10, ensure warmup |
| RL token uninformative | Insufficient training | Train longer, verify correct VLA layer |
| Slow training | VLA in gradient loop | Cache z_rl and ref during collection |
