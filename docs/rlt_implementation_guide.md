# RLT (RL Token) Implementation Guide

> Implementation blueprint for reproducing the architecture described in
> *"RL Token: Bootstrapping Online RL with Vision-Language-Action Models"*
> (Physical Intelligence, 2025)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1 — RL Token Encoder-Decoder Training](#2-phase-1--rl-token-encoder-decoder-training)
3. [Phase 2 — Online RL Actor-Critic Training](#3-phase-2--online-rl-actor-critic-training)
4. [Phase 3 — Full System Integration](#4-phase-3--full-system-integration)
5. [Data Pipeline & Replay Buffer](#5-data-pipeline--replay-buffer)
6. [Training Loop (Algorithm 1)](#6-training-loop-algorithm-1)
7. [Hyperparameters & Network Specs](#7-hyperparameters--network-specs)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Architecture Overview

The system has two sequential training stages and three runtime components:

```
┌─────────────────────────────────────────────────────────────────┐
│  FROZEN VLA (π0.6)                                              │
│  ┌──────────────┐   ┌──────────────────┐                        │
│  │ VLM Backbone  │──▸│  Action Expert   │──▸ reference chunk ã  │
│  │ SigLIP+Gemma  │   │  (diffusion)     │    (H=50 steps)      │
│  └──────┬───────┘   └──────────────────┘                        │
│         │ z₁:M  (token embeddings)                              │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ RL Token Enc │──▸ z_rl (1×2048)                              │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
          │ z_rl                          │ ã₁:C
          ▼                               ▼
   ┌────────────────────────────────────────────┐
   │  Lightweight Actor-Critic (online RL)      │
   │  x = (z_rl, s_p)                           │
   │  Actor:  πθ(a₁:C | x, ã₁:C) = N(μ, σ²I)  │
   │  Critic: Qψ(x, a₁:C) ∈ ℝ                  │
   └────────────────────────────────────────────┘
```

**Key design decisions:**
- VLA is frozen during online RL — only the small actor/critic train online.
- RL token compresses the full VLA embedding sequence into a single 2048-d vector.
- Actor is conditioned on VLA reference action chunk and regularized toward it.
- Action chunks (C=10 steps at 50 Hz = 0.2s) shorten the effective MDP horizon.

---

## 2. Phase 1 — RL Token Encoder-Decoder Training

### 2.1 Goal

Train a small encoder-decoder transformer that compresses the VLA's internal token embeddings `z₁:M` into a single bottleneck vector `z_rl`, such that a decoder can reconstruct `z₁:M` from `z_rl` alone.

### 2.2 Data Requirement

- A small task-specific demonstration dataset `D` (1–10 hours of teleoperated demos).
- Run frozen VLA forward pass on each demo to extract `z₁:M` embeddings.

### 2.3 RL Token Encoder

```python
class RLTokenEncoder(nn.Module):
    """
    Lightweight transformer encoder that compresses VLA embeddings
    into a single RL token.
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        num_layers: int = 4,    # paper does not specify; 4 is a reasonable default
        num_heads: int = 8,
        ff_dim: int = 4096,
    ):
        super().__init__()
        # Learned RL token embedding (appended to the VLA sequence)
        self.rl_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: VLA token embeddings, shape (B, M, D)
        Returns:
            z_rl: RL token, shape (B, D)
        """
        B = z.shape[0]
        # Append learned RL token to end of sequence
        rl_token = self.rl_token_embed.expand(B, -1, -1)  # (B, 1, D)
        augmented = torch.cat([z, rl_token], dim=1)        # (B, M+1, D)

        # Run encoder; extract last position (the RL token)
        out = self.encoder(augmented)                       # (B, M+1, D)
        z_rl = out[:, -1, :]                                # (B, D)
        return z_rl
```

### 2.4 RL Token Decoder

The decoder autoregressively reconstructs the original VLA embeddings from `z_rl`:

```python
class RLTokenDecoder(nn.Module):
    """
    Autoregressive transformer decoder that reconstructs VLA embeddings
    from the RL token, enforcing the bottleneck.
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 4096,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        z_rl: torch.Tensor,
        z_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive reconstruction: predict z_i from [z_rl, z_1, ..., z_{i-1}].

        Args:
            z_rl:      (B, D) — the RL token
            z_targets: (B, M, D) — stop-gradiented VLA embeddings (teacher forcing)
        Returns:
            z_pred: (B, M, D) — reconstructed embeddings
        """
        B, M, D = z_targets.shape

        # Decoder input: [z_rl, z_1, ..., z_{M-1}] (shifted right)
        z_rl_expanded = z_rl.unsqueeze(1)                              # (B, 1, D)
        decoder_input = torch.cat([z_rl_expanded, z_targets[:, :-1]], dim=1)  # (B, M, D)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(M)
        causal_mask = causal_mask.to(decoder_input.device)

        # Memory is just z_rl (cross-attention to the bottleneck)
        memory = z_rl_expanded  # (B, 1, D)

        out = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
        )
        z_pred = self.output_proj(out)  # (B, M, D)
        return z_pred
```

### 2.5 Reconstruction Loss

```python
def compute_rl_token_loss(
    encoder: RLTokenEncoder,
    decoder: RLTokenDecoder,
    vla_embeddings: torch.Tensor,  # (B, M, D) — from frozen VLA forward pass
) -> torch.Tensor:
    """
    L_ro = E_D [ Σ_i || h(d([z_rl, z̄_1:i-1]))_i - z̄_i ||² ]

    VLA embeddings are stop-gradiented (z̄ = sg(z)).
    """
    z_bar = vla_embeddings.detach()  # stop gradient on VLA embeddings
    z_rl = encoder(z_bar)            # (B, D)
    z_pred = decoder(z_rl, z_bar)    # (B, M, D)

    loss = F.mse_loss(z_pred, z_bar)
    return loss
```

### 2.6 Optional: Joint VLA Fine-Tuning

The paper optionally fine-tunes the VLA weights `θ_vla` jointly with the RL token training:

```
L_total(φ, θ_vla) = L_ro(φ) + α · L_vla(θ_vla)
```

Where `L_vla` is the standard VLA action prediction loss (diffusion denoising loss) on the task demos. The weight `α` controls the trade-off. After this stage, **both φ and θ_vla are frozen permanently**.

### 2.7 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training steps | 2,000 – 10,000 gradient steps |
| Dataset | Task-specific demos (1–10 hours) |
| VLA gradients | Frozen for L_ro; optionally unfrozen for L_vla with weight α |
| Optimizer | AdamW (lr ~1e-4, standard schedule) |
| Embedding dim | 2048 (matching VLA hidden size) |

---

## 3. Phase 2 — Online RL Actor-Critic Training

### 3.1 State Representation

The RL state `x` combines the RL token with proprioceptive information:

```python
def build_rl_state(z_rl: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """
    Args:
        z_rl:   (B, 2048) — RL token from frozen encoder
        proprio: (B, P) — proprioceptive state (joint pos, EE pose, etc.)
    Returns:
        x: (B, 2048 + P) — concatenated RL state
    """
    return torch.cat([z_rl, proprio], dim=-1)
```

Proprioceptive inputs vary by task:
- **Screw installation**: joint positions
- **Zip tie / Ethernet / Charger**: end-effector poses

### 3.2 Critic Network

```python
class ChunkedCritic(nn.Module):
    """
    Q(x, a_{1:C}) -> scalar value estimate.
    Uses TD3-style ensemble of 2 Q-functions.
    """
    def __init__(
        self,
        state_dim: int,        # 2048 + proprio_dim
        action_dim: int = 140, # C * d = 10 * 14
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.q1 = self._build_mlp(state_dim + action_dim, hidden_dim, num_layers)
        self.q2 = self._build_mlp(state_dim + action_dim, hidden_dim, num_layers)

    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([x, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, x: torch.Tensor, a: torch.Tensor):
        q1, q2 = self.forward(x, a)
        return torch.min(q1, q2)
```

### 3.3 Actor Network

The actor takes both the state and the VLA reference action chunk as input:

```python
class ChunkedActor(nn.Module):
    """
    πθ(a_{1:C} | x, ã_{1:C}) = N(μ_θ(x, ã), σ²I)

    Outputs a Gaussian over action chunks, conditioned on the
    VLA reference chunk.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 140,     # C * d = 10 * 14
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.1,
        ref_dropout_prob: float = 0.5,
    ):
        super().__init__()
        # Input: state + reference action chunk
        input_dim = state_dim + action_dim
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self.fixed_std = fixed_std
        self.ref_dropout_prob = ref_dropout_prob

    def forward(
        self,
        x: torch.Tensor,
        ref_action: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (B, action_dim) — Gaussian mean
            std:  (B, action_dim) — fixed standard deviation
        """
        # Reference action dropout during training
        if training:
            mask = (torch.rand(x.shape[0], 1, device=x.device) > self.ref_dropout_prob).float()
            ref_action = ref_action * mask  # zero out ref for dropout fraction

        inp = torch.cat([x, ref_action], dim=-1)
        mean = self.net(inp)
        std = torch.full_like(mean, self.fixed_std)
        return mean, std

    def sample(self, x, ref_action, training=False):
        mean, std = self.forward(x, ref_action, training)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return action, mean
```

### 3.4 Critic Loss (TD3-style)

```python
def critic_loss(
    critic: ChunkedCritic,
    target_critic: ChunkedCritic,
    actor: ChunkedActor,
    batch: dict,
    gamma: float = 0.99,
    chunk_length: int = 10,
) -> torch.Tensor:
    """
    L_Q = E[(Q̂ - Q_ψ(x, a))²]
    Q̂ = Σ_{t'=1}^{C} γ^{t'-1} r_{t'} + γ^C · E_{a'~π}[Q_ψ'(x', a')]
    """
    x = batch["state"]           # (B, state_dim)
    a = batch["action"]          # (B, 140)
    r = batch["reward"]          # (B,) — sparse binary, only at episode end
    x_next = batch["next_state"] # (B, state_dim)
    ref_next = batch["next_ref"] # (B, 140) — VLA ref at next state
    done = batch["done"]         # (B,)

    with torch.no_grad():
        # For sparse rewards, most intermediate r = 0
        # Multi-step return: Σ γ^{t'-1} r_{t'} computed during buffer storage
        cumulative_reward = batch["cumulative_reward"]  # (B,)

        a_next, _ = actor.sample(x_next, ref_next)
        q_next = target_critic.min_q(x_next, a_next)
        q_target = cumulative_reward + (gamma ** chunk_length) * q_next * (1 - done)

    q1, q2 = critic(x, a)
    loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    return loss
```

### 3.5 Actor Loss (Q-maximization + BC regularization)

```python
def actor_loss(
    actor: ChunkedActor,
    critic: ChunkedCritic,
    batch: dict,
    vla_policy,          # frozen VLA for sampling fresh reference actions
    beta: float = 1.0,   # BC regularization weight
) -> torch.Tensor:
    """
    L_π(θ) = E[-Q_ψ(x, a) + β ||a - ã||²]
    where a ~ πθ(·|x, ã) and ã ~ π_vla(·|s)
    """
    x = batch["state"]
    ref_action = batch["ref_action"]  # ã from VLA (stored in replay buffer)

    # Sample action from the RL actor
    a, _ = actor.sample(x, ref_action, training=True)

    # Q-value maximization term
    q_value = critic.min_q(x, a)
    q_loss = -q_value.mean()

    # BC regularization toward VLA reference
    bc_loss = F.mse_loss(a, ref_action)

    loss = q_loss + beta * bc_loss
    return loss
```

---

## 4. Phase 3 — Full System Integration

### 4.1 Critical Phase Switching

The system executes in two modes during a task:

1. **Non-critical phase**: Run the base VLA policy directly.
2. **Critical phase**: Switch to the RL actor for precision-critical segments.

```python
class RLTController:
    """
    Manages policy switching between base VLA and RL actor.
    """
    def __init__(self, vla, rl_token_encoder, actor, chunk_length_vla=50, chunk_length_rl=10):
        self.vla = vla                        # frozen
        self.rl_token_encoder = rl_token_encoder  # frozen
        self.actor = actor                    # online-trained
        self.C = chunk_length_rl
        self.H = chunk_length_vla
        self.in_critical_phase = False

    def step(self, obs, proprio, lang_instruction):
        """
        Called at each action-chunk boundary.

        Returns:
            actions: action chunk to execute
            metadata: dict with z_rl, ref_action for replay buffer
        """
        # Always run VLA to get embeddings and reference action
        z_embeddings = self.vla.get_embeddings(obs, lang_instruction)
        ref_action_full = self.vla.sample_action(obs, lang_instruction)  # (H, d)
        z_rl = self.rl_token_encoder(z_embeddings)

        ref_action_chunk = ref_action_full[:self.C]  # take first C steps

        if not self.in_critical_phase:
            # Execute VLA directly (first 20 steps of H=50, then re-plan)
            return ref_action_full[:20], {"z_rl": z_rl, "ref": ref_action_chunk}

        # Critical phase: use RL actor
        x = torch.cat([z_rl, proprio], dim=-1)
        ref_flat = ref_action_chunk.flatten()
        action, _ = self.actor.sample(x.unsqueeze(0), ref_flat.unsqueeze(0))
        action = action.squeeze(0).reshape(self.C, -1)

        return action, {"z_rl": z_rl, "ref": ref_action_chunk}
```

### 4.2 Human Intervention Interface

During online RL training, a human operator can:
1. **Switch to critical phase**: Signal when the RL policy should take over.
2. **Intervene with teleoperation**: Override the RL actor's output with human commands.
3. **Label success/failure**: Provide sparse binary reward at episode end.

```python
class HumanInterventionManager:
    """
    Manages human-in-the-loop signals during online RL training.
    """
    def __init__(self):
        self.intervention_active = False
        self.critical_phase_active = False

    def get_action(self, rl_action, human_input):
        """
        If human intervenes, override RL action.
        When intervention occurs, the human action also replaces
        the VLA reference in the replay buffer.
        """
        if human_input is not None and human_input.get("intervening"):
            self.intervention_active = True
            return human_input["action"], True  # action, is_intervention
        self.intervention_active = False
        return rl_action, False

    def get_reward(self, human_input):
        """Sparse binary reward from human label."""
        if human_input is not None and "success" in human_input:
            return 1.0 if human_input["success"] else 0.0
        return 0.0  # no reward signal yet
```

### 4.3 Autonomous Phase Switching (Post-Training)

After RL training, fine-tune the VLA to predict the switch point automatically:

```python
def train_phase_switch_classifier(vla, intervention_timestamps, demo_data):
    """
    Short fine-tuning phase: train VLA to additionally predict
    a binary 'hand-over-to-RL' signal, using human intervention
    timestamps as labels.

    This enables fully autonomous execution at test time.
    """
    # Add a small binary classification head to the VLA
    # Label = 1 at timesteps where human switched to RL during training
    # Label = 0 otherwise
    # Fine-tune with standard BCE loss for a small number of steps
    pass
```

---

## 5. Data Pipeline & Replay Buffer

### 5.1 Replay Buffer Design

The buffer stores chunk-level transitions from three sources:
- VLA warmup rollouts
- Online RL rollouts
- Human interventions

```python
@dataclass
class ChunkTransition:
    state: np.ndarray          # x = (z_rl, s_p)
    action: np.ndarray         # a_{1:C}, shape (C*d,) — the actually executed chunk
    ref_action: np.ndarray     # ã_{1:C} from VLA (or human action if intervention)
    cumulative_reward: float   # Σ γ^{t'-1} r_{t'} over the C steps
    next_state: np.ndarray     # x' at step t+C
    next_ref: np.ndarray       # ã' at next chunk boundary
    done: bool                 # episode terminated within this chunk


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: ChunkTransition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buffer, batch_size)
        return {
            "state": torch.stack([t.state for t in batch]),
            "action": torch.stack([t.action for t in batch]),
            "ref_action": torch.stack([t.ref_action for t in batch]),
            "cumulative_reward": torch.tensor([t.cumulative_reward for t in batch]),
            "next_state": torch.stack([t.next_state for t in batch]),
            "next_ref": torch.stack([t.next_ref for t in batch]),
            "done": torch.tensor([t.done for t in batch], dtype=torch.float32),
        }
```

### 5.2 Action Chunk Subsampling

To increase data efficiency, store overlapping chunks with stride 2:

```
From one C=10 chunk execution with observations at every step:
  Store: <x₀, a₀:C>, <x₂, a₂:C+2>, <x₄, a₄:C+4>, ...
```

This yields ~25 training samples per second of robot data (at 50 Hz).

```python
def subsample_and_store(
    replay_buffer: ReplayBuffer,
    observations: list,   # length C, one per control step
    actions: list,        # length C
    ref_actions: list,
    rewards: list,
    stride: int = 2,
    chunk_length: int = 10,
):
    """Store subsampled chunk transitions into replay buffer."""
    for start in range(0, chunk_length, stride):
        end = start + chunk_length
        if end > len(actions):
            break  # need future data from next chunk execution

        transition = ChunkTransition(
            state=observations[start],
            action=np.concatenate(actions[start:end]),
            ref_action=np.concatenate(ref_actions[start:end]),
            cumulative_reward=compute_discounted_return(rewards[start:end]),
            next_state=observations[min(end, len(observations) - 1)],
            next_ref=np.concatenate(ref_actions[start:end]),  # approximate
            done=(end >= len(actions) and rewards[-1] != 0),
        )
        replay_buffer.add(transition)
```

---

## 6. Training Loop (Algorithm 1)

### 6.1 Complete Training Procedure

```python
def rlt_training_loop(
    vla,
    rl_token_encoder,
    actor,
    critic,
    target_critic,
    replay_buffer,
    env,
    human_interface,
    # Hyperparameters
    warmup_steps: int = 500,
    total_episodes: int = 1000,
    chunk_length: int = 10,      # C
    vla_chunk_length: int = 50,  # H
    update_to_data_ratio: int = 5,  # G
    gamma: float = 0.99,
    beta: float = 1.0,
    critic_update_per_actor: int = 2,
    batch_size: int = 256,
    target_tau: float = 0.005,
):
    """
    Main training loop implementing Algorithm 1.
    Rollouts and learning run asynchronously in practice.
    """
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    env_steps = 0

    for episode in range(total_episodes):
        obs = env.reset()
        done = False

        while not done:
            # --- Forward pass through frozen VLA ---
            z_emb = vla.get_embeddings(obs)
            z_rl = rl_token_encoder(z_emb)
            ref_chunk_full = vla.sample_action(obs)        # (H, d)
            ref_chunk = ref_chunk_full[:chunk_length]       # (C, d)
            proprio = env.get_proprioception()
            x = concat(z_rl, proprio)

            # --- Select action ---
            if human_interface.is_intervening():
                action_chunk = human_interface.get_teleop_action()
                ref_chunk = action_chunk  # override reference with human action
            elif env_steps < warmup_steps:
                action_chunk = ref_chunk  # use VLA reference during warmup
            else:
                action_chunk = actor.sample(x, flatten(ref_chunk))

            # --- Execute chunk and collect data ---
            for t in range(chunk_length):
                next_obs, reward, done, info = env.step(action_chunk[t])
                env_steps += 1
                if done:
                    break

            # --- Store transition(s) with subsampling ---
            store_transitions(replay_buffer, ...)

            # --- Off-policy updates (async in practice) ---
            if env_steps >= warmup_steps and len(replay_buffer) >= batch_size:
                for g in range(update_to_data_ratio):
                    batch = replay_buffer.sample(batch_size)

                    # Update critic
                    c_loss = critic_loss(critic, target_critic, actor, batch, gamma, chunk_length)
                    critic_optimizer.zero_grad()
                    c_loss.backward()
                    critic_optimizer.step()

                    # Update actor (every 2 critic updates)
                    if g % critic_update_per_actor == 0:
                        a_loss = actor_loss(actor, critic, batch, vla, beta)
                        actor_optimizer.zero_grad()
                        a_loss.backward()
                        actor_optimizer.step()

                    # Soft update target critic
                    soft_update(target_critic, critic, target_tau)

            obs = next_obs

        # --- Episode end: get human reward label ---
        episode_reward = human_interface.get_reward_label()
        replay_buffer.label_episode(episode_reward)
```

### 6.2 Asynchronous Execution

In practice, the paper runs rollouts and gradient updates in **separate threads/processes**:

```
Thread 1 (Robot):  collect data at 50 Hz → push to replay buffer
Thread 2 (GPU):    sample from buffer → gradient updates (G=5 per env step)
```

This prevents gradient computation from blocking real-time robot control.

---

## 7. Hyperparameters & Network Specs

### 7.1 Architecture

| Component | Spec |
|-----------|------|
| VLA backbone | π0.6: SigLIP (400M) + Gemma (4B) + Action Expert (860M) |
| RL token dim | 2048 |
| RL token encoder | Small transformer (details in §2.3) |
| Actor (standard tasks) | 2-layer MLP, hidden=256 |
| Actor (hard tasks, e.g. screw) | 3-layer MLP, hidden=512 |
| Critic | 2-layer MLP, hidden=256 (same scaling as actor) |
| Critic ensemble | 2 Q-networks (TD3 style) |

### 7.2 RL Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control frequency | 50 Hz | |
| VLA chunk length H | 50 | 1 second of control |
| RL chunk length C | 10 | 0.2 seconds; C < H for reactivity |
| Per-timestep action dim d | 14 | Joint/EE deltas |
| Total actor action dim | 140 | C × d |
| Action subsampling stride | 2 | ~25 samples per second of data |
| Discount factor γ | 0.99 | |
| BC regularization β | Task-dependent | Start with 1.0, tune per task |
| Reference action dropout | 50% during training | Always provide ref at inference |
| Update-to-data ratio G | 5 | High ratio for sample efficiency |
| Critic updates per actor update | 2 | |
| Actor fixed σ | Small (e.g. 0.1) | Gaussian exploration noise |
| Target network τ | 0.005 | Soft update |
| Reward | Sparse binary (+1 success, 0 failure) | Human-labeled |
| Warmup episodes | Task-dependent | Fill buffer with VLA rollouts first |
| Training episodes | 400–1000 | 15 min – 5 hours of robot data |
| RL token training steps | 2,000–10,000 | On task demos |

---

## 8. Implementation Checklist

### Stage 0: Prerequisites

- [ ] Obtain a pretrained VLA model (π0.6 or equivalent) with:
  - Access to internal token embeddings `z₁:M` (hook into VLM backbone output)
  - Ability to sample action chunks from the diffusion action expert
- [ ] Set up robot environment at 50 Hz control frequency
- [ ] Implement human operator interface (intervention, phase switching, reward labeling)
- [ ] Collect 1–10 hours of task-specific teleoperation demonstrations

### Stage 1: RL Token Training (Offline)

- [ ] Run frozen VLA forward pass on all demos, cache `z₁:M` embeddings
- [ ] Implement RL token encoder (transformer + learned RL token embedding)
- [ ] Implement RL token decoder (autoregressive transformer + linear projection)
- [ ] Train encoder-decoder with MSE reconstruction loss (2k–10k steps)
- [ ] (Optional) Jointly fine-tune VLA on task demos with weight α
- [ ] Freeze both VLA and RL token encoder/decoder permanently
- [ ] Validate: check reconstruction quality, verify z_rl captures task-relevant info

### Stage 2: Online RL Training

- [ ] Implement chunked actor (MLP, ref-action conditioned, ref dropout)
- [ ] Implement chunked critic (TD3 dual-Q MLP)
- [ ] Implement replay buffer with chunk-level transitions and stride-2 subsampling
- [ ] Implement warmup phase: collect N_warm steps using VLA policy
- [ ] Implement TD learning with C-step chunk returns
- [ ] Implement actor loss: Q-maximization + β · BC regularization
- [ ] Implement soft target network updates
- [ ] Implement async rollout/update loop
- [ ] Implement human intervention handling (action override + buffer relabeling)
- [ ] Implement sparse reward propagation (label at episode end, backfill buffer)

### Stage 3: Deployment

- [ ] Implement critical-phase detection (manual during training, learned for deployment)
- [ ] Policy switching: base VLA for easy phases, RL actor for critical phases
- [ ] (Optional) Fine-tune VLA to predict phase switch point automatically
- [ ] Remove exploration noise (always provide reference action, no dropout)
- [ ] Evaluate: success rate and throughput over 50+ episodes

### Stage 4: Iteration

- [ ] Two-phase training: start with critical-phase-only, then advance to full-task
- [ ] Monitor for training instabilities (value divergence, action collapse)
- [ ] Tune β per task (higher for harder tasks where VLA prior is important)
- [ ] Compare throughput (successes per 10 min) not just success rate
