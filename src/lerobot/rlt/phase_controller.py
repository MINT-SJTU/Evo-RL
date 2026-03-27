from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn


class Phase(Enum):
    VLA_PHASE = "vla_phase"
    CRITICAL_PHASE = "critical_phase"


class HandoverClassifier(nn.Module):
    """Binary classifier on z_rl to predict critical state (for deployment)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_rl: torch.Tensor) -> torch.Tensor:
        """Returns logit for critical phase prediction, (B, 1)."""
        return self.net(z_rl)

    def predict(self, z_rl: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns boolean tensor (B,) indicating critical phase."""
        return torch.sigmoid(self.forward(z_rl)).squeeze(-1) > threshold


class PhaseController:
    """State machine for VLA -> RL handoff.

    Manual mode (training): human triggers transition.
    Learned mode (deployment): binary classifier on z_rl predicts critical state.
    """

    def __init__(
        self,
        mode: str = "manual",
        classifier: HandoverClassifier | None = None,
        threshold: float = 0.5,
    ):
        if mode not in ("manual", "learned"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.classifier = classifier
        self.threshold = threshold
        self._phase = Phase.VLA_PHASE

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_critical(self) -> bool:
        return self._phase == Phase.CRITICAL_PHASE

    def reset(self) -> None:
        """Reset to VLA phase at episode start."""
        self._phase = Phase.VLA_PHASE

    def trigger_critical(self) -> None:
        """Manual trigger: switch to critical phase."""
        self._phase = Phase.CRITICAL_PHASE

    def trigger_vla(self) -> None:
        """Manual trigger: switch back to VLA phase."""
        self._phase = Phase.VLA_PHASE

    def update(self, z_rl: torch.Tensor | None = None) -> Phase:
        """Update phase based on mode.

        In manual mode, phase only changes via trigger_critical/trigger_vla.
        In learned mode, classifier decides based on z_rl.

        Args:
            z_rl: (B, D) RL token embedding (required for learned mode)

        Returns:
            current phase
        """
        if self.mode == "learned" and z_rl is not None:
            if self.classifier is None:
                raise RuntimeError("Learned mode requires a classifier")
            is_critical = self.classifier.predict(z_rl, self.threshold)
            # Use first element of batch for phase decision
            if is_critical[0].item():
                self._phase = Phase.CRITICAL_PHASE
            else:
                self._phase = Phase.VLA_PHASE

        return self._phase

    def train_classifier(
        self,
        z_rl: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step for the handover classifier.

        Args:
            z_rl: (B, D) RL token embeddings
            labels: (B,) binary labels (1 = critical)
            optimizer: optimizer for classifier params

        Returns:
            loss value
        """
        if self.classifier is None:
            raise RuntimeError("No classifier to train")
        logits = self.classifier(z_rl).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
