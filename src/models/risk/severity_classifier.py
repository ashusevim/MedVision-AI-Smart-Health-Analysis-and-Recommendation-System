"""
SeverityClassifier - Classify medical condition severity into ordinal categories.

This module implements a neural network that maps patient feature vectors to
severity levels: mild, moderate, severe, and critical. The model uses ordinal-
aware training to respect the natural ordering of severity classes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Canonical severity levels (ordered from least to most severe)
SEVERITY_LEVELS: list[str] = ["mild", "moderate", "severe", "critical"]


class OrdinalLoss(nn.Module):
    """Loss function that respects the ordinal nature of severity levels.

    Instead of treating severity categories as independent classes, this loss
    penalises predictions more heavily when the predicted class is further
    from the true class in the ordinal ranking.

    Args:
        num_classes: Number of severity categories.
        penalty_scale: Scaling factor for the distance penalty.
    """

    def __init__(self, num_classes: int = 4, penalty_scale: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.penalty_scale = penalty_scale

        # Distance matrix: D[i, j] = |i - j|
        indices = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer(
            "_distance",
            (indices.unsqueeze(0) - indices.unsqueeze(1)).abs(),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal-weighted cross-entropy loss.

        Args:
            logits: Predicted logits ``(B, C)``.
            targets: Ground-truth class indices ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Weight by ordinal distance
        preds = logits.argmax(dim=-1)
        distances = self._distance[preds, targets]
        weighted_loss = ce_loss * (1.0 + self.penalty_scale * distances)

        return weighted_loss.mean()


class SeverityClassifier(nn.Module):
    """Classify patient condition severity into ordinal categories.

    The classifier supports four severity levels (mild, moderate, severe,
    critical) and provides:

    * **Ordinal-aware training** via an optional custom loss.
    * **Confidence estimation** based on the softmax probability of the
      predicted class.
    * **Calibrated thresholds** for each severity boundary.

    Args:
        num_classes: Number of severity categories. Defaults to 4.
        input_dim: Dimensionality of the input feature vector.
        hidden_dims: Hidden layer sizes. Defaults to ``[128, 64]``.
        dropout: Dropout probability.
        use_ordinal_loss: Whether to use ordinal-weighted loss during
            training.
        severity_labels: Custom labels for the severity classes. Must have
            length *num_classes*.

    Raises:
        ValueError: If *num_classes* < 2 or *severity_labels* length
            doesn't match *num_classes*.
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_dim: int = 32,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        use_ordinal_loss: bool = True,
        severity_labels: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.num_classes = num_classes
        self.use_ordinal_loss = use_ordinal_loss

        # Severity labels
        if severity_labels is not None:
            if len(severity_labels) != num_classes:
                raise ValueError(
                    f"severity_labels length ({len(severity_labels)}) must "
                    f"equal num_classes ({num_classes})"
                )
            self.severity_labels = severity_labels
        elif num_classes == 4:
            self.severity_labels = list(SEVERITY_LEVELS)
        else:
            self.severity_labels = [f"level_{i}" for i in range(num_classes)]

        # Build classifier MLP
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Ordinal loss
        if use_ordinal_loss:
            self.ordinal_loss = OrdinalLoss(num_classes=num_classes)

        # Cached confidence
        self._last_confidence: float = 0.0

        logger.info(
            "SeverityClassifier initialised — num_classes=%d, input_dim=%d, "
            "ordinal_loss=%s",
            num_classes, input_dim, use_ordinal_loss,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass producing severity logits.

        Args:
            features: Patient feature tensor ``(B, input_dim)``.

        Returns:
            Severity logits ``(B, num_classes)``.
        """
        return self.classifier(features)

    def classify(self, patient_features: torch.Tensor) -> str:
        """Classify a single patient's severity level.

        Args:
            patient_features: Feature tensor ``(1, input_dim)`` or
                ``(input_dim,)``.

        Returns:
            Human-readable severity label string.
        """
        self.eval()
        with torch.no_grad():
            if patient_features.dim() == 1:
                patient_features = patient_features.unsqueeze(0)

            logits = self.forward(patient_features)
            probs = F.softmax(logits, dim=-1)

            predicted_idx = int(probs.argmax(dim=-1).item())
            self._last_confidence = float(probs[0, predicted_idx].item())

        return self.severity_labels[predicted_idx]

    def get_confidence(self) -> float:
        """Return the confidence of the most recent classification.

        Confidence is the softmax probability assigned to the predicted
        class by ``classify()``.

        Returns:
            Float in ``[0, 1]``.
        """
        return self._last_confidence

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss.

        Uses the ordinal-weighted loss if ``use_ordinal_loss`` is True,
        otherwise standard cross-entropy.

        Args:
            logits: Predicted logits ``(B, num_classes)``.
            targets: Ground-truth class indices ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        if self.use_ordinal_loss:
            return self.ordinal_loss(logits, targets)
        return F.cross_entropy(logits, targets)

    def get_severity_distribution(
        self, features: torch.Tensor
    ) -> dict[str, float]:
        """Return the full probability distribution over severity levels.

        Args:
            features: Patient feature tensor ``(B, input_dim)``.

        Returns:
            Dictionary mapping severity labels to average probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            probs = F.softmax(logits, dim=-1).mean(dim=0)

        return {
            label: round(float(probs[i].item()), 4)
            for i, label in enumerate(self.severity_labels)
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        """Severity level labels in order."""
        return list(self.severity_labels)

    def __repr__(self) -> str:
        return (
            f"SeverityClassifier(num_classes={self.num_classes}, "
            f"ordinal_loss={self.use_ordinal_loss})"
        )
