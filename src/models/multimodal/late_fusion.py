"""
LateFusion - Combine modality-specific predictions with configurable strategies.

This module implements late (decision-level) fusion where each modality's
independent predictions are combined into a final prediction.  Supported
strategies include simple averaging, weighted averaging, and learned gating.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Available late-fusion strategies."""

    AVERAGE = "average"
    WEIGHTED = "weighted"
    LEARNED = "learned"


class LearnedGating(nn.Module):
    """Learn a gating distribution over modalities from their prediction logits.

    A small MLP takes the concatenated predictions and outputs a normalised
    weight for each modality via softmax.

    Args:
        num_modalities: Number of input modalities.
        num_classes: Number of prediction classes (used to determine input size).
        hidden_dim: Hidden layer size for the gating MLP.
    """

    def __init__(
        self,
        num_modalities: int,
        num_classes: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = num_modalities * num_classes
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1),
        )

    def forward(self, predictions_list: list[torch.Tensor]) -> torch.Tensor:
        """Compute gating weights.

        Args:
            predictions_list: List of ``(B, C)`` tensors, one per modality.

        Returns:
            Weight tensor ``(B, M)`` where M = number of modalities.
        """
        concat = torch.cat(predictions_list, dim=-1)  # (B, M*C)
        return self.gate(concat)  # (B, M)


class LateFusion(nn.Module):
    """Combine modality-specific predictions using a late-fusion strategy.

    Three fusion strategies are supported:

    * **average** — Simple unweighted average of all modality predictions.
    * **weighted** — Use a fixed, user-specified weight per modality.
    * **learned** — Learn a gating network that dynamically weights
      modalities based on the concatenated prediction logits.

    Args:
        num_modalities: Number of input modalities.
        num_classes: Number of prediction classes per modality.
        fusion_strategy: One of ``"average"``, ``"weighted"``, or ``"learned"``.
        modality_weights: Explicit weights for ``"weighted"`` strategy. Must
            sum to 1.0 and have length *num_modalities*. Ignored for other
            strategies.
        gating_hidden_dim: Hidden dimension for the learned gating network
            (only used when ``fusion_strategy="learned"``).
        temperature: Softmax temperature applied to predictions before fusion.
            Values > 1 soften the distribution; < 1 sharpen it.  A value of
            1.0 leaves predictions unchanged.

    Raises:
        ValueError: If *fusion_strategy* is invalid or *modality_weights*
            doesn't match *num_modalities*.
    """

    def __init__(
        self,
        num_modalities: int = 2,
        num_classes: int = 10,
        fusion_strategy: str = "weighted",
        modality_weights: Optional[list[float]] = None,
        gating_hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.temperature = temperature

        # Validate and set strategy
        try:
            self.fusion_strategy = FusionStrategy(fusion_strategy)
        except ValueError:
            raise ValueError(
                f"Invalid fusion_strategy '{fusion_strategy}'. "
                f"Choose from {[s.value for s in FusionStrategy]}"
            )

        # Strategy-specific setup
        if self.fusion_strategy == FusionStrategy.WEIGHTED:
            if modality_weights is None:
                # Default: equal weights
                modality_weights = [1.0 / num_modalities] * num_modalities
            if len(modality_weights) != num_modalities:
                raise ValueError(
                    f"modality_weights length ({len(modality_weights)}) must "
                    f"equal num_modalities ({num_modalities})"
                )
            weight_sum = sum(modality_weights)
            if abs(weight_sum - 1.0) > 1e-6:
                logger.warning(
                    "modality_weights sum to %.4f — normalising to 1.0", weight_sum
                )
                modality_weights = [w / weight_sum for w in modality_weights]
            self.register_buffer(
                "_modality_weights",
                torch.tensor(modality_weights, dtype=torch.float32),
            )

        elif self.fusion_strategy == FusionStrategy.LEARNED:
            self.gating = LearnedGating(
                num_modalities=num_modalities,
                num_classes=num_classes,
                hidden_dim=gating_hidden_dim,
            )

        logger.info(
            "LateFusion initialised — num_modalities=%d, num_classes=%d, "
            "strategy=%s, temperature=%.2f",
            num_modalities,
            num_classes,
            fusion_strategy,
            temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, predictions_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse modality-specific predictions into a single prediction.

        Args:
            predictions_list: List of prediction tensors, one per modality.
                Each tensor has shape ``(B, num_classes)`` and typically
                contains logits or probabilities.

        Returns:
            Fused prediction tensor ``(B, num_classes)``.

        Raises:
            ValueError: If the number of prediction tensors doesn't match
                *num_modalities*.
        """
        if len(predictions_list) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} prediction tensors, "
                f"got {len(predictions_list)}"
            )

        # Optionally apply temperature scaling
        if self.temperature != 1.0:
            predictions_list = [
                F.log_softmax(p / self.temperature, dim=-1).exp()
                for p in predictions_list
            ]

        # Stack into (B, M, C)
        stacked = torch.stack(predictions_list, dim=1)

        if self.fusion_strategy == FusionStrategy.AVERAGE:
            fused = stacked.mean(dim=1)

        elif self.fusion_strategy == FusionStrategy.WEIGHTED:
            weights = self._modality_weights.view(1, -1, 1)  # (1, M, 1)
            fused = (stacked * weights).sum(dim=1)

        elif self.fusion_strategy == FusionStrategy.LEARNED:
            gate_weights = self.gating(predictions_list)  # (B, M)
            gate_weights = gate_weights.unsqueeze(-1)     # (B, M, 1)
            fused = (stacked * gate_weights).sum(dim=1)

        else:
            raise RuntimeError(f"Unhandled fusion strategy: {self.fusion_strategy}")

        return fused

    def get_modality_weights(self) -> torch.Tensor:
        """Return the effective weight assigned to each modality.

        * For ``"average"``: returns equal weights ``[1/M, …]``.
        * For ``"weighted"``: returns the fixed weight vector.
        * For ``"learned"``: returns the most recent gating weights from the
          last forward pass (or equal weights if no forward pass yet).

        Returns:
            1-D weight tensor of length *num_modalities*.
        """
        if self.fusion_strategy == FusionStrategy.AVERAGE:
            return torch.ones(self.num_modalities) / self.num_modalities

        elif self.fusion_strategy == FusionStrategy.WEIGHTED:
            return self._modality_weights.clone()

        elif self.fusion_strategy == FusionStrategy.LEARNED:
            # If gating has been called, return cached; otherwise equal
            if hasattr(self, "_last_gate_weights") and self._last_gate_weights is not None:
                return self._last_gate_weights
            return torch.ones(self.num_modalities) / self.num_modalities

        return torch.ones(self.num_modalities) / self.num_modalities

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """Current fusion strategy name."""
        return self.fusion_strategy.value

    def __repr__(self) -> str:
        return (
            f"LateFusion(num_modalities={self.num_modalities}, "
            f"num_classes={self.num_classes}, strategy='{self.strategy}')"
        )
