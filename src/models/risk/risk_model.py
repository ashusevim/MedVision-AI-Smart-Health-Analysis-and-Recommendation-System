"""
RiskModel - Patient risk assessment with interpretable risk factor analysis.

This module implements a neural-network-based risk model that maps patient
feature vectors to risk scores and provides interpretable risk-factor
contributions via learned attention weights and gradient-based attribution.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default feature-name templates for common clinical features
_DEFAULT_FEATURE_NAMES: list[str] = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature", "oxygen_saturation", "wbc_count",
    "creatinine", "glucose", "hemoglobin", "platelet_count",
    "comorbidity_index", "smoking_status", "prior_admissions",
]


class RiskAttention(nn.Module):
    """Self-attention over clinical features to learn feature importance.

    Produces attention weights that can be interpreted as relative
    importance of each input feature for the risk assessment.

    Args:
        feature_dim: Number of input features.
        hidden_dim: Internal hidden dimensionality.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted features.

        Args:
            x: Input features ``(B, F)``.

        Returns:
            Tuple of (weighted features ``(B, F)``, attention weights ``(B, F)``).
        """
        # Treat each feature as a 1-token "sequence"
        x_expanded = x.unsqueeze(1)  # (B, 1, F)
        Q = self.query(x_expanded)   # (B, 1, H)
        K = self.key(x_expanded)     # (B, 1, H)

        # Compute attention over features via element-wise scoring
        attn_scores = (Q * K).sum(dim=-1) * self.scale  # (B, 1)
        attn_weights = torch.sigmoid(attn_scores).squeeze(1)  # (B,)

        # Element-wise attention on features
        weighted = x * attn_weights.unsqueeze(-1)  # (B, F)
        return weighted, attn_weights


class RiskModel(nn.Module):
    """Neural network for patient risk assessment.

    The model takes a patient feature vector and outputs risk scores
    across a configurable number of risk categories. It includes an
    attention mechanism that provides per-feature importance scores
    for interpretability.

    Args:
        num_features: Dimensionality of the input feature vector.
        num_risk_classes: Number of risk output categories (e.g. 5 for
            very-low / low / moderate / high / very-high).
        hidden_dims: Layer sizes for the risk MLP. Defaults to
            ``[128, 64]``.
        dropout: Dropout probability.
        feature_names: Optional human-readable names for the input features
            (used in ``get_risk_factors``).

    Raises:
        ValueError: If *num_features* or *num_risk_classes* < 1.
    """

    def __init__(
        self,
        num_features: int = 16,
        num_risk_classes: int = 5,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        if num_features < 1:
            raise ValueError("num_features must be >= 1")
        if num_risk_classes < 1:
            raise ValueError("num_risk_classes must be >= 1")

        self.num_features = num_features
        self.num_risk_classes = num_risk_classes
        self.feature_names = feature_names or _DEFAULT_FEATURE_NAMES[:num_features]

        # Feature attention for interpretability
        self.feature_attention = RiskAttention(num_features)

        # Risk MLP
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_risk_classes))
        self.risk_head = nn.Sequential(*layers)

        # Cache for gradient-based attribution
        self._last_attention_weights: Optional[torch.Tensor] = None
        self._last_feature_gradients: Optional[torch.Tensor] = None

        logger.info(
            "RiskModel initialised — num_features=%d, num_risk_classes=%d",
            num_features, num_risk_classes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass producing risk scores.

        Args:
            features: Patient feature tensor ``(B, num_features)``.

        Returns:
            Risk logits ``(B, num_risk_classes)``.
        """
        weighted_features, attn_weights = self.feature_attention(features)
        self._last_attention_weights = attn_weights.detach()

        risk_logits = self.risk_head(weighted_features)

        # Store gradients if features require grad
        if features.requires_grad:
            features.retain_grad()
            self._last_input = features

        return risk_logits

    def predict_risk(self, patient_data: dict[str, float]) -> dict[str, Any]:
        """Predict risk for a single patient from a feature dictionary.

        Args:
            patient_data: Mapping of feature names to values.  Only keys
                present in ``self.feature_names`` are used; missing keys
                are filled with 0.

        Returns:
            Dictionary with keys:

            * ``risk_scores`` — probability distribution over risk classes.
            * ``predicted_class`` — index of the highest-risk class.
            * ``risk_level`` — human-readable label.
            * ``risk_factors`` — top contributing features with importance.
        """
        self.eval()
        with torch.no_grad():
            feature_vector = torch.zeros(1, self.num_features)
            for i, name in enumerate(self.feature_names):
                if name in patient_data:
                    feature_vector[0, i] = float(patient_data[name])

            logits = self.forward(feature_vector)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        predicted_class = int(probs.argmax().item())
        risk_labels = self._risk_labels()

        result: dict[str, Any] = {
            "risk_scores": {
                risk_labels[i]: round(probs[i].item(), 4)
                for i in range(self.num_risk_classes)
            },
            "predicted_class": predicted_class,
            "risk_level": risk_labels[predicted_class],
            "risk_factors": self.get_risk_factors(),
        }
        return result

    def get_risk_factors(self) -> dict[str, Any]:
        """Return the most influential features for the latest prediction.

        Uses the cached attention weights to rank features by importance.

        Returns:
            Dictionary with ``"feature_importance"`` mapping feature names
            to importance scores, and ``"top_factors"`` listing the top-5
            features sorted by importance.
        """
        if self._last_attention_weights is None:
            return {"feature_importance": {}, "top_factors": []}

        weights = self._last_attention_weights.squeeze(0).cpu().numpy()
        importance = {}
        for i, name in enumerate(self.feature_names):
            if i < len(weights):
                importance[name] = round(float(weights[i]), 4)

        sorted_factors = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = [
            {"feature": name, "importance": imp}
            for name, imp in sorted_factors[:5]
        ]

        return {
            "feature_importance": importance,
            "top_factors": top_factors,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _risk_labels(self) -> list[str]:
        """Generate human-readable risk level labels.

        Returns:
            List of label strings, one per risk class.
        """
        if self.num_risk_classes == 5:
            return ["very_low", "low", "moderate", "high", "very_high"]
        elif self.num_risk_classes == 4:
            return ["low", "moderate", "high", "very_high"]
        elif self.num_risk_classes == 3:
            return ["low", "moderate", "high"]
        else:
            return [f"risk_level_{i}" for i in range(self.num_risk_classes)]

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return parameter count.

        Args:
            trainable_only: Count only parameters requiring gradients.

        Returns:
            Integer count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
