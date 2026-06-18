"""
MedicalCNN - Convolutional Neural Network model for medical image analysis.

This module provides a configurable CNN architecture supporting multiple backbone
networks (ResNet-50, EfficientNet) with pretrained weights and custom classifier
heads tailored for medical condition classification.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone registry – maps backbone name -> (model_fn, feature_dim)
# ---------------------------------------------------------------------------
_BACKBONE_REGISTRY: dict[str, Tuple[type, int]] = {
    "resnet50": (models.resnet50, 2048),
    "resnet101": (models.resnet101, 2048),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
    "efficientnet_b4": (models.efficientnet_b4, 1792),
    "efficientnet_b7": (models.efficientnet_b7, 2560),
}


class MedicalCNN(nn.Module):
    """Configurable CNN for medical image classification.

    Supports multiple pretrained backbones with a custom multi-layer classifier
    head designed for medical condition prediction. The backbone can be frozen
    or unfrozen for fine-tuning strategies.

    Args:
        num_classes: Number of output classes for the classifier head.
        backbone: Name of the backbone architecture. Must be one of
            ``resnet50``, ``resnet101``, ``efficientnet_b0``,
            ``efficientnet_b4``, ``efficientnet_b7``.
        pretrained: Whether to initialise the backbone with ImageNet weights.
        dropout_rate: Dropout probability applied in the classifier head.
        hidden_dims: Optional list of hidden-layer sizes for the classifier
            head.  When ``None`` a single hidden layer of 512 units is used.

    Raises:
        ValueError: If *backbone* is not in the supported registry.
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dims: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from {list(_BACKBONE_REGISTRY.keys())}"
            )

        self.num_classes = num_classes
        self.backbone_name = backbone
        self._pretrained = pretrained

        # ------------------------------------------------------------------
        # Build backbone
        # ------------------------------------------------------------------
        model_fn, feature_dim = _BACKBONE_REGISTRY[backbone]
        weights = "DEFAULT" if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Remove the original classification head
        if backbone.startswith("resnet"):
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone.startswith("efficientnet"):
            self.backbone = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool,
                nn.Flatten(1),
            )

        self._feature_dim = feature_dim

        # ------------------------------------------------------------------
        # Build custom classifier head
        # ------------------------------------------------------------------
        if hidden_dims is None:
            hidden_dims = [512]

        layers: list[nn.Module] = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Track whether backbone is currently frozen
        self._backbone_frozen = False

        self._initialize_classifier_weights()
        logger.info(
            "MedicalCNN initialised — backbone=%s, pretrained=%s, "
            "num_classes=%d, feature_dim=%d",
            backbone,
            pretrained,
            num_classes,
            feature_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone + classifier.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        features = self.backbone(x)
        if features.dim() == 4:
            # Global average pooling for ResNet-style backbones
            features = features.mean(dim=[2, 3])
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations from the backbone.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(B, feature_dim)``.
        """
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        return features

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters to disable gradient computation.

        This is useful during the initial phase of transfer learning where
        only the classifier head should be updated.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        logger.info("Backbone frozen — %d parameters disabled", sum(p.numel() for p in self.backbone.parameters()))

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters to enable gradient computation.

        Typically called after the classifier head has been warmed up during
        transfer learning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        logger.info("Backbone unfrozen — %d parameters enabled", sum(p.numel() for p in self.backbone.parameters()))

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector produced by the backbone."""
        return self._feature_dim

    @property
    def is_backbone_frozen(self) -> bool:
        """Whether the backbone parameters are currently frozen."""
        return self._backbone_frozen

    def _initialize_classifier_weights(self) -> None:
        """Apply Kaiming normal initialisation to the classifier head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return the total (or trainable) parameter count.

        Args:
            trainable_only: If ``True`` count only parameters that require
                gradients.

        Returns:
            Integer count of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
