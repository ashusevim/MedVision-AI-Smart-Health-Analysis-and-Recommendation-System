"""
FeatureExtractor - Wrapper for pretrained models to extract deep feature vectors.

This module provides a high-level interface for extracting feature representations
from image batches or entire dataloaders using configurable pretrained backbones.
Features can be used for downstream tasks such as similarity search, clustering,
or as input to multimodal fusion models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

logger = logging.getLogger(__name__)

# Mapping of model names to (torchvision builder, feature_dim)
_MODEL_REGISTRY: dict[str, tuple[type, int]] = {
    "resnet50": (models.resnet50, 2048),
    "resnet101": (models.resnet101, 2048),
    "resnet152": (models.resnet152, 2048),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
    "efficientnet_b4": (models.efficientnet_b4, 1792),
    "efficientnet_b7": (models.efficientnet_b7, 2560),
    "vgg16": (models.vgg16, 4096),
    "densenet121": (models.densenet121, 1024),
}


def _strip_classifier(model: nn.Module, model_name: str) -> nn.Module:
    """Remove the final classification layer, keeping the feature backbone.

    Args:
        model: The pretrained torchvision model.
        model_name: Key used to look up the model architecture.

    Returns:
        An ``nn.Sequential`` that outputs a feature vector.
    """
    if model_name.startswith("resnet"):
        # ResNet: everything except the last fc layer
        return nn.Sequential(*list(model.children())[:-1])

    if model_name.startswith("efficientnet"):
        return nn.Sequential(model.features, model.avgpool, nn.Flatten(1))

    if model_name.startswith("vgg"):
        return nn.Sequential(model.features, model.avgpool, nn.Flatten(1))

    if model_name.startswith("densenet"):
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

    raise ValueError(f"Cannot strip classifier for unknown model '{model_name}'")


class FeatureExtractor:
    """Extract deep feature vectors from images using a pretrained backbone.

    The extractor wraps a torchvision model with its classification head
    removed so that only the feature representation is returned.  It supports
    both single-batch extraction and full-dataloader iteration.

    Args:
        model_name: Name of the pretrained model. Must be a key in the
            internal model registry.
        device: Device to run inference on (``"cpu"`` or ``"cuda"``).
            When ``None`` the device is auto-detected.
        batch_size: Default batch size used by ``extract_from_dataloader``.

    Raises:
        ValueError: If *model_name* is not supported.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self._batch_size = batch_size

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        model_fn, feature_dim = _MODEL_REGISTRY[model_name]
        full_model = model_fn(weights="DEFAULT")
        self.model = _strip_classifier(full_model, model_name)
        self._feature_dim = feature_dim

        self.model = self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            "FeatureExtractor ready — model=%s, feature_dim=%d, device=%s",
            model_name,
            feature_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Extract features from a single batch of images.

        Args:
            image_batch: Image tensor ``(B, C, H, W)``.  Values should be
                normalised consistently with the model's expected input
                distribution (e.g. ImageNet mean/std).

        Returns:
            Feature tensor ``(B, feature_dim)`` on the same device as the
            input batch.
        """
        image_batch = image_batch.to(self.device)
        with torch.no_grad():
            features = self.model(image_batch)
        # Collapse spatial dims if present (e.g. ResNet outputs (B, D, 1, 1))
        if features.dim() > 2:
            features = features.flatten(2).mean(dim=2)
        return features

    def extract_from_dataloader(self, dataloader: DataLoader) -> np.ndarray:
        """Extract features for every sample in a dataloader.

        Iterates over all batches, concatenates the resulting feature vectors,
        and returns them as a single NumPy array.

        Args:
            dataloader: A PyTorch ``DataLoader`` yielding ``(images, ...)``.
                Only the first element of each batch tuple is used.

        Returns:
            NumPy array of shape ``(N, feature_dim)`` where *N* is the total
            number of samples.
        """
        all_features: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                features = self.extract(images)
                all_features.append(features.cpu().numpy())

        result = np.concatenate(all_features, axis=0)
        logger.info(
            "Extracted features for %d samples from dataloader", result.shape[0]
        )
        return result

    def get_feature_dim(self) -> int:
        """Return the dimensionality of the extracted feature vectors.

        Returns:
            Integer feature dimension.
        """
        return self._feature_dim

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(model={self.model_name}, "
            f"feature_dim={self._feature_dim}, device={self.device})"
        )
