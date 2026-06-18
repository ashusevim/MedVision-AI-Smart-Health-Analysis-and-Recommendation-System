"""
FusionTransformer - Transformer-based multimodal fusion for vision and text.

This module implements a fusion architecture that combines features from
vision and text modalities using a transformer encoder. Each modality is
projected into a shared hidden space, and a learnable [FUSE] token attends
over the concatenated modality tokens to produce a fused representation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModalityProjection(nn.Module):
    """Project a single modality's feature vector into the shared hidden space.

    Supports both single-vector and sequence inputs.

    Args:
        input_dim: Dimensionality of the modality's features.
        hidden_dim: Target hidden dimensionality.
        num_tokens: When > 1 the input is split into *num_tokens* sub-vectors
            before projection (useful for creating multiple "tokens" from a
            single feature vector).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tokens: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens

        if num_tokens > 1:
            self.pre_proj = nn.Linear(input_dim, hidden_dim * num_tokens)
        else:
            self.pre_proj = nn.Linear(input_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features into the shared hidden space.

        Args:
            x: Input tensor ``(B, input_dim)`` or ``(B, S, input_dim)``.

        Returns:
            Token tensor ``(B, num_tokens, hidden_dim)`` (or ``(B, S, hidden_dim)``
            when input is already a sequence and *num_tokens* == 1).
        """
        if x.dim() == 2:
            if self.num_tokens > 1:
                x = self.pre_proj(x)  # (B, hidden_dim * num_tokens)
                x = x.view(x.size(0), self.num_tokens, -1)  # (B, T, D)
            else:
                x = self.pre_proj(x).unsqueeze(1)  # (B, 1, D)
        else:
            x = self.pre_proj(x)  # (B, S, D)

        return self.dropout(self.norm(x))


class FusionTransformer(nn.Module):
    """Fuse vision and text features using a transformer encoder.

    The architecture works as follows:

    1. **Project** each modality's features into a shared hidden space via
       learnable ``ModalityProjection`` modules.
    2. **Concatenate** the projected vision tokens, text tokens, and a
       learnable ``[FUSE]`` token into a single sequence.
    3. **Encode** the sequence with a standard transformer encoder.
    4. **Extract** the ``[FUSE]`` token representation as the fused output.

    Args:
        vision_dim: Dimensionality of vision feature vectors.
        text_dim: Dimensionality of text feature vectors.
        hidden_dim: Shared hidden dimensionality used inside the transformer.
        num_heads: Number of attention heads in the transformer.
        num_layers: Number of transformer encoder layers.
        vision_tokens: Number of tokens to split vision features into.
        text_tokens: Number of tokens to split text features into.
        dropout: Dropout probability.
        num_classes: If > 0 a classification head is appended that returns
            logits; otherwise the raw fused features are returned.

    Raises:
        ValueError: If *hidden_dim* is not divisible by *num_heads*.
    """

    def __init__(
        self,
        vision_dim: int = 2048,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        vision_tokens: int = 1,
        text_tokens: int = 1,
        dropout: float = 0.1,
        num_classes: int = 0,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Modality projections
        self.vision_proj = ModalityProjection(
            input_dim=vision_dim,
            hidden_dim=hidden_dim,
            num_tokens=vision_tokens,
            dropout=dropout,
        )
        self.text_proj = ModalityProjection(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            num_tokens=text_tokens,
            dropout=dropout,
        )

        # Learnable [FUSE] token
        self.fuse_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.fuse_token, std=0.02)

        # Positional encoding (learnable)
        total_tokens = 1 + vision_tokens + text_tokens  # [FUSE] + V + T
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Optional classification head
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = None

        logger.info(
            "FusionTransformer initialised — vision_dim=%d, text_dim=%d, "
            "hidden_dim=%d, num_heads=%d, num_layers=%d, num_classes=%d",
            vision_dim, text_dim, hidden_dim, num_heads, num_layers, num_classes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse vision and text features.

        Args:
            vision_features: Vision feature tensor ``(B, vision_dim)``.
            text_features: Text feature tensor ``(B, text_dim)``.

        Returns:
            If ``num_classes > 0``: logits ``(B, num_classes)``.
            Otherwise: fused feature vector ``(B, hidden_dim)``.
        """
        fused = self._fuse(vision_features, text_features)

        if self.classifier is not None:
            return self.classifier(fused)
        return fused

    def extract_fused_features(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fused feature vector *without* the classification head.

        Useful when the ``FusionTransformer`` is used as a sub-module.

        Args:
            vision_features: ``(B, vision_dim)``
            text_features: ``(B, text_dim)``

        Returns:
            Fused features ``(B, hidden_dim)``.
        """
        return self._fuse(vision_features, text_features)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fuse(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Core fusion logic: project, concatenate, encode, extract [FUSE]."""
        B = vision_features.size(0)

        # Project modalities
        v_tokens = self.vision_proj(vision_features)  # (B, V, D)
        t_tokens = self.text_proj(text_features)       # (B, T, D)

        # Expand [FUSE] token
        fuse_tok = self.fuse_token.expand(B, -1, -1)   # (B, 1, D)

        # Concatenate: [FUSE] + vision + text
        sequence = torch.cat([fuse_tok, v_tokens, t_tokens], dim=1)  # (B, 1+V+T, D)

        # Add positional encoding
        sequence = sequence + self.pos_embed

        # Transformer encoding
        encoded = self.transformer(sequence)  # (B, 1+V+T, D)
        encoded = self.norm(encoded)

        # Extract [FUSE] token
        fused = encoded[:, 0]  # (B, D)
        return fused

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
