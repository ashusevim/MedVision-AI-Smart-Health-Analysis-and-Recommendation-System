"""
MedicalViT - Vision Transformer model for medical image analysis.

This module implements a Vision Transformer (ViT) architecture specifically
designed for medical imaging tasks. It supports configurable patch size,
image dimensions, attention heads, and transformer depth, with options for
extracting intermediate features and attention maps.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert input images into a sequence of patch embeddings.

    Args:
        image_size: Expected spatial size of input images (square).
        patch_size: Size of each square patch.
        in_channels: Number of input image channels.
        embed_dim: Dimensionality of patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image into patch tokens.

        Args:
            x: Image tensor ``(B, C, H, W)``.

        Returns:
            Patch embeddings ``(B, num_patches, embed_dim)``.
        """
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of self-attention heads.
        mlp_ratio: Hidden dim = embed_dim * mlp_ratio.
        dropout: Dropout rate for attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input token tensor ``(B, N, D)``.
            return_attention: Whether to return the attention weight matrix.

        Returns:
            Tuple of (output tensor, optional attention weights).
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed, need_weights=return_attention
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x, None


class MedicalViT(nn.Module):
    """Vision Transformer for medical image classification.

    Implements a full ViT pipeline: patch embedding → positional encoding →
    transformer encoder → classification head. Provides hooks for feature
    extraction and attention-map visualisation.

    Args:
        num_classes: Number of output classes.
        image_size: Spatial size of input images (square).
        patch_size: Size of each square patch.
        in_channels: Number of input image channels.
        embed_dim: Token embedding dimension.
        num_heads: Number of self-attention heads.
        num_layers: Number of transformer encoder layers.
        mlp_ratio: MLP hidden-dim expansion ratio.
        dropout: Dropout probability.
        use_cls_token: Whether to prepend a learnable [CLS] token.

    Raises:
        ValueError: If *image_size* is not divisible by *patch_size*.
    """

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )

        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Optional [CLS] token
        num_tokens = self.patch_embed.num_patches + (1 if use_cls_token else 0)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None  # type: ignore[assignment]

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        logger.info(
            "MedicalViT initialised — image_size=%d, patch_size=%d, "
            "embed_dim=%d, num_heads=%d, num_layers=%d, num_classes=%d",
            image_size, patch_size, embed_dim, num_heads, num_layers, num_classes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Input images ``(B, C, H, W)``.

        Returns:
            Classification logits ``(B, num_classes)``.
        """
        features = self.extract_features(x)
        logits = self.head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the global feature representation.

        Returns the normalised representation corresponding to the ``[CLS]``
        token (or mean-pooled patch tokens when *use_cls_token* is ``False``).

        Args:
            x: Input images ``(B, C, H, W)``.

        Returns:
            Feature tensor ``(B, embed_dim)``.
        """
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)  # (B, num_patches, D)

        # Prepend CLS token if configured
        if self.use_cls_token and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Add positional encoding
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # Transformer encoder
        for block in self.encoder_blocks:
            tokens, _ = block(tokens, return_attention=False)

        tokens = self.norm(tokens)

        # Pool to a single vector
        if self.use_cls_token:
            features = tokens[:, 0]
        else:
            features = tokens.mean(dim=1)

        return features

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract per-layer attention weight matrices.

        Useful for visualisation and interpretability of the model's focus
        regions in medical images.

        Args:
            x: Input images ``(B, C, H, W)``.

        Returns:
            List of attention tensors, one per encoder layer.  Each tensor has
            shape ``(B, num_heads, N, N)`` where ``N`` is the number of tokens.
        """
        B = x.shape[0]

        tokens = self.patch_embed(x)
        if self.use_cls_token and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        attention_maps: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            tokens, attn_weights = block(tokens, return_attention=True)
            if attn_weights is not None:
                attention_maps.append(attn_weights)

        return attention_maps

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return parameter count.

        Args:
            trainable_only: Count only parameters requiring gradients.

        Returns:
            Integer parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
