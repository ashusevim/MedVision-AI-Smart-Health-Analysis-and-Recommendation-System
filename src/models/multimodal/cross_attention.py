"""
CrossAttention - Cross-modal attention mechanism for multimodal fusion.

This module implements cross-attention where one modality provides the query
while another provides the key/value. This is essential for allowing the model
to attend from one modality (e.g. text) to another (e.g. vision), enabling
fine-grained alignment between modalities.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """Cross-attention between two modalities.

    Computes attention where the *query* originates from one modality and
    the *key* / *value* originate from another.  This enables information
    flow from the key/value modality into the query modality.

    The implementation supports:

    * Single-vector inputs (automatically unsqueezed to token sequences).
    * Sequence inputs (e.g. patch tokens, token embeddings).
    * Caching of attention weights for downstream visualisation.

    Args:
        query_dim: Dimensionality of the query modality features.
        key_dim: Dimensionality of the key/value modality features.
        num_heads: Number of attention heads. The head dimension is
            ``query_proj_dim // num_heads``.
        output_dim: Dimensionality of the output projection. Defaults to
            ``query_dim``.
        dropout: Dropout probability on attention weights.
        bias: Whether to include bias in the projection layers.

    Raises:
        ValueError: If *query_dim* or *key_dim* is not divisible by
            *num_heads*.
    """

    def __init__(
        self,
        query_dim: int = 512,
        key_dim: int = 512,
        num_heads: int = 8,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or query_dim
        self.head_dim = self.output_dim // num_heads

        if self.output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim ({self.output_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        # Linear projections
        self.q_proj = nn.Linear(query_dim, self.output_dim, bias=bias)
        self.k_proj = nn.Linear(key_dim, self.output_dim, bias=bias)
        self.v_proj = nn.Linear(key_dim, self.output_dim, bias=bias)
        self.out_proj = nn.Linear(self.output_dim, self.output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.output_dim)

        # Cache for attention weights (populated after forward)
        self._attention_weights: Optional[torch.Tensor] = None

        # Scale factor
        self._scale = math.sqrt(self.head_dim)

        self._init_weights()

        logger.info(
            "CrossAttention initialised — query_dim=%d, key_dim=%d, "
            "num_heads=%d, output_dim=%d, head_dim=%d",
            query_dim, key_dim, num_heads, self.output_dim, self.head_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-attention output.

        Args:
            query: Query tensor from the primary modality.
                Shape ``(B, Lq, Dq)`` or ``(B, Dq)``.
            key: Key tensor from the secondary modality.
                Shape ``(B, Lk, Dk)`` or ``(B, Dk)``.
            value: Value tensor from the secondary modality (typically same as
                *key*). Shape ``(B, Lk, Dk)`` or ``(B, Dk)``.
            attention_mask: Optional mask ``(B, Lk)`` where ``True`` /
                non-zero positions are *ignored* (compatible with
                ``nn.MultiheadAttention`` convention).

        Returns:
            Output tensor ``(B, Lq, output_dim)`` (or ``(B, output_dim)`` if
            input was a single vector).
        """
        single_query = query.dim() == 2
        single_kv = key.dim() == 2

        if single_query:
            query = query.unsqueeze(1)  # (B, 1, Dq)
        if single_kv:
            key = key.unsqueeze(1)      # (B, 1, Dk)
            value = value.unsqueeze(1)   # (B, 1, Dk)

        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        # Project Q, K, V
        Q = self.q_proj(query)  # (B, Lq, output_dim)
        K = self.k_proj(key)    # (B, Lk, output_dim)
        V = self.v_proj(value)  # (B, Lk, output_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, d)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, d)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, d)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # (B, H, Lq, Lk)

        if attention_mask is not None:
            # attention_mask: (B, Lk) -> (B, 1, 1, Lk)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, Lq, Lk)
        attn_weights = self.dropout(attn_weights)

        # Cache weights for later retrieval
        self._attention_weights = attn_weights.detach()

        # Weighted sum
        context = torch.matmul(attn_weights, V)  # (B, H, Lq, d)
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.output_dim)

        # Output projection + residual-style norm
        output = self.out_proj(context)
        output = self.norm(output)

        if single_query:
            output = output.squeeze(1)  # (B, output_dim)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the cached attention weights from the most recent forward pass.

        Returns:
            Attention weight tensor ``(B, num_heads, Lq, Lk)`` or ``None``
            if no forward pass has been executed yet.
        """
        return self._attention_weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def head_dimension(self) -> int:
        """Per-head dimensionality."""
        return self.head_dim

    def __repr__(self) -> str:
        return (
            f"CrossAttention(query_dim={self.query_dim}, key_dim={self.key_dim}, "
            f"num_heads={self.num_heads}, output_dim={self.output_dim})"
        )
