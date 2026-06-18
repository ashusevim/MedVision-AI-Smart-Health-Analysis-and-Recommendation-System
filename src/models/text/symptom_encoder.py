"""
SymptomEncoder - Encode structured symptom data into dense embeddings.

This module maps discrete symptom identifiers and optional severity scores
into continuous vector representations. It supports two encoding modes:

* **Basic** — each present symptom contributes a learned embedding; the
  output is the sum (or mean) of all active embeddings.
* **Severity-weighted** — each symptom embedding is scaled by its associated
  severity score, enabling the model to distinguish between mild and severe
  presentations of the same symptom.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SymptomEncoder(nn.Module):
    """Encode structured symptom data into a fixed-size embedding vector.

    Internally maintains an ``nn.Embedding`` table indexed by symptom ID.
    When severity information is provided, each symptom's embedding is
    multiplied by its severity before aggregation.

    Args:
        embedding_dim: Dimensionality of the output embedding.
        num_symptoms: Total number of unique symptom identifiers.
        aggregation: How to combine per-symptom embeddings. ``"sum"`` adds
            them; ``"mean"`` averages over the number of present symptoms.
        severity_range: Expected (min, max) for severity values.  Used to
            normalise severity to ``[0, 1]`` before weighting.
        padding_idx: Index used as a placeholder / padding token.  Its
            embedding is always zeroed out.

    Raises:
        ValueError: If *num_symptoms* < 1 or *embedding_dim* < 1.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_symptoms: int = 500,
        aggregation: str = "mean",
        severity_range: tuple[float, float] = (1.0, 10.0),
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if num_symptoms < 1:
            raise ValueError("num_symptoms must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be 'sum' or 'mean'")

        self.embedding_dim = embedding_dim
        self.num_symptoms = num_symptoms
        self.aggregation = aggregation
        self.severity_min, self.severity_max = severity_range
        self.padding_idx = padding_idx

        # Learnable symptom embeddings (padding_idx row is zeroed)
        self.symptom_embedding = nn.Embedding(
            num_embeddings=num_symptoms,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        nn.init.xavier_uniform_(self.symptom_embedding.weight)
        # Re-zero the padding row after init
        with torch.no_grad():
            self.symptom_embedding.weight[self.padding_idx].zero_()

        # Projection layer for optional refinement
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        logger.info(
            "SymptomEncoder initialised — num_symptoms=%d, embedding_dim=%d, "
            "aggregation=%s, severity_range=%s",
            num_symptoms,
            embedding_dim,
            aggregation,
            severity_range,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, symptoms: torch.Tensor) -> torch.Tensor:
        """Encode a batch of symptom ID sequences into embedding vectors.

        Each row in *symptoms* is a variable-length list of symptom IDs
        (padded with ``padding_idx``).  The embeddings for present symptoms
        are aggregated according to ``aggregation``.

        Args:
            symptoms: Long tensor ``(B, S)`` where *S* is the maximum
                sequence length.  Padding positions should use
                ``padding_idx``.

        Returns:
            Embedding tensor ``(B, embedding_dim)``.
        """
        # (B, S, D)
        emb = self.symptom_embedding(symptoms)

        # Build mask: True for real symptoms, False for padding
        mask = (symptoms != self.padding_idx).unsqueeze(-1).float()  # (B, S, 1)

        # Masked aggregation
        masked_emb = emb * mask
        if self.aggregation == "sum":
            aggregated = masked_emb.sum(dim=1)
        else:
            counts = mask.sum(dim=1).clamp(min=1.0)
            aggregated = masked_emb.sum(dim=1) / counts

        return self.projection(aggregated)

    def encode_with_severity(
        self,
        symptoms: torch.Tensor,
        severities: torch.Tensor,
    ) -> torch.Tensor:
        """Encode symptoms weighted by per-symptom severity scores.

        Each symptom's embedding is scaled by its normalised severity
        before aggregation, allowing the encoder to distinguish between
        mild and severe presentations.

        Args:
            symptoms: Long tensor ``(B, S)`` of symptom IDs.
            severities: Float tensor ``(B, S)`` of severity scores in the
                range ``[severity_min, severity_max]``.  Padding positions
                should be set to 0.

        Returns:
            Embedding tensor ``(B, embedding_dim)``.
        """
        # (B, S, D)
        emb = self.symptom_embedding(symptoms)

        # Normalise severities to [0, 1]
        severity_range = self.severity_max - self.severity_min
        if severity_range > 0:
            norm_sev = (severities - self.severity_min) / severity_range
        else:
            norm_sev = severities
        norm_sev = norm_sev.clamp(0.0, 1.0).unsqueeze(-1)  # (B, S, 1)

        # Build mask for padding
        mask = (symptoms != self.padding_idx).unsqueeze(-1).float()  # (B, S, 1)

        # Weight embeddings by normalised severity
        weighted_emb = emb * norm_sev * mask

        if self.aggregation == "sum":
            aggregated = weighted_emb.sum(dim=1)
        else:
            counts = mask.sum(dim=1).clamp(min=1.0)
            aggregated = weighted_emb.sum(dim=1) / counts

        return self.projection(aggregated)

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoder output."""
        return self.embedding_dim

    def get_symptom_embedding(self, symptom_id: int) -> torch.Tensor:
        """Retrieve the raw embedding for a single symptom.

        Args:
            symptom_id: Integer symptom identifier.

        Returns:
            1-D tensor of shape ``(embedding_dim,)``.
        """
        return self.symptom_embedding.weight[symptom_id].detach()

    def get_symptom_similarity(
        self,
        id_a: int,
        id_b: int,
    ) -> float:
        """Compute cosine similarity between two symptom embeddings.

        Args:
            id_a: First symptom ID.
            id_b: Second symptom ID.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        a = self.symptom_embedding.weight[id_a].detach()
        b = self.symptom_embedding.weight[id_b].detach()
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        return sim.item()
