"""
ClinicalBERTEncoder - Clinical text encoding with HuggingFace Transformers.

This module provides a wrapper around HuggingFace BERT-family models
(preferably Bio_ClinicalBERT) for encoding clinical narratives, discharge
summaries, and other medical text into dense vector representations suitable
for downstream tasks such as classification, similarity search, or multimodal
fusion.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default model identifier from HuggingFace Model Hub
_DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class ClinicalBERTEncoder(nn.Module):
    """Encode clinical text using a pretrained BERT-family model.

    Wraps ``AutoModel`` and ``AutoTokenizer`` from HuggingFace Transformers,
    providing convenient methods for single-text and batch encoding, as well
    as extraction of hidden states for interpretability.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``emilyalsentzer/Bio_ClinicalBERT``.
        max_length: Maximum token length for padding / truncation.
        device: Device string (``"cpu"`` or ``"cuda"``). Auto-detected when
            ``None``.
        pooling_strategy: How to reduce token-level representations into a
            single vector. ``"cls"`` uses the [CLS] token; ``"mean"`` averages
            all token representations (excluding padding).
        freeze_encoder: If ``True``, the BERT parameters are frozen after
            initialisation (useful for feature-extraction-only pipelines).

    Raises:
        ImportError: If ``transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_length: int = 512,
        device: Optional[str] = None,
        pooling_strategy: str = "cls",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required for ClinicalBERTEncoder. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert = self.bert.to(self._device)

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        self._hidden_size = self.bert.config.hidden_size

        logger.info(
            "ClinicalBERTEncoder initialised — model=%s, max_length=%d, "
            "pooling=%s, device=%s, frozen=%s",
            model_name,
            max_length,
            pooling_strategy,
            self._device,
            freeze_encoder,
        )

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize(
        self,
        texts: Union[str, list[str]],
    ) -> dict[str, torch.Tensor]:
        """Tokenize input texts and move tensors to the target device.

        Args:
            texts: Single string or list of strings.

        Returns:
            Dictionary of tokenised tensors (input_ids, attention_mask, …).
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Reduce token-level hidden states to a single vector per sample.

        Args:
            last_hidden: ``(B, L, D)`` tensor from the encoder.
            attention_mask: ``(B, L)`` mask with 1 for real tokens.

        Returns:
            Pooled tensor ``(B, D)``.
        """
        if self.pooling_strategy == "cls":
            return last_hidden[:, 0, :]
        elif self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        else:
            raise ValueError(
                f"Unknown pooling strategy '{self.pooling_strategy}'. "
                "Choose 'cls' or 'mean'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: Union[str, list[str]]) -> torch.Tensor:
        """Encode one or more texts into pooled embedding vectors.

        Args:
            texts: A single string or a list of strings.

        Returns:
            Embedding tensor of shape ``(B, hidden_size)``.
        """
        was_single = isinstance(texts, str)
        if was_single:
            texts = [texts]

        inputs = self._tokenize(texts)

        with torch.no_grad():
            outputs = self.bert(**inputs)

        embeddings = self._pool(outputs.last_hidden_state, inputs["attention_mask"])

        if was_single:
            embeddings = embeddings.squeeze(0)

        return embeddings

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode a large list of texts in mini-batches.

        Args:
            texts: List of clinical text strings.
            batch_size: Number of texts per forward pass.
            show_progress: If ``True``, log progress every 10 batches.

        Returns:
            Embedding tensor ``(N, hidden_size)`` on the device.
        """
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.encode(batch)
            all_embeddings.append(emb)

            if show_progress and (i // batch_size) % 10 == 0:
                logger.info(
                    "Encoded %d / %d texts", min(i + batch_size, len(texts)), len(texts)
                )

        result = torch.cat(all_embeddings, dim=0)
        logger.info("Batch encoding complete — %d texts encoded", result.shape[0])
        return result

    def get_hidden_states(self, text: str) -> tuple[torch.Tensor, ...]:
        """Return per-layer hidden states for a single text.

        Requires the underlying model to be configured with
        ``output_hidden_states=True``.

        Args:
            text: Input clinical text.

        Returns:
            Tuple of tensors, one per encoder layer + the embedding layer.
            Each tensor has shape ``(1, L, D)``.
        """
        inputs = self._tokenize(text)

        with torch.no_grad():
            outputs = self.bert(**inputs, output_hidden_states=True)

        return outputs.hidden_states  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # nn.Module overrides
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (useful when integrating as a sub-module).

        Args:
            input_ids: Token IDs ``(B, L)``.
            attention_mask: Attention mask ``(B, L)``.

        Returns:
            Pooled embeddings ``(B, hidden_size)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs.last_hidden_state, attention_mask)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the output embeddings."""
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        """Current device of the model parameters."""
        return self._device
