"""
Predictor Module for MedVision-AI.

Encapsulates single-sample and batch prediction logic, model management,
and prediction metadata retrieval for medical imaging inference.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PredictionStatus(str, Enum):
    """Status of a prediction result."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class PredictionResult:
    """Structured container for a single prediction outcome.

    Attributes:
        prediction_id: Unique identifier for this prediction.
        label: Predicted class or finding label.
        confidence: Confidence score in the range [0.0, 1.0].
        probabilities: Per-class probability distribution.
        status: Execution status of the prediction.
        metadata: Additional metadata such as model version and timing.
        alternatives: Top-N alternative predictions with confidences.
        explainability_map: Optional attention / grad-cam heatmap array.
    """

    prediction_id: str
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    status: PredictionStatus = PredictionStatus.SUCCESS
    metadata: dict[str, Any] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    explainability_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the prediction result to a plain dictionary."""
        result: dict[str, Any] = {
            "prediction_id": self.prediction_id,
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "status": self.status.value,
            "metadata": self.metadata,
            "alternatives": self.alternatives,
        }
        if self.explainability_map is not None:
            result["explainability_map_shape"] = list(self.explainability_map.shape)
        return result

    @property
    def is_reliable(self) -> bool:
        """Return ``True`` when the confidence exceeds the reliability threshold."""
        return self.confidence >= 0.7 and self.status == PredictionStatus.SUCCESS


class Predictor:
    """Primary prediction engine for MedVision-AI.

    The ``Predictor`` wraps a PyTorch model and exposes a clean interface
    for single-sample and batch inference.  It handles device placement,
    inference-mode context management, confidence calibration, and
    prediction metadata bookkeeping.

    Args:
        config: Runtime configuration dictionary.  Expected keys include
            ``device`` (``"cpu"`` | ``"cuda"``), ``confidence_threshold``,
            ``top_k``, and ``model_version``.

    Example::

        config = {"device": "cuda", "confidence_threshold": 0.5, "top_k": 5}
        predictor = Predictor(config)
        predictor.set_model(my_model)
        result = predictor.predict(input_tensor)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._device = torch.device(config.get("device", "cpu"))
        self._confidence_threshold: float = float(config.get("confidence_threshold", 0.5))
        self._top_k: int = int(config.get("top_k", 5))
        self._model: Optional[nn.Module] = None
        self._model_version: str = config.get("model_version", "1.0.0")
        self._class_labels: list[str] = config.get("class_labels", [])
        self._prediction_count: int = 0
        self._total_inference_time: float = 0.0

        logger.info(
            "Predictor initialised — device=%s, threshold=%.2f, top_k=%d",
            self._device,
            self._confidence_threshold,
            self._top_k,
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def set_model(self, model: nn.Module) -> None:
        """Attach a trained model to the predictor.

        The model is moved to the configured device and set to evaluation
        mode automatically.

        Args:
            model: A PyTorch ``nn.Module`` ready for inference.
        """
        self._model = model.to(self._device)
        self._model.eval()
        logger.info("Model attached and moved to %s", self._device)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, input_data: torch.Tensor | np.ndarray) -> PredictionResult:
        """Run inference on a single input sample.

        Args:
            input_data: A single input tensor of shape ``(C, H, W)`` or
                a NumPy array that will be converted internally.

        Returns:
            A ``PredictionResult`` containing the label, confidence,
            probability distribution, and execution metadata.

        Raises:
            RuntimeError: If no model has been set via :meth:`set_model`.
        """
        if self._model is None:
            raise RuntimeError("No model attached. Call set_model() first.")

        tensor = self._to_tensor(input_data)
        start = time.perf_counter()

        with torch.inference_mode():
            logits = self._model(tensor.unsqueeze(0).to(self._device))
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        elapsed = time.perf_counter() - start
        self._prediction_count += 1
        self._total_inference_time += elapsed

        return self._build_prediction_result(probabilities, elapsed)

    def predict_batch(self, inputs: list[torch.Tensor | np.ndarray]) -> list[PredictionResult]:
        """Run inference on a batch of input samples.

        The inputs are stacked into a single batch tensor so that the
        model receives them in one forward pass for maximum throughput.

        Args:
            inputs: A list of input tensors or NumPy arrays.

        Returns:
            A list of ``PredictionResult`` objects, one per input sample.

        Raises:
            RuntimeError: If no model has been set via :meth:`set_model`.
            ValueError: If *inputs* is empty.
        """
        if self._model is None:
            raise RuntimeError("No model attached. Call set_model() first.")
        if not inputs:
            raise ValueError("inputs must contain at least one element.")

        tensors = [self._to_tensor(item) for item in inputs]
        batch = torch.stack(tensors).to(self._device)

        start = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        elapsed = time.perf_counter() - start

        per_sample_time = elapsed / len(inputs)
        results: list[PredictionResult] = []
        for idx in range(probabilities.shape[0]):
            self._prediction_count += 1
            self._total_inference_time += per_sample_time
            results.append(self._build_prediction_result(probabilities[idx], per_sample_time, sample_index=idx))
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_prediction_metadata(self) -> dict[str, Any]:
        """Return aggregate metadata about the predictor's usage.

        Returns:
            A dictionary containing model version, device, total
            predictions, average latency, and configuration snapshot.
        """
        avg_latency = (
            self._total_inference_time / self._prediction_count
            if self._prediction_count > 0
            else 0.0
        )
        return {
            "model_version": self._model_version,
            "device": str(self._device),
            "total_predictions": self._prediction_count,
            "average_latency_ms": round(avg_latency * 1000, 3),
            "confidence_threshold": self._confidence_threshold,
            "top_k": self._top_k,
            "class_labels": self._class_labels,
            "model_attached": self._model is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert *data* to a float32 tensor on the CPU."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data.float()

    def _build_prediction_result(
        self,
        probabilities: np.ndarray,
        elapsed: float,
        sample_index: int | None = None,
    ) -> PredictionResult:
        """Construct a ``PredictionResult`` from a probability vector.

        Args:
            probabilities: 1-D probability array (sums to 1.0).
            elapsed: Wall-clock inference time in seconds.
            sample_index: Optional positional index within a batch.

        Returns:
            A fully populated ``PredictionResult``.
        """
        top_indices = np.argsort(probabilities)[::-1]
        top_idx = int(top_indices[0])
        confidence = float(probabilities[top_idx])
        label = self._class_labels[top_idx] if self._class_labels and top_idx < len(self._class_labels) else str(top_idx)

        prob_dict: dict[str, float] = {}
        for i, idx in enumerate(top_indices[: self._top_k]):
            key = self._class_labels[int(idx)] if self._class_labels and int(idx) < len(self._class_labels) else str(idx)
            prob_dict[key] = float(probabilities[int(idx)])

        alternatives = [
            {
                "label": self._class_labels[int(idx)] if self._class_labels and int(idx) < len(self._class_labels) else str(idx),
                "confidence": float(probabilities[int(idx)]),
            }
            for idx in top_indices[1 : self._top_k]
        ]

        status = PredictionStatus.SUCCESS
        if confidence < self._confidence_threshold:
            status = PredictionStatus.UNCERTAIN
        elif 0.5 <= confidence < 0.7:
            status = PredictionStatus.PARTIAL

        prediction_id = hashlib.sha256(
            f"{label}{time.time_ns()}{sample_index}".encode()
        ).hexdigest()[:16]

        metadata: dict[str, Any] = {
            "model_version": self._model_version,
            "device": str(self._device),
            "inference_time_ms": round(elapsed * 1000, 3),
            "input_shape": None,
            "timestamp": time.time(),
        }

        return PredictionResult(
            prediction_id=prediction_id,
            label=label,
            confidence=confidence,
            probabilities=prob_dict,
            status=status,
            metadata=metadata,
            alternatives=alternatives,
        )
