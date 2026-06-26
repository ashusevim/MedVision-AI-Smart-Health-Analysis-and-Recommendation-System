"""
Real-Time Inference Module for MedVision-AI.

Provides low-latency, request-driven inference with warm-up capabilities
and latency statistics tracking for production deployment scenarios.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.inference.predictor import Predictor, PredictionResult, PredictionStatus

logger = logging.getLogger(__name__)


class RequestPriority(str, Enum):
    """Priority level for incoming inference requests."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InferenceRequest:
    """Encapsulates a single real-time inference request.

    Attributes:
        request_id: Unique identifier for the request.
        input_data: The raw input (tensor, array, or dict).
        priority: Request priority level.
        timestamp: UNIX timestamp when the request was received.
        metadata: Optional request-level metadata.
    """

    request_id: str
    input_data: Any
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Structured response for a real-time inference request.

    Attributes:
        request_id: Echoes the original request identifier.
        prediction: The prediction result.
        latency_ms: End-to-end latency in milliseconds.
        server_timestamp: Server-side completion timestamp.
        status: Execution status string.
        error_message: Optional error details if processing failed.
    """

    request_id: str
    prediction: Optional[PredictionResult]
    latency_ms: float
    server_timestamp: float = field(default_factory=time.time)
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the response to a plain dictionary."""
        return {
            "request_id": self.request_id,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "latency_ms": round(self.latency_ms, 3),
            "server_timestamp": self.server_timestamp,
            "status": self.status,
            "error_message": self.error_message,
        }


class RealTimeInference:
    """Low-latency inference engine for production serving.

    ``RealTimeInference`` wraps a :class:`Predictor` and adds production
    features such as model warm-up, latency tracking, request
    prioritisation, and statistics collection.

    Args:
        config: Runtime configuration dictionary.  Expected keys include
            ``device``, ``warmup_iterations``, ``max_latency_ms``,
            and ``enable_latency_tracking``.

    Example::

        config = {"warmup_iterations": 10, "max_latency_ms": 200}
        engine = RealTimeInference(config)
        engine.warm_up()
        response = engine.process_request(request)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._predictor = Predictor(config)
        self._warmup_iterations: int = int(config.get("warmup_iterations", 5))
        self._max_latency_ms: float = float(config.get("max_latency_ms", 500.0))
        self._enable_tracking: bool = config.get("enable_latency_tracking", True)
        self._is_warmed_up: bool = False

        # Latency tracking buffers
        self._latencies: list[float] = []
        self._max_tracked_latencies: int = int(config.get("max_tracked_latencies", 10_000))
        self._error_count: int = 0
        self._total_requests: int = 0

        logger.info(
            "RealTimeInference initialised — warmup=%d, max_latency=%.1fms",
            self._warmup_iterations,
            self._max_latency_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request and return a response.

        The method measures end-to-end latency, tracks statistics, and
        applies priority-based handling for critical requests.

        Args:
            request: An :class:`InferenceRequest` with input data and
                metadata.

        Returns:
            An :class:`InferenceResponse` containing the prediction
            result and timing information.
        """
        self._total_requests += 1
        start = time.perf_counter()

        try:
            input_tensor = self._prepare_input(request.input_data)

            # Critical-priority requests bypass normal queuing
            if request.priority == RequestPriority.CRITICAL:
                prediction = self._predictor.predict(input_tensor)
            else:
                prediction = self._predictor.predict(input_tensor)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_latency(elapsed_ms)

            status = "success"
            if elapsed_ms > self._max_latency_ms:
                status = "degraded"
                logger.warning(
                    "Request %s exceeded latency threshold: %.1fms > %.1fms",
                    request.request_id,
                    elapsed_ms,
                    self._max_latency_ms,
                )

            return InferenceResponse(
                request_id=request.request_id,
                prediction=prediction,
                latency_ms=elapsed_ms,
                status=status,
            )

        except Exception as exc:
            self._error_count += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("Error processing request %s", request.request_id)
            return InferenceResponse(
                request_id=request.request_id,
                prediction=None,
                latency_ms=elapsed_ms,
                status="error",
                error_message=str(exc),
            )

    def warm_up(self) -> dict[str, Any]:
        """Execute warm-up inference passes to stabilise latency.

        Warm-up runs help ensure that JIT compilation, GPU kernel
        selection, and memory allocation are completed before real
        traffic arrives.

        Returns:
            A dictionary with warm-up statistics including average
            latency and whether the warm-up succeeded.
        """
        if self._predictor._model is None:
            logger.warning("Cannot warm up — no model attached")
            return {"status": "skipped", "reason": "no_model"}

        logger.info("Starting warm-up — %d iterations", self._warmup_iterations)
        warmup_latencies: list[float] = []

        input_shape = self._config.get("warmup_input_shape", (3, 224, 224))

        for i in range(self._warmup_iterations):
            dummy_input = torch.randn(*input_shape)
            start = time.perf_counter()
            try:
                self._predictor.predict(dummy_input)
            except Exception as exc:
                logger.warning("Warm-up iteration %d failed: %s", i, exc)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            warmup_latencies.append(elapsed_ms)

        if warmup_latencies:
            self._is_warmed_up = True
            avg_warmup = statistics.mean(warmup_latencies)
            logger.info(
                "Warm-up complete — avg latency: %.2fms", avg_warmup
            )
            return {
                "status": "completed",
                "iterations": len(warmup_latencies),
                "average_latency_ms": round(avg_warmup, 3),
                "min_latency_ms": round(min(warmup_latencies), 3),
                "max_latency_ms": round(max(warmup_latencies), 3),
            }

        return {"status": "failed", "iterations": 0}

    def get_latency_stats(self) -> dict[str, Any]:
        """Return comprehensive latency statistics for the inference engine.

        Returns:
            A dictionary with min, max, mean, median, p95, p99 latencies,
            request counts, and error rate.
        """
        if not self._latencies:
            return {
                "total_requests": self._total_requests,
                "error_count": self._error_count,
                "error_rate": 0.0,
                "latency_available": False,
                "is_warmed_up": self._is_warmed_up,
            }

        sorted_latencies = sorted(self._latencies)
        count = len(sorted_latencies)

        p95_index = int(count * 0.95)
        p99_index = int(count * 0.99)

        stats: dict[str, Any] = {
            "total_requests": self._total_requests,
            "tracked_requests": count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(self._total_requests, 1), 6),
            "latency_available": True,
            "is_warmed_up": self._is_warmed_up,
            "min_latency_ms": round(sorted_latencies[0], 3),
            "max_latency_ms": round(sorted_latencies[-1], 3),
            "mean_latency_ms": round(statistics.mean(sorted_latencies), 3),
            "median_latency_ms": round(statistics.median(sorted_latencies), 3),
            "p95_latency_ms": round(sorted_latencies[min(p95_index, count - 1)], 3),
            "p99_latency_ms": round(sorted_latencies[min(p99_index, count - 1)], 3),
            "stdev_latency_ms": round(
                statistics.stdev(sorted_latencies) if count > 1 else 0.0, 3
            ),
        }

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, input_data: Any) -> torch.Tensor:
        """Prepare raw request input into a model-ready tensor.

        Args:
            input_data: Tensor, NumPy array, dict with ``"array"`` key,
                or list that will be converted.

        Returns:
            A float32 tensor on the CPU.
        """
        if isinstance(input_data, torch.Tensor):
            return input_data.float()
        if isinstance(input_data, np.ndarray):
            return torch.from_numpy(input_data).float()
        if isinstance(input_data, dict):
            array = input_data.get("array")
            if array is not None:
                return self._prepare_input(array)
            raise ValueError("Dict input must contain an 'array' key")
        if isinstance(input_data, list):
            return torch.tensor(input_data, dtype=torch.float32)
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement, capping the buffer size.

        Args:
            latency_ms: Measured latency in milliseconds.
        """
        if not self._enable_tracking:
            return
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._max_tracked_latencies:
            # Trim oldest entries to stay within the buffer limit
            self._latencies = self._latencies[-self._max_tracked_latencies :]
