"""
Batch Inference Module for MedVision-AI.

Provides high-throughput, chunk-based batch inference over large datasets
with result aggregation, progress tracking, and configurable output formats.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.inference.predictor import Predictor, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for a batch inference run.

    Attributes:
        chunk_size: Number of samples per processing chunk.
        max_workers: Maximum parallel workers for chunk processing.
        output_format: Output serialization format (``"json"`` | ``"csv"``).
        save_intermediate: Whether to persist intermediate chunk results.
        confidence_threshold: Minimum confidence to include in the report.
        include_explainability: Whether to compute explainability maps.
    """

    chunk_size: int = 32
    max_workers: int = 4
    output_format: str = "json"
    save_intermediate: bool = False
    confidence_threshold: float = 0.5
    include_explainability: bool = False


@dataclass
class ChunkResult:
    """Result container for a single processed chunk.

    Attributes:
        chunk_index: Positional index of the chunk.
        predictions: Prediction results for the chunk.
        processing_time: Wall-clock time to process this chunk.
        sample_count: Number of samples in this chunk.
        errors: Any errors encountered during processing.
    """

    chunk_index: int
    predictions: list[PredictionResult]
    processing_time: float
    sample_count: int
    errors: list[str] = field(default_factory=list)


class BatchInference:
    """High-throughput batch inference engine.

    ``BatchInference`` splits a large dataset into chunks, processes each
    chunk using a :class:`Predictor`, and aggregates partial results into
    a consolidated output.  It supports configurable chunk sizes, parallel
    processing, and multiple output formats.

    Args:
        config: Configuration dictionary.  Keys are forwarded to
            :class:`BatchConfig` and the :class:`Predictor` constructor.

    Example::

        config = {"chunk_size": 64, "output_format": "json"}
        batch_engine = BatchInference(config)
        summary = batch_engine.run("data/chest_xrays/", "output/results.json")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._batch_config = BatchConfig(
            chunk_size=config.get("chunk_size", 32),
            max_workers=config.get("max_workers", 4),
            output_format=config.get("output_format", "json"),
            save_intermediate=config.get("save_intermediate", False),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            include_explainability=config.get("include_explainability", False),
        )
        self._predictor = Predictor(config)
        self._total_chunks: int = 0
        self._total_samples: int = 0
        self._all_chunk_results: list[ChunkResult] = []

        logger.info(
            "BatchInference initialised — chunk_size=%d, max_workers=%d, format=%s",
            self._batch_config.chunk_size,
            self._batch_config.max_workers,
            self._batch_config.output_format,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_source: str | Path, output_path: str | Path) -> dict[str, Any]:
        """Execute a full batch inference pipeline.

        The pipeline loads data from *data_source*, splits it into chunks,
        processes each chunk, aggregates results, and writes the output
        to *output_path*.

        Args:
            data_source: Path to a directory or file containing input data.
            output_path: Destination path for the aggregated results.

        Returns:
            A summary dictionary with total predictions, accuracy metrics,
            and processing statistics.
        """
        start_time = time.perf_counter()
        data_source = Path(data_source)
        output_path = Path(output_path)

        logger.info("Starting batch inference — source=%s", data_source)

        # Load data items from source
        items = self._load_data_source(data_source)
        self._total_samples = len(items)

        # Split into chunks
        chunks = self._split_into_chunks(items)
        self._total_chunks = len(chunks)

        # Process each chunk sequentially (parallel execution via
        # ThreadPoolExecutor can be added as a future enhancement)
        self._all_chunk_results = []
        for idx, chunk in enumerate(chunks):
            chunk_result = self.process_chunk(chunk, chunk_index=idx)
            self._all_chunk_results.append(chunk_result)

            if self._batch_config.save_intermediate:
                self._save_intermediate(chunk_result, output_path.parent / f"chunk_{idx}.json")

        # Aggregate all chunk results
        aggregated = self.aggregate_results(self._all_chunk_results)

        # Persist final output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_output(aggregated, output_path)

        total_time = time.perf_counter() - start_time
        summary = {
            **aggregated,
            "total_processing_time_s": round(total_time, 3),
            "source": str(data_source),
            "output_path": str(output_path),
        }

        logger.info(
            "Batch inference complete — %d samples, %.2fs",
            self._total_samples,
            total_time,
        )
        return summary

    def process_chunk(
        self,
        chunk: list[dict[str, Any]],
        chunk_index: int = 0,
    ) -> ChunkResult:
        """Process a single chunk of input samples.

        Args:
            chunk: A list of sample dictionaries, each expected to have
                an ``"input"`` key containing a tensor or array.
            chunk_index: Positional index used for logging and tracking.

        Returns:
            A :class:`ChunkResult` with predictions and timing info.
        """
        start = time.perf_counter()
        predictions: list[PredictionResult] = []
        errors: list[str] = []

        for sample_idx, sample in enumerate(chunk):
            try:
                input_data = sample.get("input")
                if input_data is None:
                    raise ValueError(f"Sample {sample_idx} missing 'input' key")

                input_tensor = self._prepare_input(input_data)
                result = self._predictor.predict(input_tensor)
                predictions.append(result)
            except Exception as exc:
                error_msg = f"chunk={chunk_index}, sample={sample_idx}: {exc}"
                errors.append(error_msg)
                logger.warning("Prediction error — %s", error_msg)

        elapsed = time.perf_counter() - start
        return ChunkResult(
            chunk_index=chunk_index,
            predictions=predictions,
            processing_time=elapsed,
            sample_count=len(chunk),
            errors=errors,
        )

    def aggregate_results(self, chunks: list[ChunkResult]) -> dict[str, Any]:
        """Aggregate results from multiple processed chunks.

        Computes summary statistics across all chunks including label
        distributions, average confidence, and per-class metrics.

        Args:
            chunks: A list of :class:`ChunkResult` objects.

        Returns:
            A dictionary with aggregated predictions, statistics, and
            per-class breakdowns.
        """
        all_predictions: list[PredictionResult] = []
        total_errors: list[str] = []
        total_processing_time: float = 0.0
        total_samples: int = 0

        label_counts: dict[str, int] = {}
        confidence_sum: dict[str, float] = {}
        confidence_values: list[float] = []

        for chunk in chunks:
            total_processing_time += chunk.processing_time
            total_samples += chunk.sample_count
            total_errors.extend(chunk.errors)

            for pred in chunk.predictions:
                all_predictions.append(pred)
                confidence_values.append(pred.confidence)

                label_counts[pred.label] = label_counts.get(pred.label, 0) + 1
                confidence_sum[pred.label] = confidence_sum.get(pred.label, 0.0) + pred.confidence

        avg_confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )

        per_class_avg_confidence: dict[str, float] = {
            label: round(confidence_sum.get(label, 0.0) / count, 6)
            for label, count in label_counts.items()
        }

        high_confidence_count = sum(
            1 for c in confidence_values if c >= self._batch_config.confidence_threshold
        )

        return {
            "total_samples": total_samples,
            "total_predictions": len(all_predictions),
            "total_errors": len(total_errors),
            "errors": total_errors[:50],  # Cap stored errors
            "total_processing_time_s": round(total_processing_time, 3),
            "average_latency_ms": round(
                (total_processing_time / max(total_samples, 1)) * 1000, 3
            ),
            "average_confidence": round(avg_confidence, 6),
            "high_confidence_ratio": round(
                high_confidence_count / max(len(confidence_values), 1), 6
            ),
            "label_distribution": label_counts,
            "per_class_avg_confidence": per_class_avg_confidence,
            "predictions": [p.to_dict() for p in all_predictions],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data_source(self, data_source: Path) -> list[dict[str, Any]]:
        """Load input samples from a data source directory or file.

        Args:
            data_source: Path to the input data.

        Returns:
            A list of sample dictionaries.

        Raises:
            FileNotFoundError: If *data_source* does not exist.
        """
        if not data_source.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")

        items: list[dict[str, Any]] = []

        if data_source.is_file():
            with open(data_source, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        elif data_source.is_dir():
            for file_path in sorted(data_source.rglob("*")):
                if file_path.suffix in {".npy", ".npz", ".pt", ".json"}:
                    items.append({"input": str(file_path), "source": str(file_path)})

        logger.info("Loaded %d samples from %s", len(items), data_source)
        return items

    def _split_into_chunks(
        self, items: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Split *items* into chunks of configured size.

        Args:
            items: Flat list of sample dictionaries.

        Returns:
            A list of chunked sample lists.
        """
        chunk_size = self._batch_config.chunk_size
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _prepare_input(self, input_data: Any) -> torch.Tensor:
        """Prepare raw input data into a model-ready tensor.

        Args:
            input_data: A file path string, NumPy array, or tensor.

        Returns:
            A float32 tensor suitable for model input.
        """
        if isinstance(input_data, str):
            path = Path(input_data)
            if path.suffix == ".npy":
                array = np.load(str(path))
            elif path.suffix == ".npz":
                npz = np.load(str(path))
                array = next(iter(npz.values()))
            elif path.suffix == ".pt":
                return torch.load(str(path), map_location="cpu", weights_only=True).float()
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            return torch.from_numpy(array).float()
        if isinstance(input_data, np.ndarray):
            return torch.from_numpy(input_data).float()
        if isinstance(input_data, torch.Tensor):
            return input_data.float()
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _save_intermediate(self, chunk_result: ChunkResult, path: Path) -> None:
        """Persist intermediate chunk results to disk.

        Args:
            chunk_result: The chunk result to save.
            path: Destination file path.
        """
        payload = {
            "chunk_index": chunk_result.chunk_index,
            "sample_count": chunk_result.sample_count,
            "processing_time_s": round(chunk_result.processing_time, 3),
            "predictions": [p.to_dict() for p in chunk_result.predictions],
            "errors": chunk_result.errors,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def _write_output(self, aggregated: dict[str, Any], output_path: Path) -> None:
        """Write aggregated results to disk in the configured format.

        Args:
            aggregated: Aggregated result dictionary.
            output_path: Destination file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._batch_config.output_format == "json":
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(aggregated, fh, indent=2, default=str)
        elif self._batch_config.output_format == "csv":
            self._write_csv_output(aggregated, output_path)
        else:
            raise ValueError(
                f"Unsupported output format: {self._batch_config.output_format}"
            )

    @staticmethod
    def _write_csv_output(aggregated: dict[str, Any], output_path: Path) -> None:
        """Write predictions as a CSV file.

        Args:
            aggregated: Aggregated result dictionary.
            output_path: Destination file path.
        """
        import csv

        predictions = aggregated.get("predictions", [])
        if not predictions:
            return

        fieldnames = ["prediction_id", "label", "confidence", "status"]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for pred in predictions:
                writer.writerow(
                    {
                        "prediction_id": pred.get("prediction_id", ""),
                        "label": pred.get("label", ""),
                        "confidence": pred.get("confidence", 0.0),
                        "status": pred.get("status", ""),
                    }
                )
