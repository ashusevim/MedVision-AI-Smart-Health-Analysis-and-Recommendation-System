"""
Inference Pipeline for MedVision-AI
=====================================

Provides a production-ready inference pipeline for medical AI models,
supporting batch processing, timing instrumentation, and robust error handling.

Typical usage::

    pipeline = InferencePipeline(config)
    pipeline.load_model("models/chest_xray_v1.pt")
    result = pipeline({"image": image_array})

    # Or batch processing:
    results = pipeline.run_batch(batch_data)
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & types
# ---------------------------------------------------------------------------

class InferenceMode(Enum):
    """Supported inference execution modes."""

    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class InferenceConfig:
    """Configuration container for the inference pipeline.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized model weights / checkpoint.
    device : str
        Compute device identifier (e.g. ``"cpu"``, ``"cuda:0"``).
    batch_size : int
        Maximum batch size for batched inference.
    precision : str
        Numerical precision – ``"fp32"``, ``"fp16"``, or ``"bf16"``.
    num_workers : int
        Number of data-loading worker processes.
    timeout_seconds : float
        Maximum wall-clock time (seconds) per inference call.
    mode : InferenceMode
        Inference execution mode.
    output_dir : str or Path
        Directory where inference outputs are persisted.
    return_probabilities : bool
        Whether to include class probabilities in the output.
    threshold : float
        Decision threshold for binary classification outputs.
    """

    model_path: Union[str, Path] = "models/latest.pt"
    device: str = "cpu"
    batch_size: int = 32
    precision: str = "fp32"
    num_workers: int = 4
    timeout_seconds: float = 120.0
    mode: InferenceMode = InferenceMode.SINGLE
    output_dir: Union[str, Path] = "outputs/inference"
    return_probabilities: bool = True
    threshold: float = 0.5


@dataclass
class InferenceResult:
    """Container for a single inference result.

    Attributes
    ----------
    prediction : Any
        The model prediction (class label, segmentation mask, etc.).
    probabilities : Optional[numpy.ndarray]
        Class probability vector (when ``return_probabilities`` is *True*).
    latency_ms : float
        End-to-end inference latency in milliseconds.
    metadata : Dict[str, Any]
        Arbitrary metadata attached to the result.
    """

    prediction: Any = None
    probabilities: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timing(func: Callable) -> Callable:
    """Decorator that measures and logs the wall-clock execution time.

    The elapsed time (in milliseconds) is stored on the returned result
    object under ``result.metadata["timing"][function_name]`` when the
    result is an :class:`InferenceResult`.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "%s failed after %.2f ms: %s", func.__qualname__, elapsed_ms, exc
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "%s completed in %.2f ms", func.__qualname__, elapsed_ms
        )
        # Attach timing metadata to InferenceResult when possible
        if isinstance(result, InferenceResult):
            result.latency_ms = elapsed_ms
            result.metadata.setdefault("timing", {})[func.__name__] = elapsed_ms
        elif isinstance(result, dict) and "metadata" in result:
            result["metadata"].setdefault("timing", {})[func.__name__] = elapsed_ms
            result["latency_ms"] = elapsed_ms
        elif isinstance(result, list):
            for r in result:
                if isinstance(r, InferenceResult):
                    r.metadata.setdefault("timing", {})[func.__name__] = elapsed_ms
        return result

    return wrapper


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """End-to-end inference pipeline for medical AI models.

    The pipeline encapsulates the full inference workflow:

    1. **Load model** – deserialize weights and prepare the model for
       evaluation.
    2. **Preprocess input** – apply modality-specific preprocessing
       (resizing, normalization, etc.).
    3. **Run inference** – execute the forward pass.
    4. **Postprocess output** – map raw model outputs to clinically
       meaningful predictions.

    Parameters
    ----------
    config : dict or InferenceConfig
        Pipeline configuration.  A plain dict is automatically converted
        into an :class:`InferenceConfig` instance.

    Examples
    --------
    >>> config = {"model_path": "models/chest_xray_v1.pt", "device": "cpu"}
    >>> pipeline = InferencePipeline(config)
    >>> pipeline.load_model("models/chest_xray_v1.pt")
    >>> result = pipeline({"image": image_array})
    >>> result["prediction"]
    'pneumonia'
    """

    def __init__(self, config: Union[Dict[str, Any], InferenceConfig]) -> None:
        if isinstance(config, dict):
            self._config = InferenceConfig(**{
                k: v for k, v in config.items()
                if k in InferenceConfig.__dataclass_fields__
            })
        else:
            self._config = config
        self._model: Optional[Any] = None
        self._is_loaded: bool = False
        self._model_path: Optional[str] = None
        self._preprocessor_cache: Dict[str, Any] = {}
        logger.info(
            "InferencePipeline initialised (model_path=%s, device=%s, batch_size=%d)",
            self._config.model_path,
            self._config.device,
            self._config.batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the model weights from disk and prepare for inference.

        Parameters
        ----------
        model_path : str, optional
            Path to the model checkpoint.  If *None*, falls back to the
            path specified in the pipeline configuration.

        Raises
        ------
        FileNotFoundError
            If the model checkpoint cannot be located.
        RuntimeError
            If the model fails to load or is incompatible with the
            current configuration.
        """
        path = model_path or str(self._config.model_path)
        model_path_obj = Path(path)
        if not model_path_obj.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path_obj.resolve()}"
            )
        logger.info("Loading model from %s ...", model_path_obj)
        try:
            # In a real implementation this would use torch.load or
            # an equivalent framework-specific loader.
            self._model = self._load_checkpoint(model_path_obj)
            self._model = self._prepare_model_for_inference(self._model)
            self._model_path = path
            self._is_loaded = True
            logger.info("Model loaded successfully from %s.", path)
        except Exception as exc:
            self._is_loaded = False
            raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc

    @timing
    def preprocess_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modality-specific preprocessing to raw input data.

        Parameters
        ----------
        data : Dict[str, Any]
            Raw input data dictionary.  Supported keys include
            ``"image"`` (numpy array), ``"text"`` (str), and
            ``"clinical"`` (array-like).  If the dict contains an
            ``"items"`` key with a list, each item is preprocessed
            individually for batch support.

        Returns
        -------
        Dict[str, Any]
            Preprocessed data ready for the model forward pass.
        """
        self._ensure_model_loaded()
        result: Dict[str, Any] = {}

        # Batch support: if "items" key is present, preprocess each item
        if "items" in data and isinstance(data["items"], (list, tuple)):
            result["items"] = [self._preprocess_single_item(item) for item in data["items"]]
            result["batch_size"] = len(data["items"])
            return result

        # Single-item preprocessing
        return self._preprocess_single_item(data)

    @timing
    def run_inference(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model forward pass on preprocessed input.

        Parameters
        ----------
        preprocessed : Dict[str, Any]
            Data returned by :meth:`preprocess_input`.

        Returns
        -------
        Dict[str, Any]
            Raw model output dictionary with ``"raw_output"`` key and
            optional ``"batch_outputs"`` for batch inputs.
        """
        self._ensure_model_loaded()
        batch_size = preprocessed.get("batch_size", 1)
        logger.debug("Running inference on batch of size %d", batch_size)

        try:
            if "items" in preprocessed:
                # Batch inference
                outputs = []
                for item in preprocessed["items"]:
                    output = self._forward(item)
                    outputs.append(output)
                return {"batch_outputs": outputs, "batch_size": len(outputs)}
            else:
                output = self._forward(preprocessed)
                return {"raw_output": output}
        except Exception as exc:
            logger.error("Inference forward pass failed: %s", exc)
            raise RuntimeError(f"Inference forward pass failed: {exc}") from exc

    @timing
    def postprocess_output(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw model output into a structured inference result.

        Parameters
        ----------
        raw_output : Dict[str, Any]
            Output dictionary from :meth:`run_inference`.

        Returns
        -------
        Dict[str, Any]
            Structured result dictionary with ``"prediction"``,
            ``"probabilities"``, ``"latency_ms"``, and ``"metadata"`` keys.
            For batch outputs, returns ``"results"`` with a list of
            per-item result dictionaries.
        """
        # Handle batch outputs
        if "batch_outputs" in raw_output:
            results = []
            for output in raw_output["batch_outputs"]:
                results.append(self._postprocess_single(output))
            return {
                "results": results,
                "batch_size": raw_output.get("batch_size", len(results)),
                "metadata": {"model_path": self._model_path or str(self._config.model_path)},
            }

        # Single output
        raw = raw_output.get("raw_output")
        result = self._postprocess_single(raw)
        result["metadata"] = {
            "model_path": self._model_path or str(self._config.model_path),
            "threshold": self._config.threshold,
        }
        return result

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full inference pipeline: preprocess → infer → postprocess.

        Parameters
        ----------
        data : Dict[str, Any]
            Raw input data dictionary.

        Returns
        -------
        Dict[str, Any]
            Structured inference result with prediction, probabilities,
            and metadata.
        """
        self._ensure_model_loaded()
        preprocessed = self.preprocess_input(data)
        raw_output = self.run_inference(preprocessed)
        return self.postprocess_output(raw_output)

    def run_batch(
        self,
        batch: Sequence[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of inputs with automatic mini-batching.

        Parameters
        ----------
        batch : Sequence[Dict[str, Any]]
            Collection of raw input dictionaries.
        show_progress : bool
            If *True*, log progress at each mini-batch boundary.

        Returns
        -------
        List[Dict[str, Any]]
            One result dictionary per input item, preserving order.
        """
        self._ensure_model_loaded()
        results: List[Dict[str, Any]] = []
        batch_size = self._config.batch_size
        total = len(batch)

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            mini_batch = batch[start_idx:end_idx]
            if show_progress:
                logger.info("Processing mini-batch %d–%d / %d", start_idx, end_idx, total)
            for item in mini_batch:
                results.append(self(item))
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been successfully loaded."""
        return self._is_loaded

    @property
    def config(self) -> InferenceConfig:
        """Current pipeline configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Guard that raises if the model has not been loaded."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

    def _preprocess_single_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single input item dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Input data with optional ``"image"``, ``"text"``, ``"clinical"``
            keys.

        Returns
        -------
        Dict[str, Any]
            Preprocessed input dictionary.
        """
        result: Dict[str, Any] = {}
        if "image" in data:
            image = np.asarray(data["image"])
            result["image"] = self._preprocess_image(image)
        if "text" in data:
            result["text"] = self._preprocess_text(data["text"])
        if "clinical" in data:
            result["clinical"] = np.asarray(data["clinical"], dtype=np.float32)
        # Pass through any additional keys
        for key in data:
            if key not in result and key not in ("image", "text", "clinical"):
                result[key] = data[key]
        return result

    @staticmethod
    def _load_checkpoint(path: Path) -> Any:
        """Simulate loading a model checkpoint from *path*.

        In production this would delegate to the appropriate framework
        loader (PyTorch, TensorFlow, ONNX Runtime, etc.).
        """
        logger.debug("Attempting to load checkpoint from %s", path)
        # Placeholder – returns a callable that mimics a model
        return lambda x: np.random.rand(2) if x is not None else np.zeros(2)

    def _prepare_model_for_inference(self, model: Any) -> Any:
        """Set the model to evaluation mode and apply precision settings.

        Parameters
        ----------
        model : Any
            The loaded model object.

        Returns
        -------
        Any
            The model ready for inference.
        """
        # In a real implementation this would call model.eval() and
        # optionally apply torch.cuda.amp or half-precision casting.
        logger.debug("Preparing model for inference (precision=%s)", self._config.precision)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Normalise and reshape an image array for model input.

        Parameters
        ----------
        image : numpy.ndarray
            Raw image pixel data.

        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        # Scale to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        # Standardise
        mean = np.mean(image)
        std = np.std(image) + 1e-8
        image = (image - mean) / std
        return image

    def _preprocess_text(self, text: str) -> Dict[str, Any]:
        """Tokenise and encode text for model input.

        Parameters
        ----------
        text : str
            Raw clinical text.

        Returns
        -------
        Dict[str, Any]
            Tokenised representation with attention masks, etc.
        """
        # Simplified tokenisation placeholder
        tokens = text.lower().split()
        return {
            "input_ids": np.array(tokens, dtype=object),
            "attention_mask": np.ones(len(tokens), dtype=np.int64),
        }

    def _forward(self, processed_input: Any) -> Any:
        """Execute the model forward pass.

        Parameters
        ----------
        processed_input : Any
            Preprocessed input.

        Returns
        -------
        Any
            Raw model output.
        """
        if callable(self._model):
            return self._model(processed_input)
        raise RuntimeError("Model is not callable – was load_model() called correctly?")

    def _postprocess_single(self, raw_output: Any) -> Dict[str, Any]:
        """Post-process a single raw model output into a result dictionary.

        Parameters
        ----------
        raw_output : Any
            Raw model output (logits, probabilities, etc.).

        Returns
        -------
        Dict[str, Any]
            Result dictionary with ``"prediction"`` and ``"probabilities"``.
        """
        probabilities = None
        prediction = None

        if isinstance(raw_output, np.ndarray):
            if raw_output.ndim == 1 and raw_output.size >= 2:
                # Softmax-style probability vector
                probabilities = self._softmax(raw_output)
                prediction = int(np.argmax(probabilities))
            elif raw_output.size == 1:
                # Binary / regression output
                prob = float(raw_output.flat[0])
                probabilities = np.array([1.0 - prob, prob])
                prediction = 1 if prob >= self._config.threshold else 0
            else:
                prediction = raw_output
        elif isinstance(raw_output, dict):
            prediction = raw_output.get("prediction", raw_output)
            probabilities = raw_output.get("probabilities")
            if probabilities is not None and isinstance(probabilities, np.ndarray):
                probabilities = self._softmax(probabilities)
        else:
            prediction = raw_output

        return {
            "prediction": prediction,
            "probabilities": probabilities if self._config.return_probabilities else None,
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Parameters
        ----------
        logits : numpy.ndarray
            1-D array of raw logit scores.

        Returns
        -------
        numpy.ndarray
            Probability distribution with the same shape.
        """
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
