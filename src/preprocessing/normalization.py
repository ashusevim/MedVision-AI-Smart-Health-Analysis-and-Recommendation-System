"""
Normalization Module for MedVision-AI.

Provides normalization utilities supporting both numpy arrays and PyTorch
tensors.  Implements min-max, z-score, and robust normalization with
configurable parameters and full denormalization support for reversing
transformations.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import torch to avoid hard dependency at module level
_torch: Any = None


def _get_torch() -> Any:
    """Lazily import and cache the torch module."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for tensor normalization. "
                "Install it with: pip install torch"
            )
    return _torch


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class NormalizeMethod(str, Enum):
    """Supported normalization methods."""

    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"


@dataclass
class NormalizerConfig:
    """Configuration for the Normalizer.

    Attributes:
        default_method: Default normalization method.
        feature_range: Target range for min-max normalization.
        clip: Whether to clip values to the feature range after normalization.
        epsilon: Small constant to prevent division by zero.
        quantile_range: (low, high) quantile range for robust normalization.
        axis: Axis along which to compute statistics. ``None`` means global.
        keep_params: Whether to store parameters for denormalization.
    """

    default_method: NormalizeMethod = NormalizeMethod.ZSCORE
    feature_range: tuple[float, float] = (0.0, 1.0)
    clip: bool = True
    epsilon: float = 1e-8
    quantile_range: tuple[float, float] = (25.0, 75.0)
    axis: Optional[int] = None
    keep_params: bool = True


@dataclass
class NormParams:
    """Stored normalization parameters for denormalization.

    Attributes:
        method: The normalization method used.
        min_val: Minimum value (min-max).
        max_val: Maximum value (min-max).
        mean_val: Mean value (z-score).
        std_val: Standard deviation (z-score).
        median_val: Median value (robust).
        q_low: Lower quantile (robust).
        q_high: Upper quantile (robust).
        iqr: Interquartile range (robust).
        feature_range: Target range (min-max).
        epsilon: Epsilon used.
        axis: Axis used for computation.
        original_shape: Shape of the original data.
    """

    method: NormalizeMethod = NormalizeMethod.ZSCORE
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[Any] = None
    std_val: Optional[Any] = None
    median_val: Optional[Any] = None
    q_low: Optional[Any] = None
    q_high: Optional[Any] = None
    iqr: Optional[Any] = None
    feature_range: tuple[float, float] = (0.0, 1.0)
    epsilon: float = 1e-8
    axis: Optional[int] = None
    original_shape: Optional[tuple[int, ...]] = None


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArrayType = Union[np.ndarray, Any]  # Any = torch.Tensor (lazy)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Normalizer:
    """Normalizer supporting numpy arrays and PyTorch tensors.

    Implements min-max, z-score, and robust normalization with full
    parameter storage for reversible (de)normalization.  Automatically
    detects the input type and dispatches accordingly.

    Args:
        config: A :class:`NormalizerConfig` instance.

    Example::

        normalizer = Normalizer(NormalizerConfig(default_method="minmax"))
        normalized = normalizer.normalize(data_array, method="minmax")
        restored = normalizer.denormalize(normalized, normalizer.params)
    """

    def __init__(self, config: Optional[NormalizerConfig] = None) -> None:
        self._config = config or NormalizerConfig()
        self._params: Optional[NormParams] = None
        self._param_history: list[NormParams] = []
        logger.info(
            "Normalizer initialised (method=%s, feature_range=%s)",
            self._config.default_method.value,
            self._config.feature_range,
        )

    @property
    def params(self) -> Optional[NormParams]:
        """Return the most recent normalization parameters."""
        return self._params

    @property
    def param_history(self) -> list[NormParams]:
        """Return the full history of normalization parameters."""
        return self._param_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        data: ArrayType,
        method: Optional[Union[NormalizeMethod, str]] = None,
    ) -> ArrayType:
        """Normalize *data* using the specified method.

        Args:
            data: Input numpy array or PyTorch tensor.
            method: Normalization method.  Defaults to config default.

        Returns:
            Normalized data of the same type and shape.

        Raises:
            TypeError: If *data* is not a numpy array or torch tensor.
        """
        if isinstance(method, str):
            method = NormalizeMethod(method.lower())
        method = method or self._config.default_method

        is_numpy = isinstance(data, np.ndarray)
        is_torch = not is_numpy and _is_torch_tensor(data)

        if not is_numpy and not is_torch:
            raise TypeError(
                f"Expected numpy.ndarray or torch.Tensor, got {type(data).__name__}"
            )

        if is_numpy:
            result, params = self._normalize_numpy(data, method)
        else:
            result, params = self._normalize_torch(data, method)

        # Store params
        if self._config.keep_params:
            self._params = params
            self._param_history.append(params)

        logger.debug(
            "Normalized data (method=%s, shape=%s)",
            method.value,
            params.original_shape,
        )
        return result

    def denormalize(
        self,
        data: ArrayType,
        params: Optional[NormParams] = None,
    ) -> ArrayType:
        """Reverse a previously applied normalization.

        Args:
            data: Normalized numpy array or PyTorch tensor.
            params: Normalization parameters to use for denormalization.
                Falls back to the stored ``self.params`` when ``None``.

        Returns:
            Denormalized data in the original scale.

        Raises:
            ValueError: If no parameters are available.
        """
        p = params or self._params
        if p is None:
            raise ValueError(
                "No normalization parameters available. "
                "Either pass params or run normalize() first with keep_params=True."
            )

        is_numpy = isinstance(data, np.ndarray)

        if is_numpy:
            return self._denormalize_numpy(data, p)
        else:
            return self._denormalize_torch(data, p)

    # ------------------------------------------------------------------
    # Method-specific public methods
    # ------------------------------------------------------------------

    def min_max_normalize(
        self,
        data: ArrayType,
        feature_range: Optional[tuple[float, float]] = None,
    ) -> ArrayType:
        """Apply min-max normalization.

        Transforms data to the range ``feature_range`` using the formula::

            X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

        Args:
            data: Input data.
            feature_range: Target (min, max) range. Defaults to config value.

        Returns:
            Normalized data.
        """
        # Temporarily override feature_range
        original_range = self._config.feature_range
        if feature_range is not None:
            self._config.feature_range = feature_range

        result = self.normalize(data, method=NormalizeMethod.MINMAX)

        self._config.feature_range = original_range
        return result

    def z_score_normalize(self, data: ArrayType) -> ArrayType:
        """Apply z-score (standard score) normalization.

        Transforms data using::

            X_scaled = (X - mean) / (std + epsilon)

        Args:
            data: Input data.

        Returns:
            Normalized data with approximately zero mean and unit variance.
        """
        return self.normalize(data, method=NormalizeMethod.ZSCORE)

    def robust_normalize(self, data: ArrayType) -> ArrayType:
        """Apply robust normalization using median and IQR.

        Less sensitive to outliers than z-score::

            X_scaled = (X - median) / (IQR + epsilon)

        Args:
            data: Input data.

        Returns:
            Robustly normalized data.
        """
        return self.normalize(data, method=NormalizeMethod.ROBUST)

    # ------------------------------------------------------------------
    # Numpy implementations
    # ------------------------------------------------------------------

    def _normalize_numpy(
        self,
        data: np.ndarray,
        method: NormalizeMethod,
    ) -> tuple[np.ndarray, NormParams]:
        """Normalize a numpy array."""
        axis = self._config.axis
        eps = self._config.epsilon
        params = NormParams(
            method=method,
            feature_range=self._config.feature_range,
            epsilon=eps,
            axis=axis,
            original_shape=data.shape,
        )

        if method == NormalizeMethod.MINMAX:
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            params.min_val = min_val
            params.max_val = max_val

            range_val = max_val - min_val
            range_val = np.where(range_val < eps, 1.0, range_val)

            lo, hi = self._config.feature_range
            result = (data - min_val) / range_val * (hi - lo) + lo

            if self._config.clip:
                result = np.clip(result, lo, hi)

        elif method == NormalizeMethod.ZSCORE:
            mean_val = np.mean(data, axis=axis, keepdims=True)
            std_val = np.std(data, axis=axis, keepdims=True)
            params.mean_val = mean_val
            params.std_val = std_val

            std_safe = np.where(std_val < eps, 1.0, std_val)
            result = (data - mean_val) / std_safe

        elif method == NormalizeMethod.ROBUST:
            q_low_pct, q_high_pct = self._config.quantile_range
            median_val = np.median(data, axis=axis, keepdims=True)
            q_low = np.percentile(data, q_low_pct, axis=axis, keepdims=True)
            q_high = np.percentile(data, q_high_pct, axis=axis, keepdims=True)
            iqr = q_high - q_low
            params.median_val = median_val
            params.q_low = q_low
            params.q_high = q_high
            params.iqr = iqr

            iqr_safe = np.where(iqr < eps, 1.0, iqr)
            result = (data - median_val) / iqr_safe

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result.astype(data.dtype), params

    def _denormalize_numpy(
        self,
        data: np.ndarray,
        params: NormParams,
    ) -> np.ndarray:
        """Denormalize a numpy array."""
        method = params.method

        if method == NormalizeMethod.MINMAX:
            min_val = params.min_val
            max_val = params.max_val
            lo, hi = params.feature_range

            range_val = max_val - min_val
            range_val = np.where(range_val < params.epsilon, 1.0, range_val)

            result = (data - lo) / (hi - lo) * range_val + min_val

        elif method == NormalizeMethod.ZSCORE:
            std_safe = np.where(params.std_val < params.epsilon, 1.0, params.std_val)
            result = data * std_safe + params.mean_val

        elif method == NormalizeMethod.ROBUST:
            iqr_safe = np.where(params.iqr < params.epsilon, 1.0, params.iqr)
            result = data * iqr_safe + params.median_val

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result.astype(data.dtype)

    # ------------------------------------------------------------------
    # PyTorch implementations
    # ------------------------------------------------------------------

    def _normalize_torch(
        self,
        data: Any,
        method: NormalizeMethod,
    ) -> tuple[Any, NormParams]:
        """Normalize a PyTorch tensor."""
        torch = _get_torch()
        axis = self._config.axis
        eps = self._config.epsilon

        params = NormParams(
            method=method,
            feature_range=self._config.feature_range,
            epsilon=eps,
            axis=axis,
            original_shape=tuple(data.shape),
        )

        # For PyTorch, we store params as numpy for portability
        if method == NormalizeMethod.MINMAX:
            dim = axis if axis is not None else 0
            if axis is None:
                # Global normalization
                min_val = data.min().item()
                max_val = data.max().item()
                range_val = max_val - min_val if (max_val - min_val) > eps else 1.0

                lo, hi = self._config.feature_range
                result = (data - min_val) / range_val * (hi - lo) + lo

                if self._config.clip:
                    result = torch.clamp(result, lo, hi)

                params.min_val = np.array(min_val)
                params.max_val = np.array(max_val)
            else:
                min_val = data.min(dim=dim, keepdim=True).values
                max_val = data.max(dim=dim, keepdim=True).values
                range_val = torch.where(
                    (max_val - min_val) > eps, max_val - min_val, torch.tensor(1.0, dtype=data.dtype)
                )

                lo, hi = self._config.feature_range
                result = (data - min_val) / range_val * (hi - lo) + lo

                if self._config.clip:
                    result = torch.clamp(result, lo, hi)

                params.min_val = min_val.detach().cpu().numpy()
                params.max_val = max_val.detach().cpu().numpy()

        elif method == NormalizeMethod.ZSCORE:
            if axis is None:
                mean_val = data.mean().item()
                std_val = data.std().item()
                std_safe = std_val if std_val > eps else 1.0
                result = (data - mean_val) / std_safe

                params.mean_val = np.array(mean_val)
                params.std_val = np.array(std_val)
            else:
                mean_val = data.mean(dim=axis, keepdim=True)
                std_val = data.std(dim=axis, keepdim=True)
                std_safe = torch.where(
                    std_val > eps, std_val, torch.tensor(1.0, dtype=data.dtype)
                )
                result = (data - mean_val) / std_safe

                params.mean_val = mean_val.detach().cpu().numpy()
                params.std_val = std_val.detach().cpu().numpy()

        elif method == NormalizeMethod.ROBUST:
            # PyTorch doesn't have built-in quantile, convert to numpy
            np_data = data.detach().cpu().numpy()
            q_low_pct, q_high_pct = self._config.quantile_range

            if axis is None:
                median_val = np.median(np_data)
                q_low = np.percentile(np_data, q_low_pct)
                q_high = np.percentile(np_data, q_high_pct)
            else:
                median_val = np.median(np_data, axis=axis, keepdims=True)
                q_low = np.percentile(np_data, q_low_pct, axis=axis, keepdims=True)
                q_high = np.percentile(np_data, q_high_pct, axis=axis, keepdims=True)

            iqr = q_high - q_low
            iqr_safe = np.where(iqr < eps, 1.0, iqr)

            result_np = (np_data - median_val) / iqr_safe
            result = torch.from_numpy(result_np.astype(data.dtype)).to(data.device)

            params.median_val = median_val if isinstance(median_val, np.ndarray) else np.array(median_val)
            params.q_low = q_low if isinstance(q_low, np.ndarray) else np.array(q_low)
            params.q_high = q_high if isinstance(q_high, np.ndarray) else np.array(q_high)
            params.iqr = iqr if isinstance(iqr, np.ndarray) else np.array(iqr)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result, params

    def _denormalize_torch(
        self,
        data: Any,
        params: NormParams,
    ) -> Any:
        """Denormalize a PyTorch tensor."""
        torch = _get_torch()
        method = params.method

        if method == NormalizeMethod.MINMAX:
            min_val = torch.tensor(params.min_val, dtype=data.dtype, device=data.device)
            max_val = torch.tensor(params.max_val, dtype=data.dtype, device=data.device)
            lo, hi = params.feature_range

            range_val = max_val - min_val
            range_val = torch.where(
                range_val.abs() < params.epsilon,
                torch.tensor(1.0, dtype=data.dtype, device=data.device),
                range_val,
            )

            result = (data - lo) / (hi - lo) * range_val + min_val

        elif method == NormalizeMethod.ZSCORE:
            mean_val = torch.tensor(params.mean_val, dtype=data.dtype, device=data.device)
            std_val = torch.tensor(params.std_val, dtype=data.dtype, device=data.device)
            std_safe = torch.where(
                std_val.abs() < params.epsilon,
                torch.tensor(1.0, dtype=data.dtype, device=data.device),
                std_val,
            )
            result = data * std_safe + mean_val

        elif method == NormalizeMethod.ROBUST:
            median_val = torch.tensor(params.median_val, dtype=data.dtype, device=data.device)
            iqr = torch.tensor(params.iqr, dtype=data.dtype, device=data.device)
            iqr_safe = torch.where(
                iqr.abs() < params.epsilon,
                torch.tensor(1.0, dtype=data.dtype, device=data.device),
                iqr,
            )
            result = data * iqr_safe + median_val

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_torch_tensor(obj: Any) -> bool:
    """Check if *obj* is a PyTorch tensor without importing torch eagerly."""
    type_name = type(obj).__module__
    return "torch" in type_name
