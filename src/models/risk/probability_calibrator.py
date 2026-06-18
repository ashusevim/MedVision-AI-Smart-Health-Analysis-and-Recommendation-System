"""
ProbabilityCalibrator - Calibrate model output probabilities.

This module provides calibration methods to ensure that a model's predicted
probabilities align with the true frequency of outcomes.  Supported methods
include Platt scaling (logistic regression), isotonic regression, and
temperature scaling.

Accurate probability calibration is critical in medical AI where decisions
depend on reliable uncertainty estimates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """Temperature scaling calibrator.

    Divides logits by a learned scalar temperature *T* before applying
    softmax.  Optimises *T* via negative log-likelihood on a held-out set.

    Args:
        init_temperature: Initial temperature value.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.temperature: float = init_temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Optimise temperature via grid search + refinement.

        Args:
            logits: Raw model logits ``(N, C)``.
            labels: Ground-truth labels ``(N,)``.
        """
        best_temp = 1.0
        best_nll = float("inf")

        # Coarse grid search
        for temp in np.linspace(0.1, 5.0, 50):
            nll = self._nll(logits, labels, temp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        # Fine-grained search around best
        for temp in np.linspace(max(0.01, best_temp - 0.5), best_temp + 0.5, 100):
            nll = self._nll(logits, labels, temp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp
        logger.info("TemperatureScaling fit — T=%.4f, NLL=%.4f", self.temperature, best_nll)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling.

        Args:
            logits: Raw logits ``(N, C)``.

        Returns:
            Calibrated probabilities ``(N, C)``.
        """
        scaled = logits / self.temperature
        return self._softmax(scaled)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _nll(self, logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
        """Compute negative log-likelihood for a given temperature."""
        probs = self._softmax(logits / temperature)
        n = len(labels)
        log_probs = -np.log(probs[np.arange(n), labels] + 1e-12)
        return float(np.mean(log_probs))


class PlattScaling:
    """Platt scaling calibrator using logistic regression.

    Fits ``P(y=1 | z) = 1 / (1 + exp(A * z + B))`` for binary problems or
    one-vs-rest for multi-class.

    Args:
        max_iter: Maximum iterations for logistic regression.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        self.max_iter = max_iter
        self._calibrators: list[Any] = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit one-vs-rest logistic regression calibrators.

        Args:
            logits: Raw logits ``(N, C)``.
            labels: Ground-truth labels ``(N,)``.
        """
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for PlattScaling. "
                "Install it with: pip install scikit-learn"
            ) from exc

        num_classes = logits.shape[1]
        self._calibrators = []

        for c in range(num_classes):
            binary_labels = (labels == c).astype(int)
            cal = LogisticRegression(max_iter=self.max_iter, solver="lbfgs")
            cal.fit(logits[:, c : c + 1], binary_labels)
            self._calibrators.append(cal)

        logger.info("PlattScaling fit — %d calibrators", num_classes)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling.

        Args:
            logits: Raw logits ``(N, C)``.

        Returns:
            Calibrated probabilities ``(N, C)``.
        """
        num_classes = len(self._calibrators)
        probs = np.zeros((logits.shape[0], num_classes))

        for c, cal in enumerate(self._calibrators):
            probs[:, c] = cal.predict_proba(logits[:, c : c + 1])[:, 1]

        # Normalise rows to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        return probs


class IsotonicCalibrator:
    """Isotonic regression calibrator.

    Fits a non-parametric, monotonically increasing step function that maps
    raw scores to calibrated probabilities.  More flexible than Platt scaling
    but requires more data to avoid overfitting.
    """

    def __init__(self) -> None:
        self._calibrators: list[Any] = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit one-vs-rest isotonic regression calibrators.

        Args:
            logits: Raw logits ``(N, C)``.
            labels: Ground-truth labels ``(N,)``.
        """
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for IsotonicCalibrator. "
                "Install it with: pip install scikit-learn"
            ) from exc

        num_classes = logits.shape[1]
        self._calibrators = []

        for c in range(num_classes):
            binary_labels = (labels == c).astype(float)
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(logits[:, c], binary_labels)
            self._calibrators.append(cal)

        logger.info("IsotonicCalibrator fit — %d calibrators", num_classes)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration.

        Args:
            logits: Raw logits ``(N, C)``.

        Returns:
            Calibrated probabilities ``(N, C)``.
        """
        num_classes = len(self._calibrators)
        probs = np.zeros((logits.shape[0], num_classes))

        for c, cal in enumerate(self._calibrators):
            probs[:, c] = cal.transform(logits[:, c])

        # Normalise
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        return probs


class ProbabilityCalibrator:
    """Calibrate model output probabilities using various methods.

    Provides a unified interface for Platt scaling, isotonic regression, and
    temperature scaling.  After fitting on a validation set, the calibrator
    can transform raw logits into well-calibrated probabilities.

    Args:
        method: Calibration method — ``"platt"``, ``"isotonic"``, or
            ``"temperature"``.

    Raises:
        ValueError: If *method* is not supported.
    """

    _SUPPORTED_METHODS = ("platt", "isotonic", "temperature")

    def __init__(self, method: str = "platt") -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported calibration method '{method}'. "
                f"Choose from {list(self._SUPPORTED_METHODS)}"
            )

        self.method = method
        self._fitted = False

        if method == "platt":
            self._calibrator = PlattScaling()
        elif method == "isotonic":
            self._calibrator = IsotonicCalibrator()
        elif method == "temperature":
            self._calibrator = TemperatureScaling()

        logger.info("ProbabilityCalibrator initialised — method=%s", method)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator on validation logits and labels.

        Args:
            logits: Raw model logits ``(N, C)``.
            labels: Ground-truth class indices ``(N,)``.

        Raises:
            ValueError: If *logits* and *labels* have inconsistent lengths.
        """
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits and labels must have the same number of samples "
                f"({logits.shape[0]} != {labels.shape[0]})"
            )

        self._calibrator.fit(logits, labels)
        self._fitted = True
        logger.info("ProbabilityCalibrator fitted on %d samples", logits.shape[0])

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate raw logits into well-calibrated probabilities.

        Args:
            logits: Raw model logits ``(N, C)``.

        Returns:
            Calibrated probabilities ``(N, C)``.

        Raises:
            RuntimeError: If the calibrator has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Calibrator has not been fitted. Call fit() first."
            )
        return self._calibrator.calibrate(logits)

    def evaluate_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Evaluate calibration quality using standard metrics.

        Computes:

        * **Expected Calibration Error (ECE)** — weighted average of
          bin-wise absolute difference between confidence and accuracy.
        * **Maximum Calibration Error (MCE)** — worst-case bin-wise
          calibration error.
        * **Brier Score** — mean squared error between probabilities and
          one-hot labels.

        Args:
            probs: Calibrated probabilities ``(N, C)``.
            labels: Ground-truth labels ``(N,)``.
            n_bins: Number of bins for ECE / MCE computation.

        Returns:
            Dictionary with ``ece``, ``mce``, and ``brier_score``.
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        n = len(labels)

        # ECE and MCE
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > low) & (confidences <= high)
            bin_size = mask.sum()
            if bin_size == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            error = abs(bin_conf - bin_acc)
            ece += (bin_size / n) * error
            mce = max(mce, error)

        # Brier score
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), labels] = 1.0
        brier_score = float(np.mean((probs - one_hot) ** 2))

        result = {
            "ece": round(ece, 6),
            "mce": round(mce, 6),
            "brier_score": round(brier_score, 6),
            "n_samples": n,
            "n_bins": n_bins,
        }

        logger.info(
            "Calibration evaluation — ECE=%.4f, MCE=%.4f, Brier=%.4f",
            ece, mce, brier_score,
        )

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    def __repr__(self) -> str:
        return f"ProbabilityCalibrator(method='{self.method}', fitted={self._fitted})"
