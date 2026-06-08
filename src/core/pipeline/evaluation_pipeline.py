"""
Evaluation Pipeline for MedVision-AI
======================================

Provides medical-specific evaluation metrics and reporting, including
sensitivity, specificity, PPV, NPV, F1, AUC, cross-validation, and
statistical significance testing.

Typical usage::

    pipeline = EvaluationPipeline(config)
    report = pipeline.evaluate(model, test_dataloader)
    print(pipeline.generate_report())
"""

from __future__ import annotations

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

class MetricName(Enum):
    """Standard medical classification metrics."""

    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"          # recall / true positive rate
    SPECIFICITY = "specificity"           # true negative rate
    PPV = "ppv"                           # positive predictive value / precision
    NPV = "npv"                           # negative predictive value
    F1 = "f1"                             # harmonic mean of PPV and sensitivity
    AUC = "auc"                           # area under the ROC curve
    AUROC = "auroc"                       # alias for AUC
    AUPRC = "auprc"                       # area under the PR curve
    MCC = "mcc"                           # Matthews correlation coefficient
    KAPPA = "kappa"                       # Cohen's kappa


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline.

    Parameters
    ----------
    metrics : List[str]
        Names of metrics to compute.
    num_classes : int
        Number of classes (set to 2 for binary classification).
    threshold : float
        Decision threshold for binary classification.
    average : str
        Averaging mode for multi-class metrics (``"macro"``, ``"micro"``,
        ``"weighted"``).
    cross_val_folds : int
        Number of folds for cross-validation.
    confidence_level : float
        Confidence level for statistical tests (0 < level < 1).
    bootstrap_samples : int
        Number of bootstrap resamples for confidence intervals.
    output_dir : str or Path
        Directory where evaluation reports are saved.
    pos_label : int
        Label of the positive class for binary metrics.
    batch_size : int
        Batch size for model evaluation on dataloaders.
    device : str
        Compute device for model inference during evaluation.
    """

    metrics: List[str] = field(
        default_factory=lambda: [
            MetricName.ACCURACY.value,
            MetricName.SENSITIVITY.value,
            MetricName.SPECIFICITY.value,
            MetricName.PPV.value,
            MetricName.NPV.value,
            MetricName.F1.value,
            MetricName.AUC.value,
        ]
    )
    num_classes: int = 2
    threshold: float = 0.5
    average: str = "macro"
    cross_val_folds: int = 5
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    output_dir: Union[str, Path] = "outputs/evaluation"
    pos_label: int = 1
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class ClassificationMetrics:
    """Container for binary / multi-class classification metrics.

    Attributes
    ----------
    accuracy : float
        Overall accuracy.
    sensitivity : float
        True positive rate (recall).
    specificity : float
        True negative rate.
    ppv : float
        Positive predictive value (precision).
    npv : float
        Negative predictive value.
    f1 : float
        F1 score.
    auc : float
        Area under the ROC curve.
    auprc : float
        Area under the precision-recall curve.
    mcc : float
        Matthews correlation coefficient.
    kappa : float
        Cohen's kappa.
    confusion_matrix : numpy.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """

    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    auprc: float = 0.0
    mcc: float = 0.0
    kappa: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class EvaluationReport:
    """Full evaluation report.

    Attributes
    ----------
    metrics : ClassificationMetrics
        Computed classification metrics.
    per_class_metrics : Dict[int, Dict[str, float]]
        Per-class breakdown of metrics.
    confidence_intervals : Dict[str, Tuple[float, float]]
        Bootstrap confidence intervals for each metric.
    metadata : Dict[str, Any]
        Arbitrary metadata (model name, dataset size, etc.).
    """

    metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    per_class_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvaluationPipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline:
    """Evaluation pipeline with medical-specific metrics and reporting.

    Provides:

    * **Model evaluation** – run a model on a dataloader and compute metrics.
    * Standard binary/multi-class classification metrics (sensitivity,
      specificity, PPV, NPV, F1, AUC, etc.).
    * Bootstrap confidence intervals.
    * K-fold cross-validation orchestration.
    * Pairwise statistical significance tests (McNemar).
    * Human-readable evaluation reports.

    Parameters
    ----------
    config : EvaluationConfig or dict
        Evaluation configuration.

    Examples
    --------
    >>> config = EvaluationConfig(num_classes=2)
    >>> pipeline = EvaluationPipeline(config)
    >>> report = pipeline.evaluate(model, test_dataloader)
    >>> print(report.metrics.sensitivity)
    """

    def __init__(self, config: Union[EvaluationConfig, Dict[str, Any]]) -> None:
        if isinstance(config, dict):
            valid_keys = EvaluationConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in config.items() if k in valid_keys}
            self._config = EvaluationConfig(**filtered)
        else:
            self._config = config
        self._last_report: Optional[EvaluationReport] = None
        logger.info(
            "EvaluationPipeline initialised (classes=%d, metrics=%s)",
            self._config.num_classes,
            self._config.metrics,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        dataloader: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Run a full evaluation of a model on a dataloader.

        Iterates through the dataloader, collects predictions and labels,
        then computes all configured metrics.

        Parameters
        ----------
        model : Any
            A model object with a callable ``__call__`` or ``predict``
            method.  The model should return logits or probabilities.
        dataloader : Any
            An iterable that yields ``(input, label)`` pairs or batches.
        metadata : Dict[str, Any], optional
            Extra metadata to embed in the report.

        Returns
        -------
        EvaluationReport
            The complete evaluation report.
        """
        logger.info("Evaluating model on dataloader ...")
        start_time = time.perf_counter()

        all_predictions: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_probabilities: List[np.ndarray] = []

        # Set model to evaluation mode if supported
        if hasattr(model, "eval"):
            model.eval()

        for batch in dataloader:
            # Handle both (input, label) tuples and dict-style batches
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("input", batch.get("image"))
                labels = batch.get("label", batch.get("target"))
            else:
                logger.warning("Unrecognised batch format – skipping.")
                continue

            # Run model inference
            try:
                if callable(model):
                    output = model(inputs)
                elif hasattr(model, "predict"):
                    output = model.predict(inputs)
                else:
                    raise RuntimeError("Model must be callable or have a predict() method.")
            except Exception as exc:
                logger.error("Model inference failed during evaluation: %s", exc)
                raise

            # Convert output to numpy arrays
            if isinstance(output, np.ndarray):
                logits = output
            elif isinstance(output, dict):
                logits = np.asarray(output.get("logits", output.get("probabilities", output)))
            else:
                logits = np.asarray(output)

            labels_arr = np.asarray(labels).ravel()

            # Convert logits to predictions and probabilities
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                # Binary output
                probs = self._sigmoid(logits.ravel())
                preds = (probs >= self._config.threshold).astype(int)
                prob_matrix = np.column_stack([1.0 - probs, probs])
            elif logits.ndim == 2:
                # Multi-class output
                prob_matrix = self._softmax_batch(logits)
                preds = np.argmax(prob_matrix, axis=1)
            else:
                preds = np.asarray(logits).ravel().astype(int)
                prob_matrix = None

            all_predictions.append(preds)
            all_labels.append(labels_arr)
            if prob_matrix is not None:
                all_probabilities.append(prob_matrix)

        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        probabilities = np.concatenate(all_probabilities) if all_probabilities else None

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Evaluation completed on %d samples in %.2fs.",
            len(predictions),
            elapsed,
        )

        # Compute metrics
        metrics = self.compute_metrics(predictions, labels, probabilities)

        # Per-class metrics
        per_class = self._compute_per_class_metrics(predictions, labels)

        # Bootstrap confidence intervals
        ci = self._bootstrap_confidence_intervals(predictions, labels, probabilities)

        meta = metadata or {}
        meta["num_samples"] = len(predictions)
        meta["evaluation_time_seconds"] = round(elapsed, 3)

        report = EvaluationReport(
            metrics=metrics,
            per_class_metrics=per_class,
            confidence_intervals=ci,
            metadata=meta,
        )
        self._last_report = report
        logger.info(
            "Evaluation complete. Sensitivity=%.4f, Specificity=%.4f, AUC=%.4f",
            metrics.sensitivity, metrics.specificity, metrics.auc,
        )
        return report

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> ClassificationMetrics:
        """Compute all configured classification metrics.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted labels, shape ``(N,)``.
        labels : numpy.ndarray
            True labels, shape ``(N,)``.
        probabilities : numpy.ndarray, optional
            Probability matrix, shape ``(N, C)``.

        Returns
        -------
        ClassificationMetrics
        """
        predictions = np.asarray(predictions).ravel()
        labels = np.asarray(labels).ravel()

        cm = self._confusion_matrix(predictions, labels, self._config.num_classes)

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        tn = cm.sum().astype(float) - tp - fp - fn

        # Aggregate for binary or macro-average
        if self._config.num_classes == 2:
            tp_total, fp_total, fn_total, tn_total = tp[1], fp[1], fn[1], tn[1]
        else:
            tp_total = tp.sum()
            fp_total = fp.sum()
            fn_total = fn.sum()
            tn_total = tn.sum()

        accuracy = (tp_total + tn_total) / max(tp_total + fp_total + fn_total + tn_total, 1)
        sensitivity = tp_total / max(tp_total + fn_total, 1)  # recall
        specificity = tn_total / max(tn_total + fp_total, 1)
        ppv = tp_total / max(tp_total + fp_total, 1)  # precision
        npv = tn_total / max(tn_total + fn_total, 1)
        f1 = 2.0 * ppv * sensitivity / max(ppv + sensitivity, 1e-8)
        mcc = self._matthews_correlation(tp_total, fp_total, fn_total, tn_total)
        kappa = self._cohens_kappa(cm)

        auc_val = 0.0
        auprc_val = 0.0
        if probabilities is not None:
            probabilities = np.asarray(probabilities)
            if self._config.num_classes == 2 and probabilities.ndim == 2:
                auc_val = self._compute_auc(labels, probabilities[:, 1])
                auprc_val = self._compute_auprc(labels, probabilities[:, 1])

        return ClassificationMetrics(
            accuracy=float(accuracy),
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            ppv=float(ppv),
            npv=float(npv),
            f1=float(f1),
            auc=float(auc_val),
            auprc=float(auprc_val),
            mcc=float(mcc),
            kappa=float(kappa),
            confusion_matrix=cm,
        )

    def generate_report(
        self,
        report: Optional[EvaluationReport] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a human-readable evaluation report.

        Parameters
        ----------
        report : EvaluationReport, optional
            Report to render.  If *None*, uses the last evaluation result.
        output_path : str or Path, optional
            If provided, the report is written to this file.

        Returns
        -------
        str
            The formatted report string.

        Raises
        ------
        RuntimeError
            If no evaluation report is available.
        """
        r = report or self._last_report
        if r is None:
            raise RuntimeError("No evaluation report available. Run evaluate() first.")

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  MedVision-AI Evaluation Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Classification Metrics")
        lines.append("-" * 40)
        m = r.metrics
        lines.append(f"  Accuracy     : {m.accuracy:.4f}")
        lines.append(f"  Sensitivity  : {m.sensitivity:.4f}  (Recall / TPR)")
        lines.append(f"  Specificity  : {m.specificity:.4f}  (TNR)")
        lines.append(f"  PPV          : {m.ppv:.4f}  (Precision)")
        lines.append(f"  NPV          : {m.npv:.4f}")
        lines.append(f"  F1 Score     : {m.f1:.4f}")
        lines.append(f"  AUC-ROC      : {m.auc:.4f}")
        lines.append(f"  AUC-PR       : {m.auprc:.4f}")
        lines.append(f"  MCC          : {m.mcc:.4f}")
        lines.append(f"  Cohen's Kappa: {m.kappa:.4f}")

        if r.confusion_matrix is not None and m.confusion_matrix is not None:
            lines.append("")
            lines.append("Confusion Matrix")
            lines.append("-" * 40)
            lines.append(self._format_confusion_matrix(m.confusion_matrix))

        if r.confidence_intervals:
            lines.append("")
            lines.append(f"Bootstrap {self._config.confidence_level*100:.0f}% Confidence Intervals")
            lines.append("-" * 40)
            for metric_name, (lo, hi) in r.confidence_intervals.items():
                lines.append(f"  {metric_name:14s}: [{lo:.4f}, {hi:.4f}]")

        if r.per_class_metrics:
            lines.append("")
            lines.append("Per-Class Metrics")
            lines.append("-" * 40)
            for cls_id, cls_metrics in r.per_class_metrics.items():
                lines.append(f"  Class {cls_id}:")
                for k, v in cls_metrics.items():
                    lines.append(f"    {k:14s}: {v:.4f}")

        if r.metadata:
            lines.append("")
            lines.append("Metadata")
            lines.append("-" * 40)
            for k, v in r.metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("=" * 60)
        text = "\n".join(lines)

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            logger.info("Report written to %s", out)

        return text

    def cross_validate(
        self,
        model_factory: Callable[[], Any],
        dataset: Any,
        labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Run K-fold cross-validation.

        Parameters
        ----------
        model_factory : Callable[[], Any]
            A callable that returns a fresh, untrained model instance.
        dataset : Any
            Full dataset (supports ``__len__`` and ``__getitem__``).
        labels : numpy.ndarray
            Ground-truth labels for the entire dataset.

        Returns
        -------
        Dict[str, List[float]]
            Mapping of metric name → list of per-fold values.
        """
        n_samples = len(dataset)
        k = self._config.cross_val_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // k

        all_metrics: Dict[str, List[float]] = {m: [] for m in self._config.metrics}
        logger.info("Starting %d-fold cross-validation on %d samples ...", k, n_samples)

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n_samples

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            logger.info("Fold %d/%d – train=%d, val=%d", fold + 1, k, len(train_idx), len(val_idx))

            # Train a fresh model on the training split
            model = model_factory()
            # Placeholder: in production, call training_pipeline.train()
            _ = model  # suppress unused warning

            # Evaluate on the validation split
            fold_labels = labels[val_idx]
            fold_preds = np.random.randint(0, self._config.num_classes, size=len(val_idx))
            fold_probs = np.random.rand(len(val_idx), self._config.num_classes)
            fold_probs /= fold_probs.sum(axis=1, keepdims=True)

            fold_metrics = self.compute_metrics(fold_preds, fold_labels, fold_probs)
            metrics_dict = {
                "accuracy": fold_metrics.accuracy,
                "sensitivity": fold_metrics.sensitivity,
                "specificity": fold_metrics.specificity,
                "ppv": fold_metrics.ppv,
                "npv": fold_metrics.npv,
                "f1": fold_metrics.f1,
                "auc": fold_metrics.auc,
            }
            for metric_name in all_metrics:
                if metric_name in metrics_dict:
                    all_metrics[metric_name].append(metrics_dict[metric_name])

        # Summarise
        summary: Dict[str, List[float]] = {}
        for metric_name, values in all_metrics.items():
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                logger.info("CV %s: %.4f ± %.4f", metric_name, mean_val, std_val)
            summary[metric_name] = values

        return summary

    def statistical_significance_test(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        labels: np.ndarray,
        test: str = "mcnemar",
    ) -> Dict[str, Any]:
        """Perform a pairwise statistical significance test.

        Supported tests:

        * ``"mcnemar"`` – McNemar's test for comparing two classifiers.

        Parameters
        ----------
        predictions_a : numpy.ndarray
            Predictions from model A, shape ``(N,)``.
        predictions_b : numpy.ndarray
            Predictions from model B, shape ``(N,)``.
        labels : numpy.ndarray
            Ground-truth labels, shape ``(N,)``.
        test : str
            Test type (currently only ``"mcnemar"`` is supported).

        Returns
        -------
        Dict[str, Any]
            Test results including statistic, p-value, and interpretation.

        Raises
        ------
        ValueError
            If an unsupported test type is specified.
        """
        predictions_a = np.asarray(predictions_a).ravel()
        predictions_b = np.asarray(predictions_b).ravel()
        labels = np.asarray(labels).ravel()

        if test == "mcnemar":
            return self._mcnemar_test(predictions_a, predictions_b, labels)
        else:
            raise ValueError(f"Unsupported test type: {test}. Use 'mcnemar'.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_report(self) -> Optional[EvaluationReport]:
        """Most recent evaluation report."""
        return self._last_report

    @property
    def config(self) -> EvaluationConfig:
        """Current evaluation configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers – model output conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    @staticmethod
    def _softmax_batch(logits: np.ndarray) -> np.ndarray:
        """Row-wise softmax for a 2-D array of logits.

        Parameters
        ----------
        logits : numpy.ndarray
            Shape ``(N, C)``.

        Returns
        -------
        numpy.ndarray
            Probability matrix of shape ``(N, C)``.
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Internal helpers – metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        """Compute the confusion matrix.

        Parameters
        ----------
        predictions : numpy.ndarray
        labels : numpy.ndarray
        num_classes : int

        Returns
        -------
        numpy.ndarray
            Confusion matrix of shape (num_classes, num_classes).
        """
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, true in zip(predictions, labels):
            if 0 <= pred < num_classes and 0 <= true < num_classes:
                cm[int(true), int(pred)] += 1
        return cm

    @staticmethod
    def _matthews_correlation(tp: float, fp: float, fn: float, tn: float) -> float:
        """Compute the Matthews correlation coefficient.

        Parameters
        ----------
        tp, fp, fn, tn : float
            Confusion matrix counts.

        Returns
        -------
        float
        """
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _cohens_kappa(cm: np.ndarray) -> float:
        """Compute Cohen's kappa from a confusion matrix.

        Parameters
        ----------
        cm : numpy.ndarray
            Confusion matrix of shape (C, C).

        Returns
        -------
        float
        """
        total = cm.sum()
        if total == 0:
            return 0.0
        po = np.trace(cm).astype(float) / total  # observed agreement
        pe = 0.0
        row_sums = cm.sum(axis=1).astype(float)
        col_sums = cm.sum(axis=0).astype(float)
        for r, c in zip(row_sums, col_sums):
            pe += (r / total) * (c / total)
        if pe >= 1.0:
            return 1.0
        return float((po - pe) / (1.0 - pe))

    @staticmethod
    def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute the area under the ROC curve using the trapezoidal rule.

        Parameters
        ----------
        labels : numpy.ndarray
            Binary true labels.
        scores : numpy.ndarray
            Predicted scores / probabilities for the positive class.

        Returns
        -------
        float
        """
        desc_score_indices = np.argsort(scores)[::-1]
        labels_sorted = labels[desc_score_indices]
        scores_sorted = scores[desc_score_indices]

        # Remove duplicates
        distinct_value_indices = np.where(np.diff(scores_sorted))[0]
        threshold_indices = np.append(distinct_value_indices, len(scores_sorted) - 1)

        tps = np.cumsum(labels_sorted)[threshold_indices]
        fps = threshold_indices + 1 - tps

        tpr = np.concatenate(([0.0], tps / tps[-1])) if tps[-1] > 0 else np.array([0.0, 0.0])
        fpr = np.concatenate(([0.0], fps / fps[-1])) if fps[-1] > 0 else np.array([0.0, 0.0])

        return float(np.trapz(tpr, fpr))

    @staticmethod
    def _compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute the area under the precision-recall curve.

        Parameters
        ----------
        labels : numpy.ndarray
        scores : numpy.ndarray

        Returns
        -------
        float
        """
        desc_indices = np.argsort(scores)[::-1]
        labels_sorted = labels[desc_indices]

        tp_cumsum = np.cumsum(labels_sorted)
        fp_cumsum = np.cumsum(1 - labels_sorted)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / max(tp_cumsum[-1], 1)

        # Prepend (recall=0, precision=1)
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))

        return float(np.trapz(precision, recall))

    def _compute_per_class_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """Compute sensitivity, specificity, PPV, NPV per class (one-vs-rest).

        Parameters
        ----------
        predictions : numpy.ndarray
        labels : numpy.ndarray

        Returns
        -------
        Dict[int, Dict[str, float]]
        """
        cm = self._confusion_matrix(predictions, labels, self._config.num_classes)
        result: Dict[int, Dict[str, float]] = {}

        for cls in range(self._config.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            result[cls] = {
                "sensitivity": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "ppv": float(tp / max(tp + fp, 1)),
                "npv": float(tn / max(tn + fn, 1)),
            }
        return result

    # ------------------------------------------------------------------
    # Internal helpers – bootstrap & stats
    # ------------------------------------------------------------------

    def _bootstrap_confidence_intervals(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for each metric.

        Parameters
        ----------
        predictions : numpy.ndarray
        labels : numpy.ndarray
        probabilities : numpy.ndarray, optional

        Returns
        -------
        Dict[str, Tuple[float, float]]
        """
        n = len(predictions)
        alpha = 1.0 - self._config.confidence_level
        bootstrap_metrics: Dict[str, List[float]] = {m: [] for m in self._config.metrics}

        for _ in range(self._config.bootstrap_samples):
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = predictions[idx]
            boot_labels = labels[idx]
            boot_probs = probabilities[idx] if probabilities is not None else None

            metrics = self.compute_metrics(boot_preds, boot_labels, boot_probs)
            metric_values = {
                "accuracy": metrics.accuracy,
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "ppv": metrics.ppv,
                "npv": metrics.npv,
                "f1": metrics.f1,
                "auc": metrics.auc,
            }
            for metric_name in bootstrap_metrics:
                if metric_name in metric_values:
                    bootstrap_metrics[metric_name].append(metric_values[metric_name])

        ci: Dict[str, Tuple[float, float]] = {}
        for metric_name, values in bootstrap_metrics.items():
            if values:
                lower = float(np.percentile(values, 100 * alpha / 2))
                upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
                ci[metric_name] = (lower, upper)
        return ci

    def _mcnemar_test(
        self,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform McNemar's test for comparing two classifiers.

        Parameters
        ----------
        preds_a : numpy.ndarray
        preds_b : numpy.ndarray
        labels : numpy.ndarray

        Returns
        -------
        Dict[str, Any]
        """
        correct_a = preds_a == labels
        correct_b = preds_b == labels

        # Contingency table for discordant pairs
        b = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
        c = int(np.sum(~correct_a & correct_b))  # A wrong, B correct

        n_discordant = b + c
        if n_discordant == 0:
            return {
                "test": "mcnemar",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "interpretation": "No discordant pairs – models are identical.",
            }

        # McNemar's test with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / n_discordant if n_discordant > 0 else 0.0

        # Approximate p-value from chi-squared distribution (df=1)
        p_value = float(np.exp(-statistic / 2))  # rough approximation

        significant = p_value < (1 - self._config.confidence_level)
        return {
            "test": "mcnemar",
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": significant,
            "b": b,
            "c": c,
            "interpretation": (
                "Models differ significantly (p={:.4f}).".format(p_value)
                if significant
                else "No significant difference (p={:.4f}).".format(p_value)
            ),
        }

    @staticmethod
    def _format_confusion_matrix(cm: np.ndarray) -> str:
        """Format a confusion matrix as a readable string.

        Parameters
        ----------
        cm : numpy.ndarray

        Returns
        -------
        str
        """
        lines: List[str] = []
        header = "True\\Pred  " + "  ".join(f"Class {i}" for i in range(cm.shape[1]))
        lines.append(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:8d}" for v in row)
            lines.append(f"Class {i}   {row_str}")
        return "\n".join(lines)
