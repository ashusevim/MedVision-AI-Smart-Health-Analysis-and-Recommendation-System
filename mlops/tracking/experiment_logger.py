"""
ExperimentLogger - High-level experiment tracking for MedVision-AI.

Provides a clean, framework-agnostic interface for logging experiments,
including parameters, metrics, models, and artifacts with structured
metadata and automatic timing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record of a single experiment run.

    Attributes:
        run_name: Human-readable name for the run.
        run_id: Unique identifier for the run.
        experiment_name: Parent experiment name.
        start_time: ISO-format timestamp when the run started.
        end_time: ISO-format timestamp when the run ended.
        duration_seconds: Wall-clock duration of the run.
        params: Logged hyperparameters.
        metrics: Logged metric values (latest per key).
        metric_history: Full time-series of metric values.
        artifacts: List of logged artifact paths.
        status: Run status ('running', 'completed', 'failed').
    """

    run_name: str = ""
    run_id: str = ""
    experiment_name: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"


class ExperimentLogger:
    """High-level experiment tracking logger.

    ExperimentLogger provides a structured, framework-agnostic interface
    for tracking machine learning experiments. It supports nested parameter
    and metric logging, automatic timing, model checkpoint recording, and
    artifact management. Data can optionally be persisted to MLflow.

    Args:
        experiment_name: Name of the experiment (used for organisation and storage).
        storage_dir: Local directory for persisting run data as JSON.
        use_mlflow: Whether to also log to MLflow (requires mlflow installation).

    Example::

        exp = ExperimentLogger("medvision-segmentation")
        exp.start_run("unet-baseline")
        exp.log_params({"lr": 1e-3, "batch_size": 16, "backbone": "resnet50"})
        exp.log_metrics({"dice": 0.87, "iou": 0.78}, step=10)
        exp.log_model(model)
        exp.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        storage_dir: str = "./experiment_logs",
        use_mlflow: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.storage_dir = Path(storage_dir) / experiment_name
        self.use_mlflow = use_mlflow

        self._active_run: Optional[RunRecord] = None
        self._run_history: List[RunRecord] = []
        self._start_timestamp: float = 0.0
        self._step_counter: Dict[str, int] = {}

        # Optional MLflow backend
        self._mlflow_config: Optional[Any] = None
        if self.use_mlflow:
            try:
                from mlops.tracking.mlflow_config import MLflowConfig

                self._mlflow_config = MLflowConfig(
                    experiment_name=self.experiment_name,
                )
                self._mlflow_config.setup()
            except ImportError:
                logger.warning(
                    "MLflow not available; falling back to local logging only."
                )
                self.use_mlflow = False

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ExperimentLogger initialised | experiment='%s' | mlflow=%s | storage=%s",
            self.experiment_name,
            self.use_mlflow,
            self.storage_dir,
        )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new experiment run.

        Only one run can be active at a time. If a run is already active,
        it will be ended with status 'interrupted' before starting the new one.

        Args:
            run_name: Descriptive name for this run.
            tags: Optional tags to associate with the run.

        Returns:
            A unique run ID string.
        """
        if self._active_run is not None:
            logger.warning(
                "Run '%s' is still active; ending it before starting new run.",
                self._active_run.run_name,
            )
            self.end_run(status="interrupted")

        run_id = f"{self.experiment_name}_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._start_timestamp = time.time()

        self._active_run = RunRecord(
            run_name=run_name,
            run_id=run_id,
            experiment_name=self.experiment_name,
            start_time=datetime.now().isoformat(),
            status="running",
        )

        if tags:
            self._active_run.params.update({"tag_" + k: v for k, v in tags.items()})

        # Start MLflow run if enabled
        if self.use_mlflow and self._mlflow_config is not None:
            try:
                self._mlflow_config.start_run(run_name=run_name, tags=tags)
            except Exception as exc:
                logger.warning("Failed to start MLflow run: %s", exc)

        logger.info("Run started: %s (id=%s)", run_name, run_id)
        return run_id

    def end_run(self, status: str = "completed") -> Optional[RunRecord]:
        """End the active experiment run and persist the record.

        Args:
            status: Final status for the run ('completed', 'failed', 'interrupted').

        Returns:
            The completed RunRecord, or None if no run was active.
        """
        if self._active_run is None:
            logger.warning("No active run to end.")
            return None

        self._active_run.end_time = datetime.now().isoformat()
        self._active_run.duration_seconds = time.time() - self._start_timestamp
        self._active_run.status = status

        # Persist to disk
        self._persist_run(self._active_run)
        self._run_history.append(self._active_run)

        # End MLflow run if enabled
        if self.use_mlflow and self._mlflow_config is not None:
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            try:
                self._mlflow_config.end_run(status=mlflow_status)
            except Exception as exc:
                logger.warning("Failed to end MLflow run: %s", exc)

        finished = self._active_run
        logger.info(
            "Run ended: %s | status=%s | duration=%.1fs",
            finished.run_name,
            status,
            finished.duration_seconds,
        )
        self._active_run = None
        self._step_counter = {}

        return finished

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for the active run.

        Parameters are recorded as key-value pairs. Nested dictionaries are
        flattened with dot-notation keys (e.g. {'model': {'lr': 0.01}}
        becomes {'model.lr': 0.01}).

        Args:
            params: Dictionary of parameter names to values.

        Raises:
            RuntimeError: If no run is currently active.
        """
        self._require_active_run()

        flat_params = self._flatten_dict(params)
        self._active_run.params.update(flat_params)

        # Log to MLflow
        if self.use_mlflow and self._mlflow_config is not None:
            try:
                self._mlflow_config.log_params(flat_params)
            except Exception as exc:
                logger.warning("Failed to log params to MLflow: %s", exc)

        logger.debug("Logged %d params: %s", len(flat_params), list(flat_params.keys()))

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metric values for the active run.

        Metrics can be logged at specific steps for time-series tracking.
        If step is not provided, an auto-incrementing counter per metric
        name is used.

        Args:
            metrics: Dictionary of metric names to float values.
            step: Optional step number for the metrics.

        Raises:
            RuntimeError: If no run is currently active.
        """
        self._require_active_run()

        for key, value in metrics.items():
            # Determine step
            if step is not None:
                actual_step = step
            else:
                self._step_counter[key] = self._step_counter.get(key, 0) + 1
                actual_step = self._step_counter[key]

            # Update latest metrics
            self._active_run.metrics[key] = value

            # Append to history
            if key not in self._active_run.metric_history:
                self._active_run.metric_history[key] = []
            self._active_run.metric_history[key].append(
                {"step": actual_step, "value": value, "timestamp": time.time()}
            )

        # Log to MLflow
        if self.use_mlflow and self._mlflow_config is not None:
            try:
                self._mlflow_config.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.warning("Failed to log metrics to MLflow: %s", exc)

        logger.debug(
            "Logged %d metrics at step %s: %s",
            len(metrics),
            step,
            list(metrics.keys()),
        )

    def log_model(self, model: Any, name: str = "model") -> None:
        """Record a model checkpoint reference for the active run.

        The model is saved to the storage directory with a timestamped
        filename. If MLflow is enabled, the model is also logged there.

        Args:
            model: The model object (must support .state_dict() or be serializable).
            name: Base name for the saved model artifact.

        Raises:
            RuntimeError: If no run is currently active.
        """
        self._require_active_run()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.storage_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}_{timestamp}.pt"

        try:
            import torch

            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        except ImportError:
            # Fallback: try pickle
            import pickle

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        self._active_run.artifacts.append(str(model_path))

        # Log to MLflow
        if self.use_mlflow and self._mlflow_config is not None:
            try:
                self._mlflow_config.log_model(model, name=name, flavor="pytorch")
            except Exception as exc:
                logger.warning("Failed to log model to MLflow: %s", exc)

        logger.info("Model logged: %s", model_path)

    def log_artifact(self, path: str, description: str = "") -> None:
        """Log a file or directory as an artifact of the active run.

        The artifact is copied (or referenced) in the run's storage directory.

        Args:
            path: Local path to the artifact file or directory.
            description: Optional description of the artifact.

        Raises:
            RuntimeError: If no run is currently active.
            FileNotFoundError: If the specified path does not exist.
        """
        self._require_active_run()

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        # Copy artifact to storage directory
        artifact_dir = self.storage_dir / "artifacts" / self._active_run.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        dest = artifact_dir / src.name

        if src.is_file():
            import shutil

            shutil.copy2(src, dest)
        elif src.is_dir():
            import shutil

            shutil.copytree(src, dest, dirs_exist_ok=True)

        # Record artifact with optional description
        record = str(dest)
        if description:
            record = f"{record} ({description})"
        self._active_run.artifacts.append(record)

        # Log to MLflow
        if self.use_mlflow and self._mlflow_config is not None:
            try:
                self._mlflow_config.log_artifact(path)
            except Exception as exc:
                logger.warning("Failed to log artifact to MLflow: %s", exc)

        logger.info("Artifact logged: %s -> %s", path, dest)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_run_history(self) -> List[Dict[str, Any]]:
        """Return a summary of all completed runs.

        Returns:
            List of dictionaries with run metadata and best metrics.
        """
        return [
            {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "params": r.params,
                "metrics": r.metrics,
                "num_artifacts": len(r.artifacts),
            }
            for r in self._run_history
        ]

    def get_best_run(self, metric: str, mode: str = "min") -> Optional[RunRecord]:
        """Find the run with the best value for a given metric.

        Args:
            metric: The metric name to compare.
            mode: 'min' to find the lowest value, 'max' for the highest.

        Returns:
            The RunRecord with the best metric value, or None.
        """
        if not self._run_history:
            return None

        best_run: Optional[RunRecord] = None
        best_value = float("inf") if mode == "min" else float("-inf")

        for run in self._run_history:
            if metric in run.metrics:
                value = run.metrics[metric]
                if (mode == "min" and value < best_value) or (
                    mode == "max" and value > best_value
                ):
                    best_value = value
                    best_run = run

        return best_run

    def compare_runs(
        self,
        metric_keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare metrics across all completed runs.

        Args:
            metric_keys: Specific metrics to compare. If None, all metrics are included.

        Returns:
            Dictionary mapping run names to their metric values.
        """
        comparison: Dict[str, Dict[str, float]] = {}
        for run in self._run_history:
            metrics = run.metrics
            if metric_keys:
                metrics = {k: v for k, v in metrics.items() if k in metric_keys}
            comparison[run.run_name] = metrics
        return comparison

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_run(self, run: RunRecord) -> None:
        """Save a run record to disk as JSON.

        Args:
            run: The RunRecord to persist.
        """
        run_file = self.storage_dir / f"{run.run_id}.json"
        try:
            data = {
                "run_name": run.run_name,
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration_seconds": run.duration_seconds,
                "params": run.params,
                "metrics": run.metrics,
                "metric_history": run.metric_history,
                "artifacts": run.artifacts,
                "status": run.status,
            }
            run_file.write_text(json.dumps(data, indent=2, default=str))
            logger.debug("Run record persisted: %s", run_file)
        except Exception as exc:
            logger.error("Failed to persist run record: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_active_run(self) -> None:
        """Assert that a run is currently active.

        Raises:
            RuntimeError: If no run is active.
        """
        if self._active_run is None:
            raise RuntimeError(
                "No active run. Call start_run() before logging."
            )

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary using dot-notation keys.

        Args:
            d: The dictionary to flatten.
            parent_key: Prefix for keys (used in recursion).
            sep: Separator between nested keys.

        Returns:
            A flat dictionary with dot-notation keys.
        """
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    ExperimentLogger._flatten_dict(v, new_key, sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
