"""
MLflowConfig - MLflow integration for MedVision-AI experiment tracking.

Provides a configuration-driven interface to MLflow for logging parameters,
metrics, models, and artifacts during training and evaluation runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MLflowConfig:
    """MLflow configuration and interaction manager.

    Wraps the MLflow Python client to provide a simplified, typed interface
    for experiment management, run tracking, parameter/metric logging, model
    registration, and artifact storage.

    Args:
        tracking_uri: URI of the MLflow tracking server
            (e.g. 'http://localhost:5000' or 'sqlite:///mlflow.db').
        experiment_name: Name of the experiment to create or use.

    Example::

        mlflow_cfg = MLflowConfig(
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="medvision-classification",
        )
        mlflow_cfg.setup()
        mlflow_cfg.log_params({"lr": 0.001, "epochs": 30})
        mlflow_cfg.log_metrics({"accuracy": 0.95}, step=1)
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "MedVision-AI",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._experiment_id: Optional[str] = None
        self._active_run_id: Optional[str] = None
        self._mlflow = None
        self._client = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> str:
        """Initialise MLflow and set up the experiment.

        Sets the tracking URI, creates the experiment if it doesn't exist,
        and returns the experiment ID.

        Returns:
            The MLflow experiment ID.

        Raises:
            ImportError: If mlflow is not installed.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for experiment tracking. "
                "Install it with: pip install mlflow"
            ) from exc

        self._mlflow = mlflow
        self._client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(
                "Created MLflow experiment '%s' (id=%s)",
                self.experiment_name,
                self._experiment_id,
            )
        else:
            self._experiment_id = experiment.experiment_id
            logger.info(
                "Using existing MLflow experiment '%s' (id=%s)",
                self.experiment_name,
                self._experiment_id,
            )

        mlflow.set_experiment(self.experiment_name)
        return self._experiment_id

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new MLflow run within the configured experiment.

        Args:
            run_name: Human-readable name for the run.
            tags: Optional dictionary of tags to attach.

        Returns:
            The run ID of the newly created run.
        """
        if self._mlflow is None:
            self.setup()

        run = self._mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            tags=tags,
        )
        self._active_run_id = run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", run_name, self._active_run_id)
        return self._active_run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run.

        Args:
            status: Run status string (e.g. 'FINISHED', 'FAILED', 'KILLED').
        """
        if self._mlflow is not None and self._mlflow.active_run() is not None:
            self._mlflow.end_run(status=status)
            logger.info("MLflow run ended: %s (status=%s)", self._active_run_id, status)
        self._active_run_id = None

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Retrieve metadata and data for a specific run.

        Args:
            run_id: The MLflow run ID.

        Returns:
            Dictionary containing run info, params, metrics, and tags.

        Raises:
            ImportError: If MLflow is not set up.
        """
        if self._client is None:
            self.setup()

        run = self._client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
            "artifact_uri": run.info.artifact_uri,
        }

    # ------------------------------------------------------------------
    # Parameter & Metric logging
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of parameters to the active run.

        Parameter values are converted to strings as required by MLflow.

        Args:
            params: Dictionary of parameter names to values.
        """
        if self._mlflow is None:
            self.setup()

        str_params = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(str_params)
        logger.debug("Logged %d parameters: %s", len(str_params), list(str_params.keys()))

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log a batch of metrics to the active run.

        Args:
            metrics: Dictionary of metric names to float values.
            step: Optional step number for time-series metrics.
        """
        if self._mlflow is None:
            self.setup()

        self._mlflow.log_metrics(metrics, step=step)
        logger.debug(
            "Logged %d metrics at step %s: %s",
            len(metrics),
            step,
            list(metrics.keys()),
        )

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def log_model(
        self,
        model: Any,
        name: str,
        flavor: str = "pytorch",
        conda_env: Optional[Dict[str, Any]] = None,
        registered_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Log a model to MLflow with the specified flavor.

        Args:
            model: The model object to log.
            name: Artifact path within the run.
            flavor: MLflow model flavor ('pytorch', 'sklearn', 'tensorflow', etc.).
            conda_env: Conda environment specification for reproducibility.
            registered_model_name: If provided, register the model in the Model Registry.
            **kwargs: Additional arguments passed to the flavor-specific log function.

        Returns:
            The artifact URI of the logged model.

        Raises:
            ValueError: If an unsupported flavor is specified.
        """
        if self._mlflow is None:
            self.setup()

        flavor_lower = flavor.lower()

        if flavor_lower == "pytorch":
            log_fn = self._mlflow.pytorch.log_model
            log_fn(
                model,
                artifact_path=name,
                conda_env=conda_env,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif flavor_lower == "sklearn":
            log_fn = self._mlflow.sklearn.log_model
            log_fn(
                model,
                artifact_path=name,
                conda_env=conda_env,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif flavor_lower == "tensorflow":
            log_fn = self._mlflow.tensorflow.log_model
            log_fn(
                model,
                artifact_path=name,
                conda_env=conda_env,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            # Fallback: use generic mlflow.log_artifact for custom models
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                import torch

                model_path = Path(tmpdir) / f"{name}.pt"
                torch.save(model.state_dict() if hasattr(model, "state_dict") else model, model_path)
                self._mlflow.log_artifact(str(model_path), artifact_path=name)

        artifact_uri = f"{self._mlflow.active_run().info.artifact_uri}/{name}"
        logger.info("Model logged as '%s' (flavor=%s) at %s", name, flavor, artifact_uri)
        return artifact_uri

    # ------------------------------------------------------------------
    # Artifact logging
    # ------------------------------------------------------------------

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            path: Local path to the file or directory to log.
            artifact_path: Optional subdirectory within the run's artifact store.
        """
        if self._mlflow is None:
            self.setup()

        file_path = Path(path)
        if file_path.is_dir():
            self._mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)

        logger.info("Artifact logged: %s -> %s", path, artifact_path or "root")

    def log_artifacts(self, directory: str, artifact_path: Optional[str] = None) -> None:
        """Log all files in a directory as artifacts.

        Args:
            directory: Local directory containing artifacts.
            artifact_path: Optional subdirectory within the run's artifact store.
        """
        if self._mlflow is None:
            self.setup()

        self._mlflow.log_artifacts(directory, artifact_path=artifact_path)
        logger.info("Artifacts logged from directory: %s", directory)

    # ------------------------------------------------------------------
    # Search & Query
    # ------------------------------------------------------------------

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for runs in the configured experiment.

        Args:
            filter_string: MLflow filter expression (e.g. "metrics.accuracy > 0.9").
            order_by: List of metric/param keys to order by.
            max_results: Maximum number of runs to return.

        Returns:
            List of run dictionaries with info, params, and metrics.
        """
        if self._client is None:
            self.setup()

        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
            order_by=order_by or [],
            max_results=max_results,
        )

        return [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "params": r.data.params,
                "metrics": r.data.metrics,
            }
            for r in runs
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def experiment_id(self) -> Optional[str]:
        """The current MLflow experiment ID."""
        return self._experiment_id

    @property
    def active_run_id(self) -> Optional[str]:
        """The currently active MLflow run ID."""
        return self._active_run_id

    @property
    def artifact_uri(self) -> Optional[str]:
        """The artifact URI for the active run."""
        if self._mlflow is not None and self._mlflow.active_run() is not None:
            return self._mlflow.active_run().info.artifact_uri
        return None
