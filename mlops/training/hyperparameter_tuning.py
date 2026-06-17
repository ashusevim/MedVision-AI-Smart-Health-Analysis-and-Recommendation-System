"""
HyperparameterTuning - Automated hyperparameter optimisation for MedVision-AI.

Supports random search, grid search, and Bayesian optimisation (via Optuna)
to find the best hyperparameter configuration for medical imaging models.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of trials for random/Bayesian search.
        metric: Metric to optimise ('val_loss' or 'val_accuracy').
        direction: Optimisation direction ('minimize' or 'maximize').
        timeout: Maximum time in seconds for the entire tuning run.
        pruning: Whether to enable early pruning of unpromising trials.
        pruning_patience: Epochs to wait before pruning a trial.
        seed: Random seed for reproducibility.
        study_name: Name for the Optuna study.
        storage: Optuna storage URL (e.g. 'sqlite:///optuna.db').
    """

    n_trials: int = 50
    metric: str = "val_loss"
    direction: str = "minimize"
    timeout: Optional[float] = None
    pruning: bool = True
    pruning_patience: int = 5
    seed: int = 42
    study_name: str = "medvision_tuning"
    storage: Optional[str] = None


@dataclass
class BestParams:
    """Result of a hyperparameter tuning run.

    Attributes:
        params: The best hyperparameter dictionary found.
        score: The metric value achieved with the best params.
        trial_number: The trial index that produced the best result.
        all_results: List of (params, score) for every completed trial.
    """

    params: Dict[str, Any]
    score: float
    trial_number: int = 0
    all_results: List[Dict[str, Any]] = field(default_factory=list)


# Type alias for the model factory function
ModelFactory = Callable[..., nn.Module]


class HyperparameterTuner:
    """Automated hyperparameter optimisation engine.

    Provides random search, grid search, and Bayesian optimisation strategies
    to explore the hyperparameter space and identify the best configuration
    for a given model architecture and dataset.

    Args:
        config: A TuningConfig instance controlling the search behaviour.

    Example::

        config = TuningConfig(n_trials=30, direction="minimize")
        tuner = HyperparameterTuner(config)
        param_space = {"lr": [1e-3, 3e-4, 1e-4], "dropout": [0.1, 0.3, 0.5]}
        result = tuner.tune(model_fn, param_space, train_loader, val_loader)
    """

    def __init__(self, config: Optional[TuningConfig] = None) -> None:
        self.config = config or TuningConfig()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self._results: List[Dict[str, Any]] = []
        logger.info(
            "HyperparameterTuner initialised | trials=%d | metric=%s | direction=%s",
            self.config.n_trials,
            self.config.metric,
            self.config.direction,
        )

    # ------------------------------------------------------------------
    # Helper: evaluate a single hyperparameter configuration
    # ------------------------------------------------------------------

    def _evaluate_config(
        self,
        model_fn: ModelFactory,
        params: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
    ) -> float:
        """Train a model with the given params and return the optimisation metric.

        Args:
            model_fn: Factory that returns a fresh model instance given params.
            params: Hyperparameter values to evaluate.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Quick-training epochs for evaluation.

        Returns:
            The metric value (lower is better when direction='minimize').
        """
        model = model_fn(**params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        lr = params.get("lr", params.get("learning_rate", 1e-4))
        weight_decay = params.get("weight_decay", 1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_metric = float("inf") if self.config.direction == "minimize" else float("-inf")
        no_improve = 0

        for epoch in range(1, num_epochs + 1):
            # --- Train ---
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # --- Validate ---
            model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=-1)
                    total_correct += (preds == targets).sum().item()
                    total_samples += inputs.size(0)

            val_loss = total_loss / max(total_samples, 1)
            val_accuracy = total_correct / max(total_samples, 1)

            if self.config.metric == "val_loss":
                metric_val = val_loss
                improved = metric_val < best_val_metric
            else:
                metric_val = val_accuracy
                improved = metric_val > best_val_metric

            if improved:
                best_val_metric = metric_val
                no_improve = 0
            else:
                no_improve += 1

            # Pruning check
            if self.config.pruning and no_improve >= self.config.pruning_patience:
                logger.debug("Trial pruned at epoch %d (no improvement)", epoch)
                break

        return best_val_metric

    def _is_better(self, current: float, reference: float) -> bool:
        """Check whether *current* is an improvement over *reference*."""
        if self.config.direction == "minimize":
            return current < reference
        return current > reference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        model_fn: ModelFactory,
        param_space: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
        strategy: str = "bayesian",
    ) -> BestParams:
        """Run hyperparameter tuning with the specified strategy.

        Args:
            model_fn: Factory function that creates a model from params.
            param_space: Dictionary mapping parameter names to search spaces.
                - List values indicate discrete choices.
                - Tuple (low, high) indicates a continuous range.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Number of quick-training epochs per trial.
            strategy: One of 'random', 'grid', or 'bayesian'.

        Returns:
            A BestParams instance with the best configuration found.

        Raises:
            ValueError: If an unknown strategy is specified.
        """
        self._results = []
        strategy = strategy.lower()

        if strategy == "random":
            return self.random_search(
                model_fn, param_space, train_loader, val_loader, num_epochs
            )
        if strategy == "grid":
            return self.grid_search(
                model_fn, param_space, train_loader, val_loader, num_epochs
            )
        if strategy == "bayesian":
            return self.bayesian_optimization(
                model_fn, param_space, train_loader, val_loader, num_epochs
            )
        raise ValueError(f"Unknown tuning strategy: {strategy}")

    def random_search(
        self,
        model_fn: ModelFactory,
        param_space: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
    ) -> BestParams:
        """Perform random search over the hyperparameter space.

        For each trial, parameter values are sampled uniformly at random
        from the provided search space.

        Args:
            model_fn: Factory function for model creation.
            param_space: Search space specification.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Epochs per trial.

        Returns:
            BestParams with the best found configuration.
        """
        best_score = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params: Dict[str, Any] = {}
        best_trial = 0

        for trial_idx in range(self.config.n_trials):
            params = self._sample_params(param_space)
            logger.info("Random trial %d/%d | params=%s", trial_idx + 1, self.config.n_trials, params)

            score = self._evaluate_config(model_fn, params, train_loader, val_loader, num_epochs)
            self._results.append({"trial": trial_idx, "params": params, "score": score})

            if self._is_better(score, best_score):
                best_score = score
                best_params = params
                best_trial = trial_idx
                logger.info("  -> New best: %.4f", best_score)

        return BestParams(
            params=best_params,
            score=best_score,
            trial_number=best_trial,
            all_results=list(self._results),
        )

    def grid_search(
        self,
        model_fn: ModelFactory,
        param_space: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
    ) -> BestParams:
        """Perform exhaustive grid search over the hyperparameter space.

        Every combination of parameter values is evaluated. Continuous ranges
        specified as tuples are discretised into evenly-spaced values.

        Args:
            model_fn: Factory function for model creation.
            param_space: Search space specification.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Epochs per trial.

        Returns:
            BestParams with the best found configuration.
        """
        grid = self._build_grid(param_space)
        total = len(grid)
        logger.info("Grid search: %d combinations", total)

        best_score = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params: Dict[str, Any] = {}
        best_trial = 0

        for idx, params in enumerate(grid):
            logger.info("Grid trial %d/%d | params=%s", idx + 1, total, params)
            score = self._evaluate_config(model_fn, params, train_loader, val_loader, num_epochs)
            self._results.append({"trial": idx, "params": params, "score": score})

            if self._is_better(score, best_score):
                best_score = score
                best_params = params
                best_trial = idx
                logger.info("  -> New best: %.4f", best_score)

        return BestParams(
            params=best_params,
            score=best_score,
            trial_number=best_trial,
            all_results=list(self._results),
        )

    def bayesian_optimization(
        self,
        model_fn: ModelFactory,
        param_space: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
    ) -> BestParams:
        """Perform Bayesian optimisation using Optuna.

        Uses Tree-structured Parzen Estimators (TPE) to efficiently explore
        the hyperparameter space by modelling the objective function.

        Args:
            model_fn: Factory function for model creation.
            param_space: Search space specification.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Epochs per trial.

        Returns:
            BestParams with the best found configuration.

        Raises:
            ImportError: If optuna is not installed.
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError as exc:
            raise ImportError(
                "optuna is required for Bayesian optimisation. "
                "Install it with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self.config.seed)
        pruner = MedianPruner(
            n_startup_trials=5, n_warmup_steps=3, interval_steps=1
        )

        direction = self.config.direction
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner if self.config.pruning else None,
            load_if_exists=True,
        )

        def _objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, param_space)
            return self._evaluate_config(
                model_fn, params, train_loader, val_loader, num_epochs
            )

        study.optimize(
            _objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
        )

        best_trial_result = study.best_trial
        best_params = best_trial_result.params
        best_score = best_trial_result.value

        # Collect all trial results
        all_results = [
            {"trial": t.number, "params": t.params, "score": t.value}
            for t in study.trials
            if t.value is not None
        ]
        self._results = all_results

        logger.info(
            "Bayesian optimisation complete | best_score=%.4f | best_params=%s",
            best_score,
            best_params,
        )

        return BestParams(
            params=best_params,
            score=best_score,
            trial_number=best_trial_result.number,
            all_results=all_results,
        )

    # ------------------------------------------------------------------
    # Parameter sampling utilities
    # ------------------------------------------------------------------

    def _sample_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a random configuration from the search space.

        Lists are treated as categorical choices; tuples (low, high) as
        continuous ranges with uniform sampling.
        """
        params: Dict[str, Any] = {}
        for name, space in param_space.items():
            if isinstance(space, list):
                params[name] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = random.randint(low, high)
                else:
                    params[name] = random.uniform(low, high)
            else:
                params[name] = space
        return params

    def _suggest_params(
        self,
        trial: Any,
        param_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use an Optuna trial to suggest parameters from the search space.

        Args:
            trial: An active Optuna trial object.
            param_space: Search space specification.

        Returns:
            Dictionary of suggested parameter values.
        """
        params: Dict[str, Any] = {}
        for name, space in param_space.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                elif isinstance(low, float) or isinstance(high, float):
                    # Detect log-scale for learning rates
                    log = low > 0 and high > 0 and high / low > 100
                    params[name] = trial.suggest_float(name, low, high, log=log)
                else:
                    params[name] = trial.suggest_categorical(name, [low, high])
            else:
                params[name] = space
        return params

    def _build_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build an exhaustive grid from the parameter space.

        Continuous ranges are discretised into 5 evenly-spaced values.
        """
        discretised: Dict[str, List[Any]] = {}
        grid_points = 5

        for name, space in param_space.items():
            if isinstance(space, list):
                discretised[name] = space
            elif isinstance(space, tuple) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    step = max(1, (high - low) // (grid_points - 1))
                    discretised[name] = list(range(low, high + 1, step))
                else:
                    discretised[name] = [
                        round(v, 6)
                        for v in [
                            low + i * (high - low) / (grid_points - 1)
                            for i in range(grid_points)
                        ]
                    ]
            else:
                discretised[name] = [space]

        keys = list(discretised.keys())
        values = list(discretised.values())
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]
