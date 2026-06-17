"""
TrainModel - Comprehensive training pipeline for MedVision-AI.

This module provides a full training loop with checkpointing, early stopping,
learning rate scheduling, and detailed metrics tracking for medical imaging models.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline.

    Attributes:
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 regularization factor.
        optimizer: Name of the optimizer to use (e.g., 'adam', 'sgd', 'adamw').
        scheduler: Learning rate scheduler type (e.g., 'cosine', 'step', 'plateau').
        scheduler_step_size: Step size for StepLR scheduler.
        scheduler_gamma: Gamma factor for scheduler decay.
        early_stopping_patience: Number of epochs to wait before early stopping.
        early_stopping_delta: Minimum improvement to reset patience counter.
        checkpoint_dir: Directory path for saving checkpoints.
        checkpoint_every: Save a checkpoint every N epochs.
        gradient_clipping: Max norm for gradient clipping (0 to disable).
        mixed_precision: Whether to use automatic mixed precision training.
        accumulation_steps: Number of gradient accumulation steps.
        seed: Random seed for reproducibility.
        device: Device string for training ('cuda', 'cpu', or 'auto').
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 5
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1
    seed: int = 42
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        """Resolve the torch device from the config string.

        Returns:
            Resolved torch.device instance.
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class EpochMetrics:
    """Metrics recorded for a single training or validation epoch.

    Attributes:
        epoch: The epoch number (1-indexed).
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        train_metrics: Dictionary of additional training metrics.
        val_metrics: Dictionary of additional validation metrics.
        learning_rate: Learning rate used during the epoch.
        elapsed_seconds: Wall-clock time for the epoch in seconds.
    """

    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class TrainingResult:
    """Result of a complete training run.

    Attributes:
        best_val_loss: Best validation loss achieved.
        best_epoch: Epoch at which the best validation loss was recorded.
        final_train_loss: Training loss at the last epoch.
        final_val_loss: Validation loss at the last epoch.
        total_epochs: Total number of epochs completed.
        total_training_time: Cumulative training time in seconds.
        epoch_metrics: Chronological list of per-epoch metrics.
        best_model_path: Path to the checkpoint with the best validation loss.
        stopped_early: Whether training ended due to early stopping.
        model_state_dict: State dictionary of the best model.
    """

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    final_train_loss: float = float("inf")
    final_val_loss: float = float("inf")
    total_epochs: int = 0
    total_training_time: float = 0.0
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    best_model_path: Optional[str] = None
    stopped_early: bool = False
    model_state_dict: Optional[Dict[str, Any]] = None


class EarlyStopping:
    """Early stopping handler to halt training when validation loss plateaus.

    Args:
        patience: Number of epochs with no improvement before stopping.
        delta: Minimum change in monitored value to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check whether the current score is an improvement over the best."""
        if self.mode == "min":
            return current < best - self.delta
        return current > best + self.delta

    def step(self, score: float) -> bool:
        """Record a new score and determine whether to stop.

        Args:
            score: The metric value for the current epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class TrainModel:
    """Full-featured training pipeline for medical imaging models.

    The TrainModel class encapsulates the entire training lifecycle including
    optimizer/scheduler construction, mixed-precision training, gradient
    accumulation, early stopping, checkpointing, and metric aggregation.

    Args:
        config: A TrainingConfig instance controlling training behaviour.

    Example::

        config = TrainingConfig(learning_rate=3e-4, early_stopping_patience=7)
        trainer = TrainModel(config)
        result = trainer.train(model, train_loader, val_loader, num_epochs=50)
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        self._epoch_metrics: List[EpochMetrics] = []
        self._global_step: int = 0
        self._best_val_loss: float = float("inf")
        self._best_model_path: Optional[str] = None

        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        logger.info(
            "TrainModel initialised | device=%s | lr=%.2e | optimizer=%s",
            self.device,
            self.config.learning_rate,
            self.config.optimizer,
        )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Construct the optimizer from config.

        Args:
            model: The model whose parameters will be optimised.

        Returns:
            An initialised optimizer.

        Raises:
            ValueError: If an unsupported optimizer name is specified.
        """
        name = self.config.optimizer.lower()
        params = model.parameters()

        if name == "adam":
            return torch.optim.Adam(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, num_epochs: int
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Construct the learning-rate scheduler from config.

        Args:
            optimizer: The optimizer whose learning rate will be scheduled.
            num_epochs: Total number of training epochs (used by cosine annealing).

        Returns:
            A scheduler instance, or None if no scheduler is configured.
        """
        name = self.config.scheduler.lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.config.scheduler_gamma, patience=3
            )
        raise ValueError(f"Unsupported scheduler: {name}")

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: Optional[torch.amp.GradScaler],
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a single training epoch.

        Args:
            model: The model to train.
            train_loader: DataLoader yielding training batches.
            optimizer: The optimiser instance.
            criterion: Loss function.
            scaler: GradScaler for mixed-precision (or None).

        Returns:
            A tuple of (average_loss, extra_metrics).
        """
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with optional AMP
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / self.config.accumulation_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.config.gradient_clipping > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clipping
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets) / self.config.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clipping
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * self.config.accumulation_steps * inputs.size(0)
            total_samples += inputs.size(0)

            # Compute accuracy for classification tasks
            if outputs.dim() <= 2 and outputs.size(-1) > 1:
                preds = outputs.argmax(dim=-1)
                if targets.dim() > 1:
                    targets_acc = targets.argmax(dim=-1)
                else:
                    targets_acc = targets
                correct += (preds == targets_acc).sum().item()

            self._global_step += 1

        avg_loss = running_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        extra_metrics: Dict[str, float] = {"accuracy": accuracy}
        return avg_loss, extra_metrics

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on the validation set.

        Args:
            model: The model to evaluate.
            val_loader: DataLoader yielding validation batches.
            criterion: Loss function.

        Returns:
            A tuple of (average_loss, extra_metrics).
        """
        model.eval()
        running_loss = 0.0
        total_samples = 0
        correct = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if outputs.dim() <= 2 and outputs.size(-1) > 1:
                preds = outputs.argmax(dim=-1)
                if targets.dim() > 1:
                    targets_acc = targets.argmax(dim=-1)
                else:
                    targets_acc = targets
                correct += (preds == targets_acc).sum().item()

        avg_loss = running_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        extra_metrics: Dict[str, float] = {"accuracy": accuracy}
        return avg_loss, extra_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> str:
        """Persist a training checkpoint to disk.

        Args:
            model: The model whose state will be saved.
            optimizer: Optimizer state to persist.
            scheduler: Scheduler state to persist (or None).
            epoch: Current epoch number.
            val_loss: Validation loss at this epoch.
            is_best: Whether this is the best checkpoint so far.

        Returns:
            The file path of the saved checkpoint.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:04d}_{timestamp}.pt"
        filepath = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "global_step": self._global_step,
            "config": {
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
            },
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved: %s (val_loss=%.4f)", filepath, val_loss)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._best_model_path = str(best_path)
            logger.info("Best model updated at %s", best_path)

        return str(filepath)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        criterion: Optional[nn.Module] = None,
    ) -> TrainingResult:
        """Execute the full training loop.

        Runs training for the specified number of epochs with early stopping,
        checkpointing, learning-rate scheduling, and mixed-precision support.

        Args:
            model: The neural network model to train.
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            num_epochs: Maximum number of training epochs.
            criterion: Loss function. Defaults to CrossEntropyLoss.

        Returns:
            A TrainingResult dataclass with comprehensive training statistics.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        model = model.to(self.device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, num_epochs)
        early_stopper = EarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta,
        )
        scaler: Optional[torch.amp.GradScaler] = None
        if self.config.mixed_precision and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        result = TrainingResult()
        training_start = time.time()
        best_epoch = 0

        logger.info(
            "Starting training: %d epochs | device=%s | mixed_precision=%s",
            num_epochs,
            self.device,
            scaler is not None,
        )

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_loss, train_metrics = self._train_one_epoch(
                model, train_loader, optimizer, criterion, scaler
            )

            # --- Validate ---
            val_loss, val_metrics = self._validate(model, val_loader, criterion)

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            # --- Record metrics ---
            epoch_metric = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                elapsed_seconds=epoch_time,
            )
            self._epoch_metrics.append(epoch_metric)
            result.epoch_metrics.append(epoch_metric)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | "
                "train_acc=%.4f val_acc=%.4f | lr=%.2e | %.1fs",
                epoch,
                num_epochs,
                train_loss,
                val_loss,
                train_metrics.get("accuracy", 0.0),
                val_metrics.get("accuracy", 0.0),
                current_lr,
                epoch_time,
            )

            # --- Checkpoint ---
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
                best_epoch = epoch

            if epoch % self.config.checkpoint_every == 0 or is_best:
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, is_best
                )

            # --- Scheduler step ---
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # --- Early stopping ---
            if early_stopper.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                result.stopped_early = True
                break

        total_time = time.time() - training_start

        # Populate result
        result.best_val_loss = self._best_val_loss
        result.best_epoch = best_epoch
        result.final_train_loss = train_loss
        result.final_val_loss = val_loss
        result.total_epochs = len(result.epoch_metrics)
        result.total_training_time = total_time
        result.best_model_path = self._best_model_path
        result.model_state_dict = model.state_dict()

        logger.info(
            "Training complete | epochs=%d | best_val_loss=%.4f @ epoch %d | %.1fs",
            result.total_epochs,
            result.best_val_loss,
            result.best_epoch,
            total_time,
        )

        return result

    def resume_training(self, checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Resume training from a previously saved checkpoint.

        Loads model weights, optimizer state, scheduler state, and training
        metadata so that training can continue seamlessly.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            A tuple of (model_state_dict, checkpoint_metadata).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or incompatible.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'model_state_dict'")

        self._global_step = checkpoint.get("global_step", 0)
        self._best_val_loss = checkpoint.get("val_loss", float("inf"))

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "val_loss": checkpoint.get("val_loss", float("inf")),
            "global_step": self._global_step,
            "config": checkpoint.get("config", {}),
        }

        logger.info(
            "Resumed from checkpoint: epoch=%d val_loss=%.4f",
            metadata["epoch"],
            metadata["val_loss"],
        )

        return checkpoint["model_state_dict"], metadata

    def export_model(self, output_path: str, model: Optional[nn.Module] = None) -> str:
        """Export the trained model for deployment.

        Saves the model in a self-contained format suitable for inference,
        including the model state dict and minimal metadata.

        Args:
            output_path: Destination file path for the exported model.
            model: Model instance to export. If None, the best checkpoint is used.

        Returns:
            The absolute path of the exported model file.

        Raises:
            ValueError: If no model is provided and no best checkpoint exists.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if model is not None:
            state_dict = model.state_dict()
        elif self._best_model_path is not None and Path(self._best_model_path).exists():
            checkpoint = torch.load(
                self._best_model_path, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError(
                "No model provided and no best checkpoint available for export."
            )

        export_data = {
            "model_state_dict": state_dict,
            "export_timestamp": datetime.now().isoformat(),
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
            },
            "framework": "MedVision-AI",
        }

        torch.save(export_data, out)
        abs_path = str(out.resolve())
        logger.info("Model exported to %s", abs_path)
        return abs_path
