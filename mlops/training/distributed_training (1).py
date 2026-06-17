"""
DistributedTraining - Multi-GPU and distributed training for MedVision-AI.

Supports both DataParallel (single-machine multi-GPU) and DistributedDataParallel
(multi-machine) training strategies with automatic process group management.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        backend: Distributed backend ('nccl', 'gloo', 'mpi').
        strategy: Parallelism strategy ('dp' for DataParallel, 'ddp' for DistributedDataParallel).
        world_size: Total number of processes participating in training.
        rank: Rank of the current process.
        local_rank: Local rank on the current machine.
        init_method: URL for process group initialisation (e.g. 'env://').
        master_addr: Address of the master node.
        master_port: Port of the master node.
        find_unused_parameters: Whether to find unused parameters in DDP forward pass.
        gradient_as_bucket_view: Use gradient bucket view to reduce memory in DDP.
        num_workers: Number of DataLoader workers per process.
        pin_memory: Whether to pin memory for DataLoaders.
        sync_batchnorm: Whether to use SyncBatchNorm for consistent batch norm across GPUs.
    """

    backend: str = "nccl"
    strategy: str = "ddp"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    master_addr: str = "localhost"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    sync_batchnorm: bool = True

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is enabled (world_size > 1)."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0


@dataclass
class DistributedTrainingResult:
    """Result of a distributed training run.

    Attributes:
        model_state_dict: State dict of the trained model (from rank 0).
        total_epochs: Number of epochs completed.
        final_loss: Final training loss.
        gpu_utilisation: Per-GPU memory usage statistics.
        training_time: Total wall-clock training time in seconds.
        rank_results: Per-rank metric summaries.
    """

    model_state_dict: Optional[Dict[str, Any]] = None
    total_epochs: int = 0
    final_loss: float = float("inf")
    gpu_utilisation: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    rank_results: Dict[int, Dict[str, float]] = field(default_factory=dict)


class DistributedTrainer:
    """Multi-GPU and distributed training orchestrator.

    Manages the lifecycle of distributed training including process group
    initialisation, model wrapping, data sampler configuration, and
    clean teardown.

    Supports:
    - **DataParallel (DP)**: Simple single-machine multi-GPU training.
    - **DistributedDataParallel (DDP)**: Efficient multi-machine multi-GPU training
      with gradient synchronisation and overlapping computation/communication.

    Args:
        config: A DistributedConfig instance controlling parallelism behaviour.

    Example::

        config = DistributedConfig(strategy="ddp", world_size=4)
        trainer = DistributedTrainer(config)
        trainer.setup_distributed()
        model = trainer.train_distributed(model, dataloader, num_epochs=20)
        trainer.cleanup()
    """

    def __init__(self, config: Optional[DistributedConfig] = None) -> None:
        self.config = config or DistributedConfig()
        self._is_initialized = False
        self._device: Optional[torch.device] = None
        self._wrapped_model: Optional[nn.Module] = None

        logger.info(
            "DistributedTrainer | strategy=%s | world_size=%d",
            self.config.strategy,
            self.config.world_size,
        )

    # ------------------------------------------------------------------
    # Process group management
    # ------------------------------------------------------------------

    def setup_distributed(self) -> None:
        """Initialise the distributed training environment.

        Sets environment variables, creates the process group, and configures
        the appropriate CUDA device for the current process.

        Raises:
            RuntimeError: If CUDA is not available when NCCL backend is selected.
        """
        if not self.config.is_distributed:
            logger.info("Single-GPU mode; skipping distributed setup.")
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._is_initialized = True
            return

        # Set environment variables for init_method='env://'
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)

        if self.config.backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError(
                "NCCL backend requires CUDA, but no CUDA devices are available."
            )

        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank,
        )

        # Bind this process to its corresponding GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self._device = torch.device(f"cuda:{self.config.local_rank}")
        else:
            self._device = torch.device("cpu")

        self._is_initialized = True
        logger.info(
            "Distributed setup complete | rank=%d | local_rank=%d | device=%s",
            self.config.rank,
            self.config.local_rank,
            self._device,
        )

    def cleanup(self) -> None:
        """Destroy the distributed process group and release resources.

        Should be called at the end of training to properly shut down
        the distributed backend.
        """
        if self._is_initialized and self.config.is_distributed:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed.")

        self._is_initialized = False
        self._wrapped_model = None

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap the model with the selected parallelism strategy.

        Args:
            model: The base model to wrap.

        Returns:
            The wrapped model (DP or DDP).

        Raises:
            ValueError: If an unsupported strategy is specified.
        """
        model = model.to(self._device)

        if not self.config.is_distributed:
            return model

        if self.config.strategy.lower() == "dp":
            wrapped = DP(model)
            logger.info("Model wrapped with DataParallel (%d GPUs)", torch.cuda.device_count())
            return wrapped

        if self.config.strategy.lower() == "ddp":
            if self.config.sync_batchnorm:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            wrapped = DDP(
                model,
                device_ids=[self.config.local_rank] if self._device.type == "cuda" else None,
                output_device=self._device if self._device.type == "cuda" else None,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            )
            logger.info(
                "Model wrapped with DistributedDataParallel | device_ids=[%d]",
                self.config.local_rank,
            )
            return wrapped

        raise ValueError(f"Unsupported strategy: {self.config.strategy}")

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap a DP/DDP model to get the underlying module.

        Args:
            model: The wrapped model.

        Returns:
            The original unwrapped model.
        """
        if isinstance(model, DDP):
            return model.module
        if isinstance(model, DP):
            return model.module
        return model

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def _create_distributed_sampler(
        self, dataloader: DataLoader
    ) -> Optional[DistributedSampler]:
        """Create a DistributedSampler for the given DataLoader.

        Args:
            dataloader: The original DataLoader.

        Returns:
            A DistributedSampler, or None if not running in distributed mode.
        """
        if not self.config.is_distributed or self.config.strategy.lower() != "ddp":
            return None

        dataset = dataloader.dataset
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
        )

    def prepare_dataloader(
        self,
        dataloader: DataLoader,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Prepare a DataLoader for distributed training.

        Wraps the dataset with a DistributedSampler and adjusts worker
        and memory settings from the config.

        Args:
            dataloader: The original DataLoader to prepare.
            batch_size: Override batch size (per-GPU). If None, uses original.

        Returns:
            A DataLoader configured for distributed training.
        """
        sampler = self._create_distributed_sampler(dataloader)
        if sampler is None:
            return dataloader

        new_loader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=batch_size or dataloader.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        return new_loader

    # ------------------------------------------------------------------
    # Distributed metrics helpers
    # ------------------------------------------------------------------

    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average a tensor across all distributed processes.

        Args:
            tensor: The tensor to reduce.

        Returns:
            The averaged tensor.
        """
        if not self.config.is_distributed:
            return tensor

        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.config.world_size
        return rt

    def _gather_gpu_stats(self) -> Dict[str, float]:
        """Collect GPU memory utilisation statistics.

        Returns:
            Dictionary of GPU memory stats.
        """
        stats: Dict[str, float] = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                stats[f"gpu_{i}_allocated_gb"] = round(allocated, 3)
                stats[f"gpu_{i}_reserved_gb"] = round(reserved, 3)
        return stats

    # ------------------------------------------------------------------
    # Core distributed training loop
    # ------------------------------------------------------------------

    def train_distributed(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_epochs: int = 10,
        criterion: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
    ) -> nn.Module:
        """Execute the distributed training loop.

        Wraps the model with the configured parallelism strategy, runs
        training for the specified epochs, and returns the trained model.

        Args:
            model: The neural network model to train.
            dataloader: DataLoader for training data.
            num_epochs: Number of training epochs.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            learning_rate: Learning rate for the AdamW optimiser.

        Returns:
            The trained (unwrapped) model with updated weights.

        Raises:
            RuntimeError: If setup_distributed() has not been called.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Must call setup_distributed() before train_distributed()."
            )

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Prepare dataloader with distributed sampler
        dist_loader = self.prepare_dataloader(dataloader)
        sampler = getattr(dist_loader, "sampler", None)

        # Wrap model
        wrapped_model = self._wrap_model(model)
        self._wrapped_model = wrapped_model

        optimizer = torch.optim.AdamW(
            wrapped_model.parameters(), lr=learning_rate
        )

        start_time = time.time()
        result = DistributedTrainingResult()

        for epoch in range(1, num_epochs + 1):
            if sampler is not None and isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            wrapped_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets in dist_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = wrapped_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            # Reduce loss across processes
            if self.config.is_distributed:
                loss_tensor = torch.tensor([avg_loss], device=self._device)
                reduced = self._reduce_tensor(loss_tensor)
                avg_loss = reduced.item()

            if self.config.is_main_process or not self.config.is_distributed:
                logger.info(
                    "Epoch %d/%d | loss=%.4f | strategy=%s",
                    epoch,
                    num_epochs,
                    avg_loss,
                    self.config.strategy,
                )

            result.final_loss = avg_loss

        result.total_epochs = num_epochs
        result.training_time = time.time() - start_time
        result.gpu_utilisation = self._gather_gpu_stats()

        # Unwrap to get the base model with trained weights
        unwrapped = self._unwrap_model(wrapped_model)
        result.model_state_dict = unwrapped.state_dict()

        if self.config.is_main_process:
            logger.info(
                "Distributed training complete | epochs=%d | final_loss=%.4f | %.1fs",
                num_epochs,
                result.final_loss,
                result.training_time,
            )

        return unwrapped

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_gpus() -> List[int]:
        """Return a list of available GPU indices.

        Returns:
            List of GPU device indices.
        """
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    @staticmethod
    def get_gpu_memory(gpu_id: int = 0) -> Dict[str, float]:
        """Query memory stats for a specific GPU.

        Args:
            gpu_id: GPU device index.

        Returns:
            Dictionary with 'allocated_gb' and 'total_gb'.
        """
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "total_gb": 0.0}

        allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        total = torch.cuda.get_device_properties(gpu_id).total_mem / (1024 ** 3)
        return {
            "allocated_gb": round(allocated, 3),
            "total_gb": round(total, 3),
        }

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a Python object from source rank to all processes.

        Args:
            obj: The object to broadcast (must be picklable).
            src: Source rank for the broadcast.

        Returns:
            The broadcasted object on all ranks.
        """
        if not self.config.is_distributed:
            return obj

        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def barrier(self) -> None:
        """Synchronise all distributed processes at a barrier."""
        if self.config.is_distributed:
            dist.barrier()
