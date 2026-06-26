"""
Model Loader Module for MedVision-AI.

Handles loading, saving, and managing deep learning models with support
for versioned model registries, device placement, and model introspection.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata record for a registered model.

    Attributes:
        name: Human-readable model name.
        version: Semantic version string.
        path: Filesystem path to the model artefact.
        architecture: Model architecture identifier.
        description: Brief description of the model.
        created_at: UNIX timestamp of creation.
        size_bytes: Model file size in bytes.
        tags: Arbitrary tags for filtering and grouping.
        checksum: SHA-256 checksum of the model file.
    """

    name: str
    version: str
    path: str
    architecture: str = ""
    description: str = ""
    created_at: float = 0.0
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise model info to a plain dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "architecture": self.architecture,
            "description": self.description,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "checksum": self.checksum,
        }


class ModelRegistry:
    """Simple file-based model registry.

    Stores model metadata as JSON files alongside model artefacts.
    The registry maps ``(name, version)`` pairs to model paths and
    metadata records.

    Args:
        registry_path: Root directory for the model registry.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)
        self._registry_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._registry_path / "registry_index.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def register(self, model_info: ModelInfo) -> None:
        """Add a model entry to the registry index.

        Args:
            model_info: Metadata for the model to register.
        """
        key = f"{model_info.name}:{model_info.version}"
        self._index[key] = model_info.to_dict()
        self._save_index()
        logger.info("Registered model %s v%s", model_info.name, model_info.version)

    def lookup(self, model_name: str, version: str) -> Optional[ModelInfo]:
        """Look up a model by name and version.

        Args:
            model_name: Name of the model.
            version: Version string.

        Returns:
            A :class:`ModelInfo` if found, else ``None``.
        """
        key = f"{model_name}:{version}"
        data = self._index.get(key)
        if data is None:
            return None
        return ModelInfo(**data)

    def list_models(self) -> list[ModelInfo]:
        """Return all registered models.

        Returns:
            A list of :class:`ModelInfo` objects.
        """
        return [ModelInfo(**data) for data in self._index.values()]

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the registry index from disk.

        Returns:
            The index dictionary, or an empty dict if the file does not
            exist.
        """
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_index(self) -> None:
        """Persist the registry index to disk."""
        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2, default=str)


class ModelLoader:
    """Model loading and management utility for MedVision-AI.

    ``ModelLoader`` handles loading PyTorch models from disk or a
    versioned registry, saving models with metadata, and listing
    available models.  It supports automatic device placement and
    model validation.

    Args:
        config: Configuration dictionary.  Expected keys include
            ``device``, ``registry_path``, and ``strict_loading``.

    Example::

        config = {"device": "cuda", "registry_path": "./model_registry"}
        loader = ModelLoader(config)
        model = loader.load("models/chest_xray_v2.pt")
        loader.save(model, "models/chest_xray_v3.pt")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._device = torch.device(config.get("device", "cpu"))
        self._strict_loading: bool = config.get("strict_loading", True)
        self._registry_path = Path(config.get("registry_path", "./model_registry"))
        self._registry = ModelRegistry(self._registry_path)
        self._loaded_models: dict[str, nn.Module] = {}

        logger.info(
            "ModelLoader initialised — device=%s, registry=%s",
            self._device,
            self._registry_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: str | Path) -> nn.Module:
        """Load a PyTorch model from a file path.

        The model is moved to the configured device and set to evaluation
        mode.  The loaded model is also cached internally for fast
        subsequent retrieval.

        Args:
            model_path: Path to a ``.pt`` or ``.pth`` model file.

        Returns:
            The loaded ``nn.Module`` in evaluation mode.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
            RuntimeError: If the file cannot be deserialised as a model.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cache_key = str(model_path.resolve())
        if cache_key in self._loaded_models:
            logger.debug("Returning cached model: %s", model_path)
            return self._loaded_models[cache_key]

        logger.info("Loading model from %s", model_path)
        start = time.perf_counter()

        try:
            model = torch.load(
                str(model_path),
                map_location=self._device,
                weights_only=False,
            )

            # Handle cases where the checkpoint is a dict with a 'model_state_dict' key
            if isinstance(model, dict):
                state_dict = model.get("model_state_dict", model.get("state_dict", model))
                if isinstance(state_dict, dict):
                    # We need the model class to load state_dict;
                    # if no class is available we wrap it as a placeholder
                    model = _StateDictWrapper(state_dict)
            elif not isinstance(model, nn.Module):
                raise RuntimeError(
                    f"Loaded object is not an nn.Module: {type(model)}"
                )

            model = model.to(self._device)
            model.eval()

            elapsed = time.perf_counter() - start
            logger.info(
                "Model loaded successfully in %.2fs — params: %d",
                elapsed,
                sum(p.numel() for p in model.parameters()),
            )

            self._loaded_models[cache_key] = model
            return model

        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

    def load_from_registry(self, model_name: str, version: str) -> nn.Module:
        """Load a model from the versioned registry by name and version.

        Args:
            model_name: Registered model name.
            version: Semantic version string.

        Returns:
            The loaded ``nn.Module`` in evaluation mode.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        model_info = self._registry.lookup(model_name, version)
        if model_info is None:
            available = [
                f"{m.name}:{m.version}" for m in self._registry.list_models()
            ]
            raise ValueError(
                f"Model '{model_name}' v{version} not found in registry. "
                f"Available: {available}"
            )

        logger.info(
            "Loading model from registry — %s v%s", model_name, version
        )
        return self.load(model_info.path)

    def save(self, model: nn.Module, path: str | Path) -> ModelInfo:
        """Save a model to disk and register it in the model registry.

        The model is saved as a full ``nn.Module`` using
        :func:`torch.save`, and its metadata is registered in the
        internal model registry.

        Args:
            model: The PyTorch model to save.
            path: Destination file path for the model.

        Returns:
            A :class:`ModelInfo` record for the saved model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model to %s", path)
        start = time.perf_counter()

        model_cpu = model.cpu()
        torch.save(model_cpu, str(path))

        elapsed = time.perf_counter() - start
        size_bytes = path.stat().st_size
        checksum = self._compute_checksum(path)

        # Extract metadata from the model if available
        model_name = getattr(model, "model_name", path.stem)
        version = getattr(model, "model_version", "1.0.0")
        architecture = type(model).__name__
        description = getattr(model, "description", "")

        model_info = ModelInfo(
            name=model_name,
            version=version,
            path=str(path.resolve()),
            architecture=architecture,
            description=description,
            created_at=time.time(),
            size_bytes=size_bytes,
            tags=["saved"],
            checksum=checksum,
        )

        self._registry.register(model_info)

        logger.info(
            "Model saved in %.2fs — size: %.2fMB, checksum: %s",
            elapsed,
            size_bytes / (1024 * 1024),
            checksum[:12],
        )

        return model_info

    def list_available_models(self) -> list[ModelInfo]:
        """List all models available in the registry.

        Returns:
            A list of :class:`ModelInfo` objects for all registered
            models.
        """
        return self._registry.list_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_checksum(path: Path, chunk_size: int = 8192) -> str:
        """Compute a SHA-256 checksum of a file.

        Args:
            path: Path to the file.
            chunk_size: Read buffer size in bytes.

        Returns:
            Hex-encoded SHA-256 checksum string.
        """
        import hashlib

        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()


class _StateDictWrapper(nn.Module):
    """Minimal wrapper that loads a state dict into a sequential container.

    This is a fallback for checkpoints that only contain state dicts
    without the model class definition.

    Args:
        state_dict: A dictionary of tensor parameters.
    """

    def __init__(self, state_dict: dict[str, Any]) -> None:
        super().__init__()
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(key.replace(".", "_"), value)
