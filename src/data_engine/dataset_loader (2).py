"""
Dataset Loader Module for MedVision-AI.

Provides unified dataset loading across multiple medical data formats
including CSV, JSON, Parquet, and DICOM. Supports train/val/test splitting
and PyTorch DataLoader integration for batched iteration.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class DataFormat(str, Enum):
    """Supported medical data formats."""

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    DICOM = "dicom"
    AUTO = "auto"


@dataclass
class LoaderConfig:
    """Configuration for the DatasetLoader.

    Attributes:
        default_batch_size: Default batch size for created DataLoaders.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for reproducible splitting.
        pin_memory: Whether to pin memory for GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        dicom_pixel_data_tag: DICOM tag name for extracting pixel arrays.
        image_extensions: Recognised image file extensions.
        max_file_size_mb: Maximum individual file size in megabytes.
    """

    default_batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42
    pin_memory: bool = True
    drop_last: bool = False
    dicom_pixel_data_tag: str = "PixelData"
    image_extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".dcm"]
    )
    max_file_size_mb: float = 512.0


@dataclass
class SplitRatios:
    """Train / validation / test split ratios.

    The three values must sum to 1.0.
    """

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------

class _TabularDataset(Dataset):
    """Thin wrapper that serves rows of a DataFrame as tensors."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._data = dataframe.reset_index(drop=True)
        self._features = self._data.select_dtypes(include=[np.number]).values.astype(np.float32)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._data.iloc[idx]
        feature_vec = torch.from_numpy(self._features[idx])
        return {
            "index": idx,
            "features": feature_vec,
            "label": torch.tensor(row.get("label", 0), dtype=torch.long),
        }


class _ImageArrayDataset(Dataset):
    """Dataset backed by a list of image file paths."""

    def __init__(self, image_paths: list[Path], labels: Optional[list[int]] = None) -> None:
        self._paths = image_paths
        self._labels = labels

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image  # lazy import to avoid hard dependency at module level

        img = Image.open(self._paths[idx]).convert("RGB")
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(arr)
        label = self._labels[idx] if self._labels is not None else -1
        return {"image": tensor, "path": str(self._paths[idx]), "label": torch.tensor(label, dtype=torch.long)}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Unified dataset loader for medical AI workloads.

    Supports loading from CSV, JSON, Parquet, and DICOM formats, automatic
    format detection, flexible train/val/test splitting, and conversion to
    PyTorch ``DataLoader`` objects.

    Args:
        config: A :class:`LoaderConfig` instance.  When ``None`` the default
            configuration is used.

    Example::

        loader = DatasetLoader(LoaderConfig(batch_size=64))
        df = loader.load_dataset("chest_xray.csv", DataFormat.CSV)
        train_dl, val_dl, test_dl = loader.split_dataset(df, SplitRatios(0.7, 0.15, 0.15))
    """

    def __init__(self, config: Optional[LoaderConfig] = None) -> None:
        self._config = config or LoaderConfig()
        self._rng = np.random.default_rng(self._config.seed)
        logger.info("DatasetLoader initialised (seed=%d)", self._config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        path: Union[str, Path],
        fmt: Union[DataFormat, str] = DataFormat.AUTO,
    ) -> pd.DataFrame:
        """Load a dataset from *path* and return a :class:`pandas.DataFrame`.

        Args:
            path: File or directory path.
            fmt: Expected format.  ``DataFormat.AUTO`` triggers extension-based
                detection.

        Returns:
            A DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the format cannot be determined or is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        if isinstance(fmt, str):
            fmt = DataFormat(fmt.lower())

        if fmt == DataFormat.AUTO:
            fmt = self._detect_format(path)

        logger.info("Loading dataset from %s (format=%s)", path, fmt.value)

        loaders = {
            DataFormat.CSV: self._load_tabular,
            DataFormat.JSON: self._load_json,
            DataFormat.PARQUET: self._load_parquet,
            DataFormat.DICOM: self._load_dicom_dataset,
        }

        loader_fn = loaders.get(fmt)
        if loader_fn is None:
            raise ValueError(f"Unsupported format: {fmt}")

        df = loader_fn(path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def split_dataset(
        self,
        df: pd.DataFrame,
        ratios: Union[SplitRatios, tuple[float, float, float], None] = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Split a DataFrame into train / val / test DataLoaders.

        Args:
            df: The full dataset.
            ratios: A :class:`SplitRatios` or 3-tuple ``(train, val, test)``.
                Defaults to 80 / 10 / 10.

        Returns:
            A tuple ``(train_loader, val_loader, test_loader)``.
        """
        if ratios is None:
            ratios = SplitRatios()
        elif isinstance(ratios, tuple):
            ratios = SplitRatios(*ratios)

        n = len(df)
        indices = np.arange(n)
        self._rng.shuffle(indices)

        train_end = int(n * ratios.train)
        val_end = train_end + int(n * ratios.val)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_ds = _TabularDataset(df.iloc[train_idx])
        val_ds = _TabularDataset(df.iloc[val_idx])
        test_ds = _TabularDataset(df.iloc[test_idx])

        train_dl = self.get_data_loader(train_ds, self._config.default_batch_size, shuffle=True)
        val_dl = self.get_data_loader(val_ds, self._config.default_batch_size, shuffle=False)
        test_dl = self.get_data_loader(test_ds, self._config.default_batch_size, shuffle=False)

        logger.info(
            "Split dataset: train=%d, val=%d, test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )
        return train_dl, val_dl, test_dl

    def get_data_loader(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        """Create a :class:`torch.utils.data.DataLoader` from a Dataset.

        Args:
            dataset: Any PyTorch ``Dataset``.
            batch_size: Overrides the config default when provided.
            shuffle: Overrides the config default when provided.

        Returns:
            A configured ``DataLoader``.
        """
        bs = batch_size or self._config.default_batch_size
        sh = shuffle if shuffle is not None else self._config.shuffle

        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=sh,
            num_workers=self._config.num_workers,
            pin_memory=self._config.pin_memory,
            drop_last=self._config.drop_last,
        )

    # ------------------------------------------------------------------
    # Image & text directory loaders
    # ------------------------------------------------------------------

    def _load_images(self, dir_path: Path) -> pd.DataFrame:
        """Recursively load image file paths from *dir_path*.

        Returns:
            A DataFrame with columns ``["path", "filename", "extension"]``.
        """
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Expected a directory: {dir_path}")

        exts = set(self._config.image_extensions)
        records: list[dict[str, str]] = []

        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in exts:
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    if size_mb > self._config.max_file_size_mb:
                        logger.warning("Skipping large file (%.1f MB): %s", size_mb, fpath)
                        continue
                    records.append({
                        "path": str(fpath),
                        "filename": fpath.name,
                        "extension": fpath.suffix.lower(),
                    })

        if not records:
            logger.warning("No image files found under %s", dir_path)
            return pd.DataFrame(columns=["path", "filename", "extension"])

        df = pd.DataFrame(records)
        logger.info("Discovered %d image files under %s", len(df), dir_path)
        return df

    def _load_text(self, path: Path) -> pd.DataFrame:
        """Load a plain-text or JSON-Lines file into a DataFrame.

        Supports:
        - ``.txt`` (one document per line)
        - ``.jsonl`` / ``.json`` (JSON-Lines, each line is a JSON object)

        Returns:
            A DataFrame with at least a ``"text"`` column.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Text file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".jsonl" or suffix == ".json":
            records: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON at line %d", line_no)
            return pd.DataFrame(records)

        # Fallback: plain text, one document per line
        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        return pd.DataFrame({"text": lines})

    # ------------------------------------------------------------------
    # Format-specific loaders (private)
    # ------------------------------------------------------------------

    def _load_tabular(self, path: Path) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        return pd.read_csv(path)

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load a JSON file (object or array) into a DataFrame."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            # Try normalising if it looks like a records-style dict
            return pd.DataFrame([data])
        raise ValueError(f"Unrecognised JSON structure in {path}")

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load a Parquet file into a DataFrame."""
        return pd.read_parquet(path)

    def _load_dicom_dataset(self, path: Path) -> pd.DataFrame:
        """Load DICOM files from a directory into a DataFrame.

        Each DICOM file is parsed and its metadata fields are extracted into
        columns.  Pixel data is stored as a base64-encoded string so it can
        be round-tripped through a DataFrame.

        Raises:
            ImportError: If the ``pydicom`` package is not installed.
        """
        try:
            import pydicom  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pydicom is required for DICOM loading. "
                "Install it with: pip install pydicom"
            ) from exc

        if path.is_file() and path.suffix.lower() == ".dcm":
            dcm_paths = [path]
        elif path.is_dir():
            dcm_paths = sorted(path.rglob("*.dcm"))
        else:
            raise ValueError(f"Expected a .dcm file or directory: {path}")

        records: list[dict[str, Any]] = []
        for dcm_path in dcm_paths:
            try:
                ds = pydicom.dcmread(str(dcm_path))
                record: dict[str, Any] = {"_source_path": str(dcm_path)}
                for elem in ds.iterall():
                    if elem.keyword:
                        record[elem.keyword] = str(elem.value) if not isinstance(elem.value, (int, float)) else elem.value
                records.append(record)
            except Exception:
                logger.warning("Failed to parse DICOM file: %s", dcm_path)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(path: Path) -> DataFormat:
        """Detect file format from the path extension.

        Args:
            path: File or directory path.

        Returns:
            The detected :class:`DataFormat`.

        Raises:
            ValueError: If the format cannot be inferred.
        """
        if path.is_dir():
            # Check if directory contains DICOM files
            if any(path.rglob("*.dcm")):
                return DataFormat.DICOM
            raise ValueError(
                f"Cannot auto-detect format for directory: {path}. "
                "Please specify the format explicitly."
            )

        ext = path.suffix.lower()
        mapping = {
            ".csv": DataFormat.CSV,
            ".tsv": DataFormat.CSV,
            ".json": DataFormat.JSON,
            ".jsonl": DataFormat.JSON,
            ".parquet": DataFormat.PARQUET,
            ".pq": DataFormat.PARQUET,
            ".dcm": DataFormat.DICOM,
            ".dicom": DataFormat.DICOM,
        }

        fmt = mapping.get(ext)
        if fmt is None:
            raise ValueError(
                f"Cannot auto-detect format from extension '{ext}'. "
                f"Supported extensions: {sorted(mapping.keys())}"
            )
        return fmt
