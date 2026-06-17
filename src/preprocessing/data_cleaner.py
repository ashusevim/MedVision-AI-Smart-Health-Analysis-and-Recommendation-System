"""
Data Cleaner Module for MedVision-AI.

Provides comprehensive data cleaning capabilities for medical datasets,
including missing value imputation, duplicate removal, schema validation,
column normalization, and outlier detection.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & enums
# ---------------------------------------------------------------------------

class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    DROP = "drop"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    CONSTANT = "constant"


class NormalizationMethod(str, Enum):
    """Column normalization methods."""

    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"
    LOG = "log"


class OutlierMethod(str, Enum):
    """Outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"


@dataclass
class CleanerConfig:
    """Configuration for the DataCleaner.

    Attributes:
        default_missing_strategy: Default strategy for missing value imputation.
        outlier_threshold: Threshold multiplier for outlier detection.
        normalization_method: Default normalization method.
        drop_columns: Columns to drop before cleaning.
        constant_fill_value: Value used when strategy is CONSTANT.
        max_missing_ratio: Columns with a missing ratio above this are dropped.
        outlier_action: What to do with detected outliers (``"clip"`` or ``"remove"``).
        iqr_multiplier: Multiplier for IQR-based outlier bounds.
        zscore_threshold: Absolute z-score threshold for outlier detection.
    """

    default_missing_strategy: MissingValueStrategy = MissingValueStrategy.MEAN
    outlier_threshold: float = 3.0
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE
    drop_columns: list[str] = field(default_factory=list)
    constant_fill_value: Any = 0
    max_missing_ratio: float = 0.8
    outlier_action: str = "clip"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0


@dataclass
class CleaningReport:
    """Summary report of cleaning operations applied.

    Attributes:
        rows_before: Number of rows before cleaning.
        rows_after: Number of rows after cleaning.
        cols_before: Number of columns before cleaning.
        cols_after: Number of columns after cleaning.
        missing_imputed: Count of missing values imputed.
        duplicates_removed: Count of duplicate rows removed.
        outliers_handled: Count of outlier values handled.
        columns_dropped: List of columns that were dropped.
        columns_normalized: List of columns that were normalized.
    """

    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    missing_imputed: int = 0
    duplicates_removed: int = 0
    outliers_handled: int = 0
    columns_dropped: list[str] = field(default_factory=list)
    columns_normalized: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"Cleaning Report\n"
            f"{'=' * 40}\n"
            f"Rows:     {self.rows_before} -> {self.rows_after} "
            f"({self.rows_before - self.rows_after} removed)\n"
            f"Columns:  {self.cols_before} -> {self.cols_after} "
            f"({self.cols_before - self.cols_after} dropped)\n"
            f"Missing values imputed:  {self.missing_imputed}\n"
            f"Duplicates removed:      {self.duplicates_removed}\n"
            f"Outliers handled:        {self.outliers_handled}\n"
            f"Columns dropped:         {self.columns_dropped}\n"
            f"Columns normalized:      {self.columns_normalized}\n"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataCleaner:
    """Comprehensive data cleaner for medical datasets.

    Provides a pipeline of cleaning operations that can be applied
    sequentially or individually to tabular medical data.

    Args:
        config: A :class:`CleanerConfig` instance. Uses defaults when ``None``.

    Example::

        cleaner = DataCleaner(CleanerConfig(default_missing_strategy="median"))
        cleaned_df = cleaner.clean(raw_df)
        report = cleaner.report
    """

    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self._config = config or CleanerConfig()
        self._report = CleaningReport()
        self._normalization_params: dict[str, dict[str, float]] = {}
        logger.info("DataCleaner initialised (strategy=%s)", self._config.default_missing_strategy.value)

    @property
    def report(self) -> CleaningReport:
        """Return the latest :class:`CleaningReport`."""
        return self._report

    @property
    def normalization_params(self) -> dict[str, dict[str, float]]:
        """Return stored normalization parameters for denormalization."""
        return self._normalization_params

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the full cleaning pipeline on *data*.

        The pipeline consists of:
        1. Drop configured columns
        2. Remove columns exceeding the max-missing ratio
        3. Handle missing values
        4. Remove duplicates
        5. Detect and handle outliers
        6. Normalize numeric columns

        Args:
            data: The raw DataFrame.

        Returns:
            A cleaned copy of the DataFrame.
        """
        df = data.copy()
        self._report = CleaningReport(
            rows_before=len(df),
            cols_before=len(df.columns),
        )

        # 1. Drop configured columns
        cols_to_drop = [c for c in self._config.drop_columns if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self._report.columns_dropped.extend(cols_to_drop)
            logger.info("Dropped columns: %s", cols_to_drop)

        # 2. Drop columns with excessive missing values
        missing_ratio = df.isnull().mean()
        high_missing = missing_ratio[missing_ratio > self._config.max_missing_ratio].index.tolist()
        if high_missing:
            df.drop(columns=high_missing, inplace=True)
            self._report.columns_dropped.extend(high_missing)
            logger.info("Dropped high-missing columns: %s", high_missing)

        # 3. Handle missing values
        df = self.handle_missing_values(strategy=self._config.default_missing_strategy, dataframe=df)

        # 4. Remove duplicates
        df = self.remove_duplicates(dataframe=df)

        # 5. Outlier handling
        df = self.detect_outliers(method=OutlierMethod.IQR, dataframe=df)

        # 6. Normalize
        df = self.normalize_columns(method=self._config.normalization_method, dataframe=df)

        # Finalize report
        self._report.rows_after = len(df)
        self._report.cols_after = len(df.columns)
        logger.info("Cleaning complete: %d -> %d rows", self._report.rows_before, self._report.rows_after)
        return df

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def handle_missing_values(
        self,
        strategy: Union[MissingValueStrategy, str] = MissingValueStrategy.MEAN,
        columns: Optional[list[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            strategy: Imputation strategy.
            columns: Restrict to these columns. ``None`` means all columns.
            dataframe: Operate on this DataFrame.  Falls back to the stored
                internal frame when ``None`` (used during :meth:`clean`).

        Returns:
            DataFrame with missing values handled.
        """
        if isinstance(strategy, str):
            strategy = MissingValueStrategy(strategy.lower())

        df = dataframe if dataframe is not None else pd.DataFrame()
        if df.empty:
            return df

        target_cols = columns or df.columns.tolist()
        total_imputed = 0

        for col in target_cols:
            if col not in df.columns:
                continue
            n_missing = df[col].isnull().sum()
            if n_missing == 0:
                continue

            if strategy == MissingValueStrategy.MEAN and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == MissingValueStrategy.MEDIAN and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == MissingValueStrategy.MODE:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val.iloc[0], inplace=True)
            elif strategy == MissingValueStrategy.DROP:
                df.dropna(subset=[col], inplace=True)
            elif strategy == MissingValueStrategy.FORWARD_FILL:
                df[col].ffill(inplace=True)
            elif strategy == MissingValueStrategy.BACKWARD_FILL:
                df[col].bfill(inplace=True)
            elif strategy == MissingValueStrategy.CONSTANT:
                df[col].fillna(self._config.constant_fill_value, inplace=True)
            else:
                logger.warning("Strategy %s not applicable to non-numeric column '%s'", strategy.value, col)
                continue

            total_imputed += n_missing

        self._report.missing_imputed += total_imputed
        logger.info("Imputed %d missing values (strategy=%s)", total_imputed, strategy.value)
        return df

    def remove_duplicates(
        self,
        subset: Optional[list[str]] = None,
        keep: str = "first",
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame.

        Args:
            subset: Only consider these columns for duplicates.
            keep: Which duplicates to keep (``"first"``, ``"last"``, or ``False``).
            dataframe: Operate on this DataFrame.

        Returns:
            DataFrame with duplicates removed.
        """
        df = dataframe if dataframe is not None else pd.DataFrame()
        if df.empty:
            return df

        before = len(df)
        df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)
        self._report.duplicates_removed += removed
        if removed:
            logger.info("Removed %d duplicate rows", removed)
        return df

    def validate_schema(
        self,
        schema: dict[str, Any],
        dataframe: Optional[pd.DataFrame] = None,
    ) -> list[str]:
        """Validate the DataFrame against a schema specification.

        Args:
            schema: Mapping of column names to expected types or constraints.
                Example::

                    {
                        "patient_id": {"dtype": "int64", "required": True},
                        "age": {"dtype": "float64", "min": 0, "max": 150},
                        "diagnosis": {"dtype": "object", "required": True},
                    }

            dataframe: Operate on this DataFrame.

        Returns:
            A list of validation error strings. An empty list means the data
            passes validation.
        """
        df = dataframe if dataframe is not None else pd.DataFrame()
        errors: list[str] = []

        for col_name, spec in schema.items():
            if spec.get("required", False) and col_name not in df.columns:
                errors.append(f"Required column '{col_name}' is missing.")
                continue

            if col_name not in df.columns:
                continue

            expected_dtype = spec.get("dtype")
            if expected_dtype and str(df[col_name].dtype) != expected_dtype:
                errors.append(
                    f"Column '{col_name}' has dtype '{df[col_name].dtype}', "
                    f"expected '{expected_dtype}'."
                )

            if "min" in spec:
                below = df[col_name] < spec["min"]
                if below.any():
                    errors.append(
                        f"Column '{col_name}' has {below.sum()} values below minimum {spec['min']}."
                    )

            if "max" in spec:
                above = df[col_name] > spec["max"]
                if above.any():
                    errors.append(
                        f"Column '{col_name}' has {above.sum()} values above maximum {spec['max']}."
                    )

            if "allowed_values" in spec:
                invalid = ~df[col_name].isin(spec["allowed_values"])
                if invalid.any():
                    errors.append(
                        f"Column '{col_name}' has {invalid.sum()} values outside allowed set."
                    )

        if errors:
            logger.warning("Schema validation found %d issue(s)", len(errors))
        else:
            logger.info("Schema validation passed")
        return errors

    def normalize_columns(
        self,
        method: Union[NormalizationMethod, str] = NormalizationMethod.ZSCORE,
        columns: Optional[list[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Normalize numeric columns in the DataFrame.

        Args:
            method: Normalization method to apply.
            columns: Restrict to these columns. ``None`` means all numeric.
            dataframe: Operate on this DataFrame.

        Returns:
            DataFrame with normalized columns.
        """
        if isinstance(method, str):
            method = NormalizationMethod(method.lower())

        df = dataframe if dataframe is not None else pd.DataFrame()
        if df.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = columns if columns else numeric_cols
        normalized: list[str] = []

        for col in target_cols:
            if col not in numeric_cols:
                continue

            series = df[col].astype(float)

            if method == NormalizationMethod.MINMAX:
                col_min, col_max = series.min(), series.max()
                if col_max - col_min > 0:
                    df[col] = (series - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.0
                self._normalization_params[col] = {"min": float(col_min), "max": float(col_max), "method": "minmax"}

            elif method == NormalizationMethod.ZSCORE:
                mean, std = series.mean(), series.std()
                if std > 0:
                    df[col] = (series - mean) / std
                else:
                    df[col] = 0.0
                self._normalization_params[col] = {"mean": float(mean), "std": float(std), "method": "zscore"}

            elif method == NormalizationMethod.ROBUST:
                median = series.median()
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df[col] = (series - median) / iqr
                else:
                    df[col] = 0.0
                self._normalization_params[col] = {
                    "median": float(median), "q1": float(q1), "q3": float(q3), "method": "robust"
                }

            elif method == NormalizationMethod.LOG:
                min_val = series.min()
                if min_val <= 0:
                    series = series - min_val + 1  # shift to positive
                df[col] = np.log1p(series)
                self._normalization_params[col] = {"shift": float(max(0, 1 - min_val)), "method": "log"}

            normalized.append(col)

        self._report.columns_normalized.extend(normalized)
        logger.info("Normalized %d columns with %s", len(normalized), method.value)
        return df

    def detect_outliers(
        self,
        method: Union[OutlierMethod, str] = OutlierMethod.IQR,
        columns: Optional[list[str]] = None,
        action: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Detect and optionally handle outliers in numeric columns.

        Args:
            method: Outlier detection method.
            columns: Restrict to these columns. ``None`` means all numeric.
            action: ``"clip"`` to clip outliers, ``"remove"`` to drop rows.
                Falls back to the config default when ``None``.
            dataframe: Operate on this DataFrame.

        Returns:
            DataFrame with outliers handled.
        """
        if isinstance(method, str):
            method = OutlierMethod(method.lower())

        df = dataframe if dataframe is not None else pd.DataFrame()
        if df.empty:
            return df

        action = action or self._config.outlier_action
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = columns if columns else numeric_cols
        total_handled = 0

        for col in target_cols:
            if col not in numeric_cols:
                continue

            if method == OutlierMethod.IQR:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self._config.iqr_multiplier * iqr
                upper = q3 + self._config.iqr_multiplier * iqr
                mask = (df[col] < lower) | (df[col] > upper)

            elif method == OutlierMethod.ZSCORE:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    continue
                z_scores = (df[col] - mean).abs() / std
                mask = z_scores > self._config.zscore_threshold
                lower = mean - self._config.zscore_threshold * std
                upper = mean + self._config.zscore_threshold * std

            elif method == OutlierMethod.MODIFIED_ZSCORE:
                median = df[col].median()
                mad = (df[col] - median).abs().median() * 1.4826  # consistency constant
                if mad == 0:
                    continue
                modified_z = 0.6745 * (df[col] - median) / mad
                mask = modified_z.abs() > self._config.zscore_threshold
                lower = median - (self._config.zscore_threshold / 0.6745) * mad
                upper = median + (self._config.zscore_threshold / 0.6745) * mad
            else:
                continue

            n_outliers = int(mask.sum())
            if n_outliers == 0:
                continue

            total_handled += n_outliers

            if action == "clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
                logger.debug("Clipped %d outliers in '%s'", n_outliers, col)
            elif action == "remove":
                df = df[~mask].reset_index(drop=True)
                logger.debug("Removed %d outlier rows from '%s'", n_outliers, col)

        self._report.outliers_handled += total_handled
        logger.info("Handled %d outliers (method=%s, action=%s)", total_handled, method.value, action)
        return df
