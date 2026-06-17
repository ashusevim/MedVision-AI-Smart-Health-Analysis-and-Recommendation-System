"""
Data Validation Module for MedVision-AI.

Provides comprehensive data validation for medical datasets including
image quality checks, text validation, clinical data verification,
HIPAA PHI compliance scanning, and detailed reporting.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes & enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""

    IMAGE = "image"
    TEXT = "text"
    CLINICAL = "clinical"
    PHI = "phi"
    SCHEMA = "schema"
    INTEGRITY = "integrity"


@dataclass
class ValidationIssue:
    """A single validation finding.

    Attributes:
        category: The check category.
        severity: Issue severity.
        message: Human-readable description.
        field_name: Optional field / column name.
        value: Optional offending value.
        suggestion: Optional remediation suggestion.
    """

    category: ValidationCategory
    severity: Severity
    message: str
    field_name: Optional[str] = None
    value: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "field_name": self.field_name,
            "value": str(self.value) if self.value is not None else None,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Aggregate result of a validation run.

    Attributes:
        is_valid: ``True`` when no error- or critical-severity issues found.
        issues: List of :class:`ValidationIssue` instances.
        timestamp: When the validation was performed.
        record_count: Number of records examined.
    """

    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    record_count: int = 0

    @property
    def error_count(self) -> int:
        """Number of error-level issues."""
        return sum(1 for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL))

    @property
    def warning_count(self) -> int:
        """Number of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue and update :attr:`is_valid` if necessary."""
        self.issues.append(issue)
        if issue.severity in (Severity.ERROR, Severity.CRITICAL):
            self.is_valid = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "timestamp": self.timestamp,
            "record_count": self.record_count,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# PHI patterns (HIPAA Safe-Harbour identifiers)
# ---------------------------------------------------------------------------

_PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "date_of_birth": re.compile(
        r"\b(0[1-9]|1[0-2])[/\-](0[1-9]|[12]\d|3[01])[/\-](19|20)\d{2}\b"
    ),
    "mrn": re.compile(r"\bMRN[\s:]*\d{4,12}\b", re.IGNORECASE),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "zip_code": re.compile(r"\b\d{5}(-\d{4})?\b"),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidatorConfig:
    """Configuration for the DataValidator.

    Attributes:
        max_image_size_mb: Maximum allowed image file size in MB.
        min_image_resolution: Minimum (width, height) in pixels.
        max_text_length: Maximum allowed text field length.
        min_text_length: Minimum allowed text field length.
        phi_check_enabled: Whether to scan for PHI.
        required_clinical_fields: Fields that must be present in clinical records.
        allowed_image_modes: Accepted PIL image modes.
        max_pixel_value: Maximum expected pixel value for images.
        min_pixel_value: Minimum expected pixel value for images.
        null_value_patterns: Strings that represent null / missing values.
    """

    max_image_size_mb: float = 100.0
    min_image_resolution: tuple[int, int] = (32, 32)
    max_text_length: int = 100_000
    min_text_length: int = 1
    phi_check_enabled: bool = True
    required_clinical_fields: list[str] = field(
        default_factory=lambda: ["patient_id", "timestamp"]
    )
    allowed_image_modes: list[str] = field(
        default_factory=lambda: ["L", "RGB", "RGBA", "I;16"]
    )
    max_pixel_value: float = 255.0
    min_pixel_value: float = 0.0
    null_value_patterns: list[str] = field(
        default_factory=lambda: ["", "NULL", "null", "N/A", "n/a", "NaN", "None", "none", "-"]
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataValidator:
    """Comprehensive data validator for medical AI workloads.

    Validates images, text, and clinical data for quality, completeness,
    and HIPAA compliance.  Produces structured :class:`ValidationResult`
    objects that can be aggregated into reports.

    Args:
        config: A :class:`ValidatorConfig` instance.

    Example::

        validator = DataValidator(ValidatorConfig(phi_check_enabled=True))
        result = validator.validate(clinical_dataframe)
        if not result.is_valid:
            print(result.error_count, "issues found")
    """

    def __init__(self, config: Optional[ValidatorConfig] = None) -> None:
        self._config = config or ValidatorConfig()
        self._results: list[ValidationResult] = []
        logger.info("DataValidator initialised (phi_check=%s)", self._config.phi_check_enabled)

    # ------------------------------------------------------------------
    # Top-level validate
    # ------------------------------------------------------------------

    def validate(
        self,
        data: Union[pd.DataFrame, dict[str, Any], list[Any]],
    ) -> ValidationResult:
        """Validate a dataset and return a :class:`ValidationResult`.

        Dispatches to the appropriate sub-validator based on the type
        and contents of *data*.

        Args:
            data: A DataFrame, dict, or list to validate.

        Returns:
            A structured validation result.
        """
        result = ValidationResult(record_count=self._count_records(data))

        if isinstance(data, pd.DataFrame):
            self._validate_dataframe(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    self._validate_dict(item, result, prefix=f"[{idx}]")
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=Severity.ERROR,
                message=f"Unsupported data type: {type(data).__name__}",
                suggestion="Provide a DataFrame, dict, or list of dicts.",
            ))

        self._results.append(result)
        logger.info(
            "Validation complete: valid=%s, errors=%d, warnings=%d",
            result.is_valid, result.error_count, result.warning_count,
        )
        return result

    # ------------------------------------------------------------------
    # Image validation
    # ------------------------------------------------------------------

    def validate_image(
        self,
        image: Union[np.ndarray, "PIL.Image.Image", Path, str],
    ) -> ValidationResult:
        """Validate a single medical image.

        Checks dimensions, pixel range, mode, and basic quality metrics.

        Args:
            image: Image as numpy array, PIL Image, or file path.

        Returns:
            Validation result for this image.
        """
        result = ValidationResult(record_count=1)

        # Load image if a path is given
        img_array: Optional[np.ndarray] = None
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE,
                    severity=Severity.ERROR,
                    message=f"Image file not found: {path}",
                    field_name="path",
                    suggestion="Verify the file path.",
                ))
                return result

            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self._config.max_image_size_mb:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE,
                    severity=Severity.WARNING,
                    message=f"Image file size ({size_mb:.1f} MB) exceeds limit ({self._config.max_image_size_mb} MB)",
                    field_name="file_size",
                    suggestion="Consider compressing or resizing the image.",
                ))

            try:
                from PIL import Image  # type: ignore[import-untyped]
                pil_img = Image.open(path)
                if pil_img.mode not in self._config.allowed_image_modes:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.IMAGE,
                        severity=Severity.WARNING,
                        message=f"Unexpected image mode: {pil_img.mode}",
                        field_name="mode",
                        suggestion=f"Expected one of {self._config.allowed_image_modes}.",
                    ))
                img_array = np.array(pil_img)
            except Exception as exc:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE,
                    severity=Severity.ERROR,
                    message=f"Failed to open image: {exc}",
                    field_name="path",
                ))
                return result
        elif hasattr(image, "mode"):
            # PIL Image
            img_array = np.array(image)
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.ERROR,
                message=f"Unsupported image type: {type(image).__name__}",
                suggestion="Provide a numpy array, PIL Image, or file path.",
            ))
            return result

        if img_array is not None:
            self._check_image_array(img_array, result)

        self._results.append(result)
        return result

    def _check_image_array(self, img: np.ndarray, result: ValidationResult) -> None:
        """Run quality checks on a numpy image array."""
        # Dimension check
        if img.ndim < 2:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.ERROR,
                message=f"Image has fewer than 2 dimensions (shape={img.shape})",
                field_name="shape",
            ))
            return

        h, w = img.shape[:2]
        min_w, min_h = self._config.min_image_resolution
        if w < min_w or h < min_h:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.WARNING,
                message=f"Image resolution ({w}x{h}) below minimum ({min_w}x{min_h})",
                field_name="resolution",
                suggestion="Upsample or exclude this image from the dataset.",
            ))

        # Pixel value range
        img_float = img.astype(np.float64)
        if img_float.min() < self._config.min_pixel_value:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.WARNING,
                message=f"Pixel values below expected minimum ({self._config.min_pixel_value})",
                field_name="pixel_min",
                value=img_float.min(),
            ))

        if img_float.max() > self._config.max_pixel_value:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.WARNING,
                message=f"Pixel values above expected maximum ({self._config.max_pixel_value})",
                field_name="pixel_max",
                value=img_float.max(),
            ))

        # Uniform image check (blank / constant)
        if img_float.std() < 1e-6:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE,
                severity=Severity.ERROR,
                message="Image appears to be blank (constant pixel values)",
                field_name="std",
                suggestion="Check the image source for acquisition errors.",
            ))

    # ------------------------------------------------------------------
    # Text validation
    # ------------------------------------------------------------------

    def validate_text(self, text: str) -> ValidationResult:
        """Validate a single text field.

        Checks length, encoding issues, null patterns, and optionally
        scans for PHI.

        Args:
            text: The text string to validate.

        Returns:
            Validation result for this text.
        """
        result = ValidationResult(record_count=1)

        if not isinstance(text, str):
            result.add_issue(ValidationIssue(
                category=ValidationCategory.TEXT,
                severity=Severity.ERROR,
                message=f"Expected str, got {type(text).__name__}",
                field_name="text_type",
            ))
            self._results.append(result)
            return result

        # Length checks
        if len(text) < self._config.min_text_length:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.TEXT,
                severity=Severity.WARNING,
                message=f"Text length ({len(text)}) below minimum ({self._config.min_text_length})",
                field_name="length",
            ))

        if len(text) > self._config.max_text_length:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.TEXT,
                severity=Severity.WARNING,
                message=f"Text length ({len(text)}) exceeds maximum ({self._config.max_text_length})",
                field_name="length",
                suggestion="Consider truncating or splitting the text.",
            ))

        # Null-value patterns
        if text.strip() in self._config.null_value_patterns:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.TEXT,
                severity=Severity.WARNING,
                message="Text field appears to be a null placeholder",
                field_name="null_pattern",
                value=text,
            ))

        # Encoding issues
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.TEXT,
                severity=Severity.WARNING,
                message="Text contains non-UTF-8 encodable characters",
                field_name="encoding",
                suggestion="Sanitise or re-encode the text.",
            ))

        # PHI scan
        if self._config.phi_check_enabled:
            phi_result = self.check_phi_compliance(text)
            result.issues.extend(phi_result.issues)
            if not phi_result.is_valid:
                result.is_valid = False

        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Clinical data validation
    # ------------------------------------------------------------------

    def validate_clinical_data(
        self,
        record: dict[str, Any],
    ) -> ValidationResult:
        """Validate a single clinical record.

        Checks for required fields, type consistency, value ranges,
        and PHI compliance.

        Args:
            record: A dictionary representing a clinical record.

        Returns:
            Validation result for this record.
        """
        result = ValidationResult(record_count=1)

        # Required fields
        for req_field in self._config.required_clinical_fields:
            if req_field not in record or record[req_field] is None:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CLINICAL,
                    severity=Severity.ERROR,
                    message=f"Required field '{req_field}' is missing or null",
                    field_name=req_field,
                    suggestion=f"Provide a value for '{req_field}'.",
                ))

        # Type checks for common clinical fields
        type_checks: dict[str, tuple[type, ...]] = {
            "patient_id": (str, int),
            "age": (int, float),
            "weight": (int, float),
            "height": (int, float),
            "heart_rate": (int, float),
            "blood_pressure_systolic": (int, float),
            "blood_pressure_diastolic": (int, float),
            "temperature": (int, float),
        }
        for col, expected_types in type_checks.items():
            if col in record and record[col] is not None:
                if not isinstance(record[col], expected_types):
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.CLINICAL,
                        severity=Severity.WARNING,
                        message=f"Field '{col}' has unexpected type {type(record[col]).__name__}",
                        field_name=col,
                        suggestion=f"Expected one of {expected_types}.",
                    ))

        # Value-range checks for vital signs
        range_checks: dict[str, tuple[float, float]] = {
            "age": (0, 150),
            "heart_rate": (0, 300),
            "blood_pressure_systolic": (0, 300),
            "blood_pressure_diastolic": (0, 200),
            "temperature": (30, 45),
            "weight": (0, 500),
            "height": (0, 300),
            "oxygen_saturation": (0, 100),
        }
        for col, (lo, hi) in range_checks.items():
            if col in record and record[col] is not None:
                try:
                    val = float(record[col])
                    if val < lo or val > hi:
                        result.add_issue(ValidationIssue(
                            category=ValidationCategory.CLINICAL,
                            severity=Severity.WARNING,
                            message=f"Field '{col}' value ({val}) outside expected range [{lo}, {hi}]",
                            field_name=col,
                            value=val,
                            suggestion="Verify the recorded value.",
                        ))
                except (TypeError, ValueError):
                    pass  # type check already caught this

        # PHI scan on all string fields
        if self._config.phi_check_enabled:
            for key, value in record.items():
                if isinstance(value, str):
                    phi_result = self.check_phi_compliance(value)
                    for issue in phi_result.issues:
                        issue.field_name = key
                    result.issues.extend(phi_result.issues)
                    if not phi_result.is_valid:
                        result.is_valid = False

        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # PHI compliance
    # ------------------------------------------------------------------

    def check_phi_compliance(
        self,
        data: Union[str, pd.DataFrame, dict[str, Any]],
    ) -> ValidationResult:
        """Scan data for Protected Health Information (PHI).

        Detects common HIPAA identifiers such as SSNs, phone numbers,
        email addresses, and MRNs.

        Args:
            data: Text string, DataFrame, or dict to scan.

        Returns:
            Validation result with PHI findings.
        """
        result = ValidationResult()

        if isinstance(data, str):
            self._scan_text_for_phi(data, result)
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                for idx, val in enumerate(data[col]):
                    if isinstance(val, str):
                        sub_result = ValidationResult()
                        self._scan_text_for_phi(val, sub_result)
                        for issue in sub_result.issues:
                            issue.field_name = f"{col}[{idx}]"
                        result.issues.extend(sub_result.issues)
                        if not sub_result.is_valid:
                            result.is_valid = False
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, str):
                    sub_result = ValidationResult()
                    self._scan_text_for_phi(val, sub_result)
                    for issue in sub_result.issues:
                        issue.field_name = key
                    result.issues.extend(sub_result.issues)
                    if not sub_result.is_valid:
                        result.is_valid = False

        return result

    def _scan_text_for_phi(self, text: str, result: ValidationResult) -> None:
        """Scan a single text string for PHI patterns."""
        for phi_type, pattern in _PHI_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.PHI,
                    severity=Severity.CRITICAL,
                    message=f"Potential {phi_type.upper()} detected ({len(matches)} occurrence(s))",
                    value=matches[0] if isinstance(matches[0], str) else matches[0],
                    suggestion=f"Anonymise or remove {phi_type} data before processing.",
                ))

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_validation_report(self) -> dict[str, Any]:
        """Generate a comprehensive validation report across all runs.

        Returns:
            A dictionary containing aggregated statistics, per-category
            breakdowns, and the full list of issues.
        """
        if not self._results:
            return {"status": "no_validations_run", "issues": []}

        total_records = sum(r.record_count for r in self._results)
        total_errors = sum(r.error_count for r in self._results)
        total_warnings = sum(r.warning_count for r in self._results)

        # Per-category breakdown
        category_counts: dict[str, int] = {}
        for result in self._results:
            for issue in result.issues:
                cat = issue.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

        all_issues: list[dict[str, Any]] = []
        for result in self._results:
            all_issues.extend(issue.to_dict() for issue in result.issues)

        report = {
            "status": "pass" if total_errors == 0 else "fail",
            "total_validations": len(self._results),
            "total_records_examined": total_records,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "category_breakdown": category_counts,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": all_issues,
        }

        logger.info(
            "Validation report generated: status=%s, errors=%d, warnings=%d",
            report["status"], total_errors, total_warnings,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate a DataFrame by delegating to sub-validators."""
        # Null column check
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.WARNING,
                    message=f"Column '{col}' has {null_count} null values",
                    field_name=col,
                    suggestion="Consider imputing or dropping null values.",
                ))

        # PHI scan on string columns
        if self._config.phi_check_enabled:
            string_cols = df.select_dtypes(include=["object"]).columns
            for col in string_cols:
                sample = df[col].dropna().head(100).astype(str)
                for val in sample:
                    phi_result = self.check_phi_compliance(val)
                    for issue in phi_result.issues:
                        issue.field_name = col
                    result.issues.extend(phi_result.issues)
                    if not phi_result.is_valid:
                        result.is_valid = False

    def _validate_dict(
        self,
        data: dict[str, Any],
        result: ValidationResult,
        prefix: str = "",
    ) -> None:
        """Validate a dict as a clinical record."""
        sub_result = self.validate_clinical_data(data)
        for issue in sub_result.issues:
            if prefix:
                issue.message = f"{prefix} {issue.message}"
        result.issues.extend(sub_result.issues)
        if not sub_result.is_valid:
            result.is_valid = False

    @staticmethod
    def _count_records(data: Any) -> int:
        """Estimate the number of records in *data*."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return 1
        return 0
