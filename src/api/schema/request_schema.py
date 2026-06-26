"""
Request Schemas for MedVision-AI API.

Defines Pydantic models for all incoming request payloads including
image predictions, symptom analysis, multimodal fusion, risk scoring,
and report generation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.patient_schema import PatientDemographics


# -----------------------------------------------------------------------
# Image Prediction
# -----------------------------------------------------------------------

class ImagePredictionRequest(BaseModel):
    """Request payload for image-based medical prediction.

    Attributes:
        image_data: Base64-encoded medical image (PNG, JPEG, or DICOM).
        image_type: Type of medical image for pre-processing hints.
        patient_id: Optional patient identifier for record linking.
        demographics: Optional patient demographics for context.
        include_explainability: Whether to generate an explainability map.
        priority: Request priority (``"normal"`` | ``"high"`` | ``"critical"``).
    """

    image_data: str = Field(
        ...,
        description="Base64-encoded medical image data.",
        min_length=1,
    )
    image_type: str = Field(
        default="x_ray",
        description="Type of medical image: x_ray, ct, mri, ultrasound, pathology.",
    )
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient identifier for record linking.",
    )
    demographics: Optional[PatientDemographics] = Field(
        default=None,
        description="Optional patient demographics for contextual analysis.",
    )
    include_explainability: bool = Field(
        default=False,
        description="Whether to generate an explainability / attention heatmap.",
    )
    priority: str = Field(
        default="normal",
        description="Request priority level: normal, high, critical.",
    )

    @field_validator("image_type")
    @classmethod
    def validate_image_type(cls, v: str) -> str:
        """Ensure the image type is one of the supported modalities."""
        allowed = {"x_ray", "ct", "mri", "ultrasound", "pathology", "dermoscopy"}
        if v.lower() not in allowed:
            raise ValueError(f"image_type must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Ensure the priority is a recognised level."""
        allowed = {"normal", "high", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"priority must be one of {allowed}, got '{v}'")
        return v.lower()

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "image_data": "iVBORw0KGgoAAAANSUhEUg...",
                "image_type": "x_ray",
                "patient_id": "P-12345",
                "include_explainability": True,
                "priority": "normal",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Symptom Prediction
# -----------------------------------------------------------------------

class SymptomPredictionRequest(BaseModel):
    """Request payload for symptom-based condition prediction.

    Attributes:
        symptoms: List of reported symptom descriptions.
        demographics: Patient demographics for risk adjustment.
        vital_signs: Optional vital sign measurements.
        duration_days: How long the symptoms have been present.
        severity: Self-reported severity on a 1–10 scale.
    """

    symptoms: list[str] = Field(
        ...,
        description="List of reported symptom descriptions.",
        min_length=1,
    )
    demographics: PatientDemographics = Field(
        ...,
        description="Patient demographics (required for risk adjustment).",
    )
    vital_signs: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional vital sign measurements (e.g., heart_rate, blood_pressure).",
    )
    duration_days: Optional[int] = Field(
        default=None,
        description="Number of days symptoms have been present.",
        ge=0,
    )
    severity: Optional[int] = Field(
        default=None,
        description="Self-reported severity on a 1–10 scale.",
        ge=1,
        le=10,
    )

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: list[str]) -> list[str]:
        """Ensure at least one non-empty symptom is provided."""
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty symptom is required.")
        return cleaned

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "symptoms": ["chest pain", "shortness of breath", "fatigue"],
                "demographics": {"age": 55, "sex": "male"},
                "vital_signs": {"heart_rate": 95, "blood_pressure_systolic": 140},
                "duration_days": 3,
                "severity": 7,
            }
        ]
    }}


# -----------------------------------------------------------------------
# Multimodal Prediction
# -----------------------------------------------------------------------

class MultimodalPredictionRequest(BaseModel):
    """Request payload for multimodal (image + symptoms) prediction.

    Combines medical imaging data with patient-reported symptoms and
    demographics for a fused, higher-accuracy diagnostic prediction.

    Attributes:
        image_data: Base64-encoded medical image.
        image_type: Type of medical image.
        symptoms: List of reported symptom descriptions.
        demographics: Patient demographics.
        vital_signs: Optional vital sign measurements.
        fusion_strategy: Strategy for combining modalities.
    """

    image_data: str = Field(
        ...,
        description="Base64-encoded medical image data.",
        min_length=1,
    )
    image_type: str = Field(
        default="x_ray",
        description="Type of medical image.",
    )
    symptoms: list[str] = Field(
        ...,
        description="List of reported symptom descriptions.",
        min_length=1,
    )
    demographics: PatientDemographics = Field(
        ...,
        description="Patient demographics.",
    )
    vital_signs: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional vital sign measurements.",
    )
    fusion_strategy: str = Field(
        default="late",
        description="Fusion strategy: 'early', 'late', or 'attention'.",
    )

    @field_validator("fusion_strategy")
    @classmethod
    def validate_fusion_strategy(cls, v: str) -> str:
        """Ensure the fusion strategy is recognised."""
        allowed = {"early", "late", "attention"}
        if v.lower() not in allowed:
            raise ValueError(f"fusion_strategy must be one of {allowed}, got '{v}'")
        return v.lower()

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "image_data": "iVBORw0KGgoAAAANSUhEUg...",
                "image_type": "ct",
                "symptoms": ["persistent cough", "weight loss"],
                "demographics": {"age": 62, "sex": "female"},
                "fusion_strategy": "late",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Risk Prediction
# -----------------------------------------------------------------------

class RiskPredictionRequest(BaseModel):
    """Request payload for risk score prediction.

    Evaluates a patient's risk for specified conditions based on
    demographics, medical history, lifestyle, and family history.

    Attributes:
        demographics: Patient demographics.
        medical_history: List of past diagnoses and conditions.
        lifestyle_factors: Lifestyle data such as smoking, BMI, exercise.
        family_history: Family history of relevant conditions.
        target_conditions: Conditions to assess risk for.
        assessment_type: Type of risk assessment.
    """

    demographics: PatientDemographics = Field(
        ...,
        description="Patient demographics.",
    )
    medical_history: list[str] = Field(
        default_factory=list,
        description="List of past diagnoses and conditions.",
    )
    lifestyle_factors: dict[str, Any] = Field(
        default_factory=dict,
        description="Lifestyle data (smoking, alcohol, exercise, BMI, etc.).",
    )
    family_history: list[str] = Field(
        default_factory=list,
        description="Family history of relevant conditions.",
    )
    target_conditions: list[str] = Field(
        ...,
        description="Conditions to assess risk for.",
        min_length=1,
    )
    assessment_type: str = Field(
        default="comprehensive",
        description="Assessment type: 'screening', 'comprehensive', or 'targeted'.",
    )

    @field_validator("assessment_type")
    @classmethod
    def validate_assessment_type(cls, v: str) -> str:
        """Ensure the assessment type is recognised."""
        allowed = {"screening", "comprehensive", "targeted"}
        if v.lower() not in allowed:
            raise ValueError(f"assessment_type must be one of {allowed}, got '{v}'")
        return v.lower()

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "demographics": {"age": 50, "sex": "male"},
                "medical_history": ["hypertension", "type_2_diabetes"],
                "lifestyle_factors": {
                    "smoking": True,
                    "alcohol_units_per_week": 14,
                    "exercise_hours_per_week": 1,
                    "bmi": 28.5,
                },
                "family_history": ["coronary_heart_disease"],
                "target_conditions": ["cardiovascular_disease", "stroke"],
                "assessment_type": "comprehensive",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------

class ReportGenerateRequest(BaseModel):
    """Request payload for diagnostic report generation.

    Attributes:
        patient_id: Patient identifier.
        prediction_ids: List of prediction IDs to include in the report.
        report_type: Type of report to generate.
        include_images: Whether to embed image references in the report.
        additional_notes: Optional free-text notes for the clinician.
        format: Desired output format for the report.
    """

    patient_id: str = Field(
        ...,
        description="Patient identifier.",
        min_length=1,
    )
    prediction_ids: list[str] = Field(
        ...,
        description="List of prediction IDs to include in the report.",
        min_length=1,
    )
    report_type: str = Field(
        default="diagnostic",
        description="Type of report: 'diagnostic', 'screening', or 'follow_up'.",
    )
    include_images: bool = Field(
        default=True,
        description="Whether to include image references in the report.",
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Optional free-text notes for the clinician.",
    )
    format: str = Field(
        default="json",
        description="Desired output format: 'json', 'pdf', or 'text'.",
    )

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Ensure the report type is recognised."""
        allowed = {"diagnostic", "screening", "follow_up"}
        if v.lower() not in allowed:
            raise ValueError(f"report_type must be one of {allowed}, got '{v}'")
        return v.lower()

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "patient_id": "P-12345",
                "prediction_ids": ["a1b2c3d4", "e5f6g7h8"],
                "report_type": "diagnostic",
                "include_images": True,
                "format": "json",
            }
        ]
    }}
