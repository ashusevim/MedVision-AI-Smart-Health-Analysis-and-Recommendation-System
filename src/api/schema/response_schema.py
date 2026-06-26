"""
Response Schemas for MedVision-AI API.

Defines Pydantic models for all outgoing response payloads including
image predictions, symptom analysis, multimodal fusion, risk scoring,
and report retrieval/download.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------
# Shared / common fields
# -----------------------------------------------------------------------

class PredictionMetadata(BaseModel):
    """Metadata attached to every prediction response.

    Attributes:
        model_version: Version of the model used for inference.
        inference_time_ms: Model inference wall-clock time in ms.
        device: Device used for computation (``"cpu"`` or ``"cuda"``).
        timestamp: UNIX timestamp of the prediction.
    """

    model_version: str = Field(default="unknown", description="Model version string.")
    inference_time_ms: float = Field(default=0.0, description="Inference time in milliseconds.")
    device: str = Field(default="cpu", description="Computation device.")
    timestamp: float = Field(default=0.0, description="UNIX timestamp.")


# -----------------------------------------------------------------------
# Image Prediction
# -----------------------------------------------------------------------

class ImagePredictionResponse(BaseModel):
    """Response for image-based medical prediction.

    Attributes:
        request_id: Unique request identifier.
        status: Execution status (``"success"`` | ``"error"`` | ``"degraded"``).
        predictions: Prediction result dictionary with label, confidence, and probabilities.
        latency_ms: End-to-end latency including image decoding.
        model_version: Version of the model used.
    """

    request_id: str = Field(..., description="Unique request identifier.")
    status: str = Field(..., description="Execution status.")
    predictions: dict[str, Any] = Field(
        default_factory=dict,
        description="Prediction result with label, confidence, probabilities, and alternatives.",
    )
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")
    model_version: str = Field(
        default="unknown",
        description="Version of the model used for inference.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "request_id": "a1b2c3d4e5f6",
                "status": "success",
                "predictions": {
                    "label": "pneumonia",
                    "confidence": 0.92,
                    "probabilities": {"pneumonia": 0.92, "normal": 0.06, "effusion": 0.02},
                    "alternatives": [
                        {"label": "normal", "confidence": 0.06},
                        {"label": "effusion", "confidence": 0.02},
                    ],
                },
                "latency_ms": 145.3,
                "model_version": "1.2.0",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Symptom Prediction
# -----------------------------------------------------------------------

class SymptomPredictionResponse(BaseModel):
    """Response for symptom-based condition prediction.

    Attributes:
        request_id: Unique request identifier.
        status: Execution status.
        conditions: Sorted list of predicted conditions with confidences.
        recommended_actions: Clinical action recommendations.
        latency_ms: End-to-end latency in milliseconds.
        model_version: Version of the model used.
    """

    request_id: str = Field(..., description="Unique request identifier.")
    status: str = Field(..., description="Execution status.")
    conditions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Predicted conditions sorted by confidence.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Recommended clinical actions.",
    )
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")
    model_version: str = Field(
        default="unknown",
        description="Version of the model used for inference.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "request_id": "f1e2d3c4b5a6",
                "status": "success",
                "conditions": [
                    {"condition": "acute_bronchitis", "confidence": 0.78},
                    {"condition": "pneumonia", "confidence": 0.45},
                ],
                "recommended_actions": [
                    "Moderate-confidence finding: acute_bronchitis. Recommend further diagnostic testing.",
                    "This is an AI-assisted prediction and should not replace professional medical judgment.",
                ],
                "latency_ms": 52.1,
                "model_version": "1.2.0",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Multimodal Prediction
# -----------------------------------------------------------------------

class MultimodalPredictionResponse(BaseModel):
    """Response for multimodal (image + symptoms) prediction.

    Attributes:
        request_id: Unique request identifier.
        status: Execution status.
        fused_predictions: Combined predictions from image and symptom modalities.
        image_contribution: Relative contribution of the image modality (0–1).
        symptom_contribution: Relative contribution of the symptom modality (0–1).
        confidence_score: Overall fused confidence score.
        latency_ms: End-to-end latency in milliseconds.
        model_version: Version of the model used.
    """

    request_id: str = Field(..., description="Unique request identifier.")
    status: str = Field(..., description="Execution status.")
    fused_predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Combined predictions from all modalities.",
    )
    image_contribution: float = Field(
        ...,
        description="Relative contribution of the image modality (0–1).",
        ge=0.0,
        le=1.0,
    )
    symptom_contribution: float = Field(
        ...,
        description="Relative contribution of the symptom modality (0–1).",
        ge=0.0,
        le=1.0,
    )
    confidence_score: float = Field(
        ...,
        description="Overall fused confidence score.",
        ge=0.0,
        le=1.0,
    )
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")
    model_version: str = Field(
        default="unknown",
        description="Version of the model used for inference.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "request_id": "m1n2o3p4q5r6",
                "status": "success",
                "fused_predictions": [
                    {"condition": "pulmonary_embolism", "confidence": 0.88},
                    {"condition": "pneumonia", "confidence": 0.52},
                ],
                "image_contribution": 0.65,
                "symptom_contribution": 0.35,
                "confidence_score": 0.88,
                "latency_ms": 210.5,
                "model_version": "1.2.0",
            }
        ]
    }}


# -----------------------------------------------------------------------
# Risk Prediction
# -----------------------------------------------------------------------

class RiskPredictionResponse(BaseModel):
    """Response for risk score prediction.

    Attributes:
        request_id: Unique request identifier.
        status: Execution status.
        risk_scores: Per-condition risk scores in the range [0, 1].
        overall_risk_score: Composite risk score.
        risk_level: Categorical risk level.
        contributing_factors: Key factors contributing to the risk assessment.
        recommendations: Risk-based recommendations.
        latency_ms: End-to-end latency in milliseconds.
    """

    request_id: str = Field(..., description="Unique request identifier.")
    status: str = Field(..., description="Execution status.")
    risk_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-condition risk scores.",
    )
    overall_risk_score: float = Field(
        ...,
        description="Composite overall risk score (0–1).",
        ge=0.0,
        le=1.0,
    )
    risk_level: str = Field(
        ...,
        description="Categorical risk level: low, moderate, high, or critical.",
    )
    contributing_factors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Key contributing factors.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Risk-based clinical recommendations.",
    )
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "request_id": "r1s2t3u4v5w6",
                "status": "success",
                "risk_scores": {
                    "cardiovascular_disease": 0.62,
                    "stroke": 0.45,
                },
                "overall_risk_score": 0.62,
                "risk_level": "high",
                "contributing_factors": [
                    {"factor": "hypertension", "category": "medical_history", "impact": "moderate"},
                    {"factor": "smoking", "category": "lifestyle", "impact": "high"},
                ],
                "recommendations": [
                    "High risk level. Schedule a consultation with a specialist soon.",
                    "Risk factor: cardiovascular_disease (score: 0.62). Discuss with your healthcare provider.",
                ],
                "latency_ms": 85.7,
            }
        ]
    }}


# -----------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------

class ReportGenerateResponse(BaseModel):
    """Response for report generation requests.

    Attributes:
        report_id: Unique report identifier.
        status: Generation status.
        patient_id: Patient identifier.
        report_type: Type of report generated.
        generated_at: ISO 8601 timestamp of generation.
        findings_count: Number of findings in the report.
        download_url: URL to download the generated report.
        latency_ms: Generation latency in milliseconds.
    """

    report_id: str = Field(..., description="Unique report identifier.")
    status: str = Field(..., description="Generation status.")
    patient_id: str = Field(..., description="Patient identifier.")
    report_type: str = Field(..., description="Type of report generated.")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp.")
    findings_count: int = Field(..., description="Number of findings in the report.")
    download_url: str = Field(..., description="URL to download the report.")
    latency_ms: float = Field(..., description="Generation latency in milliseconds.")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "report_id": "a1b2c3d4e5f6g7h8",
                "status": "generated",
                "patient_id": "P-12345",
                "report_type": "diagnostic",
                "generated_at": "2025-01-15T10:30:00Z",
                "findings_count": 3,
                "download_url": "/api/v1/report/a1b2c3d4e5f6g7h8/download",
                "latency_ms": 120.5,
            }
        ]
    }}


class ReportRetrieveResponse(BaseModel):
    """Response for report retrieval requests.

    Attributes:
        report_id: Unique report identifier.
        status: Report status.
        patient_id: Patient identifier.
        report_type: Type of report.
        generated_at: ISO 8601 generation timestamp.
        findings: List of finding dictionaries.
        recommendations: List of recommendation strings.
        risk_assessment: Risk assessment summary.
        clinician_notes: Free-text clinician notes.
        metadata: Report metadata.
    """

    report_id: str = Field(..., description="Unique report identifier.")
    status: str = Field(..., description="Report status.")
    patient_id: str = Field(..., description="Patient identifier.")
    report_type: str = Field(..., description="Type of report.")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp.")
    findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of clinical findings.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Clinical recommendations.",
    )
    risk_assessment: dict[str, Any] = Field(
        default_factory=dict,
        description="Risk assessment summary.",
    )
    clinician_notes: str = Field(
        default="",
        description="Free-text clinician notes.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Report metadata.",
    )


class ReportDownloadResponse(BaseModel):
    """Response for report download requests.

    Attributes:
        report_id: Unique report identifier.
        format: Output format (``"json"``, ``"text"``, ``"pdf"``).
        content: The report content as a string.
        content_type: MIME type of the content.
        size_bytes: Content size in bytes.
        filename: Suggested download filename.
    """

    report_id: str = Field(..., description="Unique report identifier.")
    format: str = Field(..., description="Output format.")
    content: str = Field(..., description="Report content as a string.")
    content_type: str = Field(..., description="MIME type of the content.")
    size_bytes: int = Field(..., description="Content size in bytes.")
    filename: str = Field(..., description="Suggested download filename.")
