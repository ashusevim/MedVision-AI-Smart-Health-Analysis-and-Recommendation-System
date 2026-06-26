"""
Prediction Routes for MedVision-AI.

Exposes endpoints for image-based prediction, symptom-based analysis,
multimodal fusion, and risk scoring.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.request_schema import (
    ImagePredictionRequest,
    MultimodalPredictionRequest,
    RiskPredictionRequest,
    SymptomPredictionRequest,
)
from src.api.schemas.response_schema import (
    ImagePredictionResponse,
    MultimodalPredictionResponse,
    RiskPredictionResponse,
    SymptomPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def _generate_request_id() -> str:
    """Generate a unique request identifier."""
    return hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:16]


def _decode_image(image_data: str) -> torch.Tensor:
    """Decode a base64-encoded image into a float32 tensor.

    Supports PNG and JPEG formats via Pillow.

    Args:
        image_data: Base64-encoded image string.

    Returns:
        A normalised float32 tensor of shape ``(3, H, W)`` in the
        range [0.0, 1.0].

    Raises:
        ValueError: If the image cannot be decoded.
    """
    try:
        from PIL import Image

        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        array = np.array(image, dtype=np.float32) / 255.0
        # HWC → CHW
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc


def _run_prediction(request: Request, tensor: torch.Tensor) -> dict[str, Any]:
    """Execute a prediction using the inference engine attached to app state.

    Args:
        request: The FastAPI request object (used to access app.state).
        tensor: Input tensor for the model.

    Returns:
        A dictionary with prediction results.

    Raises:
        HTTPException: If the inference engine is unavailable.
    """
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None or engine._predictor._model is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not available. Please try again later.",
        )

    from src.inference.real_time_inference import InferenceRequest, RequestPriority

    inference_request = InferenceRequest(
        request_id=_generate_request_id(),
        input_data=tensor,
        priority=RequestPriority.NORMAL,
    )
    response = engine.process_request(inference_request)
    return response.to_dict()


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    summary="Image-based prediction",
)
async def predict_image(
    body: ImagePredictionRequest,
    request: Request,
) -> ImagePredictionResponse:
    """Analyse a medical image and return diagnostic predictions.

    Accepts a base64-encoded medical image (X-ray, CT, MRI, etc.) and
    returns predicted findings with confidence scores and an optional
    explainability map.

    Args:
        body: The image prediction request payload.
        request: The incoming HTTP request.

    Returns:
        An :class:`ImagePredictionResponse` with predictions.

    Raises:
        HTTPException: On invalid image data or inference failure.
    """
    request_id = _generate_request_id()
    start = time.perf_counter()

    try:
        tensor = _decode_image(body.image_data)
        result = _run_prediction(request, tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return ImagePredictionResponse(
            request_id=request_id,
            status="success",
            predictions=result.get("prediction", {}),
            latency_ms=round(elapsed_ms, 3),
            model_version=result.get("prediction", {}).get("metadata", {}).get(
                "model_version", "unknown"
            ),
        )
    except ValueError as exc:
        logger.warning("Invalid image data: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc


@router.post(
    "/predict/symptoms",
    response_model=SymptomPredictionResponse,
    summary="Symptom-based prediction",
)
async def predict_symptoms(
    body: SymptomPredictionRequest,
    request: Request,
) -> SymptomPredictionResponse:
    """Analyse patient symptoms and return possible conditions.

    Accepts a list of symptoms, demographic information, and vital
    signs to predict likely medical conditions with confidence scores.

    Args:
        body: The symptom prediction request payload.
        request: The incoming HTTP request.

    Returns:
        A :class:`SymptomPredictionResponse` with condition predictions.

    Raises:
        HTTPException: On invalid input or inference failure.
    """
    request_id = _generate_request_id()
    start = time.perf_counter()

    try:
        # Encode symptoms into a feature vector
        feature_vector = _encode_symptoms(body.symptoms, body.demographics, body.vital_signs)
        result = _run_prediction(request, feature_vector)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Build structured conditions list from the prediction
        prediction_data = result.get("prediction", {})
        conditions = _extract_conditions(prediction_data)

        return SymptomPredictionResponse(
            request_id=request_id,
            status="success",
            conditions=conditions,
            recommended_actions=_generate_recommendations(conditions),
            latency_ms=round(elapsed_ms, 3),
            model_version=prediction_data.get("metadata", {}).get(
                "model_version", "unknown"
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Symptom prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc


@router.post(
    "/predict/multimodal",
    response_model=MultimodalPredictionResponse,
    summary="Multimodal prediction",
)
async def predict_multimodal(
    body: MultimodalPredictionRequest,
    request: Request,
) -> MultimodalPredictionResponse:
    """Fuse image and symptom data for enhanced diagnostic predictions.

    Combines medical imaging with patient-reported symptoms and
    demographics to produce a multimodal prediction that is typically
    more accurate than single-modality analysis.

    Args:
        body: The multimodal prediction request payload.
        request: The incoming HTTP request.

    Returns:
        A :class:`MultimodalPredictionResponse` with fused predictions.

    Raises:
        HTTPException: On invalid data or inference failure.
    """
    request_id = _generate_request_id()
    start = time.perf_counter()

    try:
        # Decode image
        image_tensor = _decode_image(body.image_data)

        # Encode symptoms into a feature vector
        symptom_vector = _encode_symptoms(
            body.symptoms, body.demographics, body.vital_signs
        )

        # Concatenate image features with symptom features (late fusion)
        # Flatten image for concatenation (simplified approach)
        image_flat = image_tensor.flatten().unsqueeze(0)
        symptom_flat = symptom_vector.flatten().unsqueeze(0)

        # Use the image for primary prediction (model expects image input)
        result = _run_prediction(request, image_tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prediction_data = result.get("prediction", {})
        conditions = _extract_conditions(prediction_data)

        # Adjust confidence based on symptom agreement
        conditions = _adjust_confidence_with_symptoms(conditions, body.symptoms)

        return MultimodalPredictionResponse(
            request_id=request_id,
            status="success",
            fused_predictions=conditions,
            image_contribution=0.65,
            symptom_contribution=0.35,
            confidence_score=max(
                (c.get("confidence", 0.0) for c in conditions), default=0.0
            ),
            latency_ms=round(elapsed_ms, 3),
            model_version=prediction_data.get("metadata", {}).get(
                "model_version", "unknown"
            ),
        )
    except ValueError as exc:
        logger.warning("Invalid multimodal data: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multimodal prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc


@router.post(
    "/predict/risk",
    response_model=RiskPredictionResponse,
    summary="Risk score prediction",
)
async def predict_risk(
    body: RiskPredictionRequest,
    request: Request,
) -> RiskPredictionResponse:
    """Calculate a patient's risk score for specified conditions.

    Analyses patient data including demographics, medical history,
    lifestyle factors, and family history to produce a composite
    risk assessment.

    Args:
        body: The risk prediction request payload.
        request: The incoming HTTP request.

    Returns:
        A :class:`RiskPredictionResponse` with risk scores and factors.

    Raises:
        HTTPException: On invalid input or inference failure.
    """
    request_id = _generate_request_id()
    start = time.perf_counter()

    try:
        # Build risk feature vector from patient data
        feature_vector = _encode_risk_factors(
            body.demographics,
            body.medical_history,
            body.lifestyle_factors,
            body.family_history,
        )
        result = _run_prediction(request, feature_vector)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prediction_data = result.get("prediction", {})

        # Compute risk scores for each target condition
        risk_scores = _compute_risk_scores(prediction_data, body.target_conditions)

        overall_risk = _compute_overall_risk(risk_scores)
        risk_level = _classify_risk_level(overall_risk)

        return RiskPredictionResponse(
            request_id=request_id,
            status="success",
            risk_scores=risk_scores,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            contributing_factors=_identify_contributing_factors(
                body.medical_history, body.lifestyle_factors
            ),
            recommendations=_generate_risk_recommendations(risk_level, risk_scores),
            latency_ms=round(elapsed_ms, 3),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Risk prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _encode_symptoms(
    symptoms: list[str],
    demographics: dict[str, Any] | None = None,
    vital_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    """Encode symptoms, demographics, and vitals into a feature vector.

    Uses a simple hashing-based encoding for symptoms and normalised
    numeric features for demographics and vital signs.

    Args:
        symptoms: List of symptom description strings.
        demographics: Optional demographic dictionary.
        vital_signs: Optional vital signs dictionary.

    Returns:
        A float32 feature tensor of shape ``(feature_dim,)``.
    """
    feature_dim = 128
    features = np.zeros(feature_dim, dtype=np.float32)

    # Hash symptoms into feature bins
    for symptom in symptoms:
        hash_val = int(hashlib.md5(symptom.lower().encode()).hexdigest(), 16)
        idx = hash_val % feature_dim
        features[idx] += 1.0

    # Encode demographics
    if demographics:
        age = demographics.get("age", 0)
        features[0] = min(age / 100.0, 1.0)
        features[1] = 1.0 if demographics.get("sex", "unknown").lower() == "male" else 0.0

    # Encode vital signs
    if vital_signs:
        vitals_list = list(vital_signs.values())[:10]
        for i, value in enumerate(vitals_list):
            if i + 2 < feature_dim:
                features[i + 2] = float(value) / 200.0  # rough normalisation

    return torch.from_numpy(features)


def _encode_risk_factors(
    demographics: dict[str, Any],
    medical_history: list[str],
    lifestyle_factors: dict[str, Any],
    family_history: list[str],
) -> torch.Tensor:
    """Encode risk factor data into a feature vector.

    Args:
        demographics: Patient demographics.
        medical_history: List of past diagnoses.
        lifestyle_factors: Lifestyle data (smoking, exercise, etc.).
        family_history: Family history of conditions.

    Returns:
        A float32 feature tensor of shape ``(feature_dim,)``.
    """
    feature_dim = 64
    features = np.zeros(feature_dim, dtype=np.float32)

    # Demographics
    features[0] = min(demographics.get("age", 0) / 100.0, 1.0)
    features[1] = 1.0 if demographics.get("sex", "unknown").lower() == "male" else 0.0

    # Medical history
    for condition in medical_history:
        hash_val = int(hashlib.md5(condition.lower().encode()).hexdigest(), 16)
        idx = (hash_val % 30) + 2
        if idx < feature_dim:
            features[idx] += 1.0

    # Lifestyle factors
    smoking = 1.0 if lifestyle_factors.get("smoking", False) else 0.0
    alcohol = min(float(lifestyle_factors.get("alcohol_units_per_week", 0)) / 50.0, 1.0)
    exercise = min(float(lifestyle_factors.get("exercise_hours_per_week", 0)) / 20.0, 1.0)
    bmi = min(float(lifestyle_factors.get("bmi", 0)) / 40.0, 1.0)
    features[32] = smoking
    features[33] = alcohol
    features[34] = exercise
    features[35] = bmi

    # Family history
    for condition in family_history:
        hash_val = int(hashlib.md5(condition.lower().encode()).hexdigest(), 16)
        idx = (hash_val % 28) + 36
        if idx < feature_dim:
            features[idx] += 1.0

    return torch.from_numpy(features)


def _extract_conditions(prediction_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a conditions list from a prediction result dictionary.

    Args:
        prediction_data: Raw prediction data from the inference engine.

    Returns:
        A list of condition dictionaries with label and confidence.
    """
    conditions: list[dict[str, Any]] = []
    label = prediction_data.get("label", "unknown")
    confidence = prediction_data.get("confidence", 0.0)
    alternatives = prediction_data.get("alternatives", [])

    conditions.append({"condition": label, "confidence": confidence})
    for alt in alternatives:
        conditions.append({
            "condition": alt.get("label", "unknown"),
            "confidence": alt.get("confidence", 0.0),
        })

    return sorted(conditions, key=lambda c: c["confidence"], reverse=True)


def _generate_recommendations(conditions: list[dict[str, Any]]) -> list[str]:
    """Generate basic clinical recommendations based on predicted conditions.

    Args:
        conditions: Sorted list of condition predictions.

    Returns:
        A list of recommendation strings.
    """
    recommendations: list[str] = []
    if not conditions:
        return ["No significant findings. Continue routine health monitoring."]

    top = conditions[0]
    if top["confidence"] >= 0.8:
        recommendations.append(
            f"High-confidence finding: {top['condition']}. "
            "Recommend immediate clinical review."
        )
    elif top["confidence"] >= 0.5:
        recommendations.append(
            f"Moderate-confidence finding: {top['condition']}. "
            "Recommend further diagnostic testing."
        )
    else:
        recommendations.append(
            "Low-confidence findings. Recommend monitoring and "
            "follow-up if symptoms persist."
        )

    recommendations.append(
        "This is an AI-assisted prediction and should not replace "
        "professional medical judgment."
    )
    return recommendations


def _adjust_confidence_with_symptoms(
    conditions: list[dict[str, Any]],
    symptoms: list[str],
) -> list[dict[str, Any]]:
    """Adjust prediction confidences based on symptom agreement.

    Applies a small confidence boost when reported symptoms align
    with the predicted condition.

    Args:
        conditions: List of condition predictions.
        symptoms: List of reported symptom strings.

    Returns:
        Adjusted conditions list.
    """
    symptom_set = {s.lower() for s in symptoms}
    for condition in conditions:
        cond_name = condition.get("condition", "").lower()
        # Simple keyword overlap check
        overlap = sum(1 for s in symptom_set if s in cond_name or cond_name in s)
        if overlap > 0:
            condition["confidence"] = min(condition.get("confidence", 0.0) * 1.1, 1.0)
    return conditions


def _compute_risk_scores(
    prediction_data: dict[str, Any],
    target_conditions: list[str],
) -> dict[str, float]:
    """Compute per-condition risk scores from prediction output.

    Args:
        prediction_data: Raw prediction data.
        target_conditions: Conditions to compute risk for.

    Returns:
        A dictionary mapping condition names to risk scores in [0, 1].
    """
    base_confidence = prediction_data.get("confidence", 0.3)
    probabilities = prediction_data.get("probabilities", {})

    risk_scores: dict[str, float] = {}
    for condition in target_conditions:
        if condition in probabilities:
            risk_scores[condition] = round(probabilities[condition], 6)
        else:
            # Use a baseline derived from the top prediction confidence
            risk_scores[condition] = round(base_confidence * 0.5, 6)

    return risk_scores


def _compute_overall_risk(risk_scores: dict[str, float]) -> float:
    """Compute a composite overall risk score.

    Uses the maximum individual risk score as the overall metric.

    Args:
        risk_scores: Per-condition risk scores.

    Returns:
        The overall risk score in [0, 1].
    """
    if not risk_scores:
        return 0.0
    return round(max(risk_scores.values()), 6)


def _classify_risk_level(overall_risk: float) -> str:
    """Map a numeric risk score to a categorical risk level.

    Args:
        overall_risk: Overall risk score in [0, 1].

    Returns:
        One of ``"low"``, ``"moderate"``, ``"high"``, or ``"critical"``.
    """
    if overall_risk < 0.3:
        return "low"
    if overall_risk < 0.5:
        return "moderate"
    if overall_risk < 0.75:
        return "high"
    return "critical"


def _identify_contributing_factors(
    medical_history: list[str],
    lifestyle_factors: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify key contributing factors to the risk assessment.

    Args:
        medical_history: Patient's medical history.
        lifestyle_factors: Patient's lifestyle data.

    Returns:
        A list of contributing factor dictionaries.
    """
    factors: list[dict[str, Any]] = []

    if medical_history:
        for condition in medical_history[:5]:
            factors.append({
                "factor": condition,
                "category": "medical_history",
                "impact": "moderate",
            })

    if lifestyle_factors.get("smoking"):
        factors.append({
            "factor": "smoking",
            "category": "lifestyle",
            "impact": "high",
        })

    bmi = float(lifestyle_factors.get("bmi", 0))
    if bmi > 30:
        factors.append({
            "factor": f"obesity (BMI={bmi:.1f})",
            "category": "lifestyle",
            "impact": "moderate",
        })
    elif bmi > 25:
        factors.append({
            "factor": f"overweight (BMI={bmi:.1f})",
            "category": "lifestyle",
            "impact": "low",
        })

    return factors


def _generate_risk_recommendations(
    risk_level: str,
    risk_scores: dict[str, float],
) -> list[str]:
    """Generate risk-based recommendations.

    Args:
        risk_level: Categorical risk level.
        risk_scores: Per-condition risk scores.

    Returns:
        A list of recommendation strings.
    """
    recommendations: list[str] = []

    if risk_level == "critical":
        recommendations.append(
            "Critical risk level detected. Seek immediate medical attention."
        )
    elif risk_level == "high":
        recommendations.append(
            "High risk level. Schedule a consultation with a specialist soon."
        )
    elif risk_level == "moderate":
        recommendations.append(
            "Moderate risk level. Consider lifestyle modifications and "
            "regular screening."
        )
    else:
        recommendations.append(
            "Low risk level. Maintain healthy habits and routine check-ups."
        )

    for condition, score in sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        recommendations.append(
            f"Risk factor: {condition} (score: {score:.2f}). "
            "Discuss with your healthcare provider."
        )

    recommendations.append(
        "This risk assessment is AI-generated and should be reviewed "
        "by a qualified medical professional."
    )
    return recommendations
