"""
Report Routes for MedVision-AI.

Provides endpoints for generating, retrieving, and downloading
diagnostic reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.request_schema import ReportGenerateRequest
from src.api.schemas.response_schema import (
    ReportDownloadResponse,
    ReportGenerateResponse,
    ReportRetrieveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory report store (in production, use a database)
_report_store: dict[str, dict[str, Any]] = {}


@router.post(
    "/report/generate",
    response_model=ReportGenerateResponse,
    summary="Generate a diagnostic report",
)
async def generate_report(
    body: ReportGenerateRequest,
    request: Request,
) -> ReportGenerateResponse:
    """Generate a comprehensive diagnostic report.

    Creates a structured report from prediction results and patient
    data.  The report includes findings, recommendations, risk
    assessments, and a clinician review section.

    Args:
        body: The report generation request payload.
        request: The incoming HTTP request.

    Returns:
        A :class:`ReportGenerateResponse` with the report ID and
        summary information.

    Raises:
        HTTPException: On generation failure.
    """
    start = time.perf_counter()

    try:
        # Generate a unique report ID
        report_id = hashlib.sha256(
            f"{body.patient_id}{time.time_ns()}".encode()
        ).hexdigest()[:16]

        # Build the report document
        report_document = _build_report_document(
            report_id=report_id,
            patient_id=body.patient_id,
            prediction_ids=body.prediction_ids,
            report_type=body.report_type,
            include_images=body.include_images,
            additional_notes=body.additional_notes,
        )

        # Store the report
        _report_store[report_id] = report_document

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "Report generated — id=%s, patient=%s, type=%s, %.1fms",
            report_id,
            body.patient_id,
            body.report_type,
            elapsed_ms,
        )

        return ReportGenerateResponse(
            report_id=report_id,
            status="generated",
            patient_id=body.patient_id,
            report_type=body.report_type,
            generated_at=report_document["generated_at"],
            findings_count=len(report_document.get("findings", [])),
            download_url=f"/api/v1/report/{report_id}/download",
            latency_ms=round(elapsed_ms, 3),
        )
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {exc}"
        ) from exc


@router.get(
    "/report/{report_id}",
    response_model=ReportRetrieveResponse,
    summary="Retrieve a diagnostic report",
)
async def retrieve_report(
    report_id: str,
    request: Request,
) -> ReportRetrieveResponse:
    """Retrieve a previously generated diagnostic report by its ID.

    Args:
        report_id: The unique report identifier.
        request: The incoming HTTP request.

    Returns:
        A :class:`ReportRetrieveResponse` with the full report content.

    Raises:
        HTTPException: If the report is not found.
    """
    report = _report_store.get(report_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {report_id}",
        )

    return ReportRetrieveResponse(
        report_id=report_id,
        status=report.get("status", "unknown"),
        patient_id=report.get("patient_id", ""),
        report_type=report.get("report_type", ""),
        generated_at=report.get("generated_at", ""),
        findings=report.get("findings", []),
        recommendations=report.get("recommendations", []),
        risk_assessment=report.get("risk_assessment", {}),
        clinician_notes=report.get("clinician_notes", ""),
        metadata=report.get("metadata", {}),
    )


@router.get(
    "/report/{report_id}/download",
    response_model=ReportDownloadResponse,
    summary="Download a diagnostic report",
)
async def download_report(
    report_id: str,
    request: Request,
) -> ReportDownloadResponse:
    """Download a diagnostic report in the requested format.

    Supports JSON, plain text, and PDF-like structured text formats.

    Args:
        report_id: The unique report identifier.
        request: The incoming HTTP request.

    Returns:
        A :class:`ReportDownloadResponse` with the report content and
        format information.

    Raises:
        HTTPException: If the report is not found.
    """
    report = _report_store.get(report_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {report_id}",
        )

    # Determine requested format (default: json)
    format_param = request.query_params.get("format", "json")

    if format_param == "text":
        content = _format_report_as_text(report)
        content_type = "text/plain"
    elif format_param == "pdf":
        content = _format_report_as_structured_text(report)
        content_type = "text/plain"
    else:
        content = json.dumps(report, indent=2, default=str)
        content_type = "application/json"

    return ReportDownloadResponse(
        report_id=report_id,
        format=format_param,
        content=content,
        content_type=content_type,
        size_bytes=len(content.encode("utf-8")),
        filename=f"medvision_report_{report_id}.{format_param}",
    )


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _build_report_document(
    report_id: str,
    patient_id: str,
    prediction_ids: list[str],
    report_type: str,
    include_images: bool,
    additional_notes: str | None,
) -> dict[str, Any]:
    """Construct a full report document from prediction references.

    Args:
        report_id: Unique report identifier.
        patient_id: Patient identifier.
        prediction_ids: References to prediction results.
        report_type: Type of report (``"diagnostic"``, ``"screening"``,
            ``"follow_up"``).
        include_images: Whether to include image references.
        additional_notes: Optional free-text notes.

    Returns:
        A complete report document dictionary.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Build findings from prediction references
    findings: list[dict[str, Any]] = []
    for idx, pred_id in enumerate(prediction_ids):
        findings.append({
            "finding_id": f"F-{idx + 1:03d}",
            "prediction_id": pred_id,
            "description": f"Finding based on prediction {pred_id}",
            "confidence": 0.85,  # Placeholder — in production, look up the prediction
            "severity": "moderate",
        })

    # Generate recommendations based on report type
    recommendations = _build_recommendations(report_type, findings)

    # Risk assessment
    risk_assessment: dict[str, Any] = {
        "overall_risk": "moderate",
        "risk_factors": [f["description"] for f in findings],
        "requires_follow_up": report_type in ("diagnostic", "screening"),
    }

    metadata: dict[str, Any] = {
        "generator": "MedVision-AI v1.0.0",
        "prediction_ids": prediction_ids,
        "include_images": include_images,
        "generated_by": "automated",
    }

    return {
        "report_id": report_id,
        "patient_id": patient_id,
        "report_type": report_type,
        "status": "completed",
        "generated_at": now,
        "findings": findings,
        "recommendations": recommendations,
        "risk_assessment": risk_assessment,
        "clinician_notes": additional_notes or "",
        "metadata": metadata,
    }


def _build_recommendations(
    report_type: str,
    findings: list[dict[str, Any]],
) -> list[str]:
    """Build clinical recommendations for the report.

    Args:
        report_type: Type of report.
        findings: List of finding dictionaries.

    Returns:
        A list of recommendation strings.
    """
    recommendations: list[str] = []

    if report_type == "diagnostic":
        recommendations.append(
            "Review all findings with the patient during the next appointment."
        )
        recommendations.append(
            "Order follow-up imaging if any finding has confidence > 0.7."
        )
    elif report_type == "screening":
        recommendations.append(
            "Continue routine screening per established guidelines."
        )
        if findings:
            recommendations.append(
                "Flag any abnormal findings for specialist review."
            )
    elif report_type == "follow_up":
        recommendations.append(
            "Compare current findings with prior reports to assess progression."
        )

    recommendations.append(
        "This report is AI-generated and should be reviewed by a "
        "qualified healthcare professional before clinical use."
    )

    return recommendations


def _format_report_as_text(report: dict[str, Any]) -> str:
    """Format the report as a human-readable plain-text string.

    Args:
        report: The report document dictionary.

    Returns:
        A formatted text string.
    """
    lines: list[str] = [
        "=" * 60,
        "MedVision-AI Diagnostic Report",
        "=" * 60,
        f"Report ID    : {report.get('report_id', 'N/A')}",
        f"Patient ID   : {report.get('patient_id', 'N/A')}",
        f"Type         : {report.get('report_type', 'N/A')}",
        f"Generated At : {report.get('generated_at', 'N/A')}",
        f"Status       : {report.get('status', 'N/A')}",
        "",
        "-" * 40,
        "Findings",
        "-" * 40,
    ]

    for finding in report.get("findings", []):
        lines.append(
            f"  [{finding.get('finding_id', '?')}] "
            f"{finding.get('description', 'No description')} "
            f"(confidence: {finding.get('confidence', 0):.2f}, "
            f"severity: {finding.get('severity', 'unknown')})"
        )

    lines.append("")
    lines.append("-" * 40)
    lines.append("Recommendations")
    lines.append("-" * 40)
    for rec in report.get("recommendations", []):
        lines.append(f"  • {rec}")

    lines.append("")
    lines.append("-" * 40)
    lines.append("Risk Assessment")
    lines.append("-" * 40)
    risk = report.get("risk_assessment", {})
    lines.append(f"  Overall Risk    : {risk.get('overall_risk', 'N/A')}")
    lines.append(f"  Requires Follow-up: {risk.get('requires_follow_up', False)}")

    if report.get("clinician_notes"):
        lines.append("")
        lines.append("-" * 40)
        lines.append("Clinician Notes")
        lines.append("-" * 40)
        lines.append(f"  {report['clinician_notes']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _format_report_as_structured_text(report: dict[str, Any]) -> str:
    """Format the report as a structured text suitable for PDF conversion.

    Args:
        report: The report document dictionary.

    Returns:
        A structured text string.
    """
    # Use a more formal layout suitable for printing / PDF generation
    header = (
        "MEDVISION-AI — DIAGNOSTIC REPORT\n"
        "================================\n\n"
        f"Report ID : {report.get('report_id', 'N/A')}\n"
        f"Patient   : {report.get('patient_id', 'N/A')}\n"
        f"Date      : {report.get('generated_at', 'N/A')}\n"
        f"Type      : {report.get('report_type', 'N/A').upper()}\n\n"
    )

    findings_section = "FINDINGS\n--------\n"
    for finding in report.get("findings", []):
        findings_section += (
            f"  {finding.get('finding_id', '?')}. "
            f"{finding.get('description', '')}\n"
            f"     Confidence: {finding.get('confidence', 0):.1%}  |  "
            f"Severity: {finding.get('severity', 'N/A')}\n\n"
        )

    recommendations_section = "RECOMMENDATIONS\n---------------\n"
    for i, rec in enumerate(report.get("recommendations", []), 1):
        recommendations_section += f"  {i}. {rec}\n"

    risk_section = (
        "\nRISK ASSESSMENT\n---------------\n"
        f"  Overall: {report.get('risk_assessment', {}).get('overall_risk', 'N/A').upper()}\n"
        f"  Follow-up required: "
        f"{'Yes' if report.get('risk_assessment', {}).get('requires_follow_up') else 'No'}\n"
    )

    disclaimer = (
        "\n\nDISCLAIMER\n----------\n"
        "This report was generated by MedVision-AI, an artificial intelligence\n"
        "system. It is intended to assist, not replace, clinical judgment.\n"
        "All findings should be reviewed by a qualified healthcare professional.\n"
    )

    return header + findings_section + recommendations_section + risk_section + disclaimer
