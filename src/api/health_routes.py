"""
Health Check Routes for MedVision-AI.

Provides liveness, readiness, and detailed health endpoints for
Kubernetes probes, load-balancer checks, and operational monitoring.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def health_check(request: Request) -> dict[str, Any]:
    """Return a lightweight liveness check.

    This endpoint is designed for Kubernetes liveness probes and
    load-balancer health checks.  It confirms the application process
    is responsive without performing any dependency checks.

    Returns:
        A dictionary with status ``"healthy"`` and a timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "MedVision-AI",
    }


@router.get("/health/detailed", summary="Detailed health report")
async def health_detailed(request: Request) -> dict[str, Any]:
    """Return a comprehensive health report with dependency status.

    Checks the inference engine, model loader, system resources, and
    runtime environment to produce a detailed health snapshot.

    Returns:
        A dictionary with component-level health status, system
        information, and uptime.
    """
    app_state = request.app.state

    # ---- Component checks ----
    components: dict[str, Any] = {}

    # Inference engine
    engine = getattr(app_state, "inference_engine", None)
    if engine is not None:
        try:
            latency_stats = engine.get_latency_stats()
            components["inference_engine"] = {
                "status": "healthy",
                "warmed_up": latency_stats.get("is_warmed_up", False),
                "total_requests": latency_stats.get("total_requests", 0),
                "mean_latency_ms": latency_stats.get("mean_latency_ms", 0.0),
            }
        except Exception as exc:
            components["inference_engine"] = {"status": "degraded", "error": str(exc)}
    else:
        components["inference_engine"] = {"status": "unavailable"}

    # Model loader
    loader = getattr(app_state, "model_loader", None)
    if loader is not None:
        try:
            available_models = loader.list_available_models()
            components["model_loader"] = {
                "status": "healthy",
                "available_models": len(available_models),
                "model_names": [m.name for m in available_models[:10]],
            }
        except Exception as exc:
            components["model_loader"] = {"status": "degraded", "error": str(exc)}
    else:
        components["model_loader"] = {"status": "unavailable"}

    # ---- Overall status ----
    statuses = [c.get("status") for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "degraded" for s in statuses):
        overall = "degraded"
    else:
        overall = "unavailable"

    # ---- System info ----
    startup_ts = getattr(app_state, "startup_timestamp", time.time())
    uptime_seconds = time.time() - startup_ts

    system_info: dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
    }

    # ---- GPU info ----
    gpu_info: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated_mb": round(
                    torch.cuda.memory_allocated(0) / (1024 * 1024), 2
                ),
                "memory_reserved_mb": round(
                    torch.cuda.memory_reserved(0) / (1024 * 1024), 2
                ),
            }
    except Exception:
        pass

    return {
        "status": overall,
        "timestamp": time.time(),
        "uptime_seconds": round(uptime_seconds, 2),
        "components": components,
        "system": system_info,
        "gpu": gpu_info,
        "request_count": getattr(app_state, "request_count", 0),
        "error_count": getattr(app_state, "error_count", 0),
    }


@router.get("/ready", summary="Readiness probe")
async def readiness_check(request: Request) -> dict[str, Any]:
    """Return a readiness probe result.

    This endpoint is designed for Kubernetes readiness probes.  It
    returns ``"ready"`` only when the inference engine is initialised
    and a model is loaded; otherwise it reports ``"not_ready"`` with
    a reason.

    Returns:
        A dictionary with readiness status and an optional reason.
    """
    engine = getattr(request.app.state, "inference_engine", None)
    loader = getattr(request.app.state, "model_loader", None)

    ready: bool = True
    reasons: list[str] = []

    if engine is None:
        ready = False
        reasons.append("inference_engine_not_initialised")
    elif not getattr(engine, "_is_warmed_up", False):
        reasons.append("inference_engine_not_warmed_up")

    if loader is None:
        ready = False
        reasons.append("model_loader_not_initialised")

    if ready:
        return {"status": "ready", "timestamp": time.time()}

    return {
        "status": "not_ready",
        "reasons": reasons,
        "timestamp": time.time(),
    }
