"""
FastAPI Application Entry Point for MedVision-AI.

Creates and configures the FastAPI application instance with middleware,
routers, startup/shutdown lifecycle events, and CORS support.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import health_routes, prediction_routes, report_routes

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------

APP_TITLE = "MedVision-AI"
APP_DESCRIPTION = (
    "Intelligent Medical Imaging & Diagnostic Assistance Platform — "
    "provides image-based predictions, symptom analysis, multimodal "
    "fusion, risk scoring, and automated diagnostic reports."
)
APP_VERSION = "1.0.0"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
CORS_ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Request-ID",
    "X-Client-Version",
]


# -----------------------------------------------------------------------
# Lifespan (startup / shutdown)
# -----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On startup the function initialises inference engines and loads
    models.  On shutdown it releases GPU memory and flushes logs.

    Args:
        app: The FastAPI application instance (injected automatically).
    """
    logger.info("MedVision-AI starting up …")

    # ---- Startup ----
    startup_time = time.perf_counter()

    # Initialise shared state that routes can access via app.state
    app.state.startup_timestamp = time.time()
    app.state.inference_engine = None
    app.state.model_loader = None
    app.state.request_count = 0
    app.state.error_count = 0

    # Attempt to warm up the inference engine (non-fatal if it fails)
    try:
        from src.inference.real_time_inference import RealTimeInference

        config: dict[str, Any] = {
            "device": "cpu",
            "warmup_iterations": 3,
            "max_latency_ms": 500.0,
            "enable_latency_tracking": True,
        }
        engine = RealTimeInference(config)
        warmup_result = engine.warm_up()
        app.state.inference_engine = engine
        logger.info("Inference engine warm-up: %s", warmup_result.get("status"))
    except Exception as exc:
        logger.warning("Inference engine warm-up skipped: %s", exc)

    # Attempt to initialise the model loader
    try:
        from src.inference.model_loader import ModelLoader

        loader = ModelLoader({"device": "cpu", "registry_path": "./model_registry"})
        app.state.model_loader = loader
        available = loader.list_available_models()
        logger.info("Model loader ready — %d model(s) in registry", len(available))
    except Exception as exc:
        logger.warning("Model loader initialisation skipped: %s", exc)

    elapsed = time.perf_counter() - startup_time
    logger.info("Startup completed in %.2fs", elapsed)

    yield  # ← application is running

    # ---- Shutdown ----
    logger.info("MedVision-AI shutting down …")
    import gc

    import torch

    # Release GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(
        "Shutdown complete — total requests: %d, errors: %d",
        getattr(app.state, "request_count", 0),
        getattr(app.state, "error_count", 0),
    )


# -----------------------------------------------------------------------
# Application factory
# -----------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the fully configured FastAPI application.

    Returns:
        A :class:`FastAPI` instance with all middleware, routers, and
        lifecycle handlers attached.
    """
    application = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ---- Middleware (order matters — outermost first) ----

    # GZip compression for responses > 1 KB
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )

    # Custom middleware stack
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(AuthMiddleware)

    # ---- Routers ----
    application.include_router(
        health_routes.router,
        prefix="/api/v1",
        tags=["Health"],
    )
    application.include_router(
        prediction_routes.router,
        prefix="/api/v1",
        tags=["Predictions"],
    )
    application.include_router(
        report_routes.router,
        prefix="/api/v1",
        tags=["Reports"],
    )

    # ---- Root redirect ----
    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Redirect root to API documentation."""
        return {
            "message": "MedVision-AI API",
            "version": APP_VERSION,
            "docs": "/docs",
        }

    return application


# -----------------------------------------------------------------------
# Module-level app instance (used by uvicorn / ASGI servers)
# -----------------------------------------------------------------------

app = create_app()
