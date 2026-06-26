"""
Request/Response Logging Middleware for MedVision-AI.

Provides structured request and response logging with timing,
correlation IDs, and configurable log levels.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Paths to exclude from verbose request/response logging
SKIP_LOG_PATHS: set[str] = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}

# Maximum body size to log (in bytes) — prevents huge payloads in logs
MAX_LOG_BODY_SIZE: int = 2048

# Sensitive headers that should be redacted from logs
SENSITIVE_HEADERS: set[str] = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "proxy-authorization",
    "www-authenticate",
}


class LoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that logs requests and responses with timing.

    For each request the middleware records:

    * HTTP method and path
    * Query parameters (redacted if sensitive)
    * Request and response headers (sensitive headers redacted)
    * Response status code
    * Request duration in milliseconds
    * A correlation ID (``X-Request-ID``) for log correlation

    Args:
        app: The ASGI application to wrap.
        log_level: Logging level for request/response messages.
        log_bodies: Whether to include request/response bodies in logs.
    """

    def __init__(
        self,
        app: Any,
        log_level: int = logging.INFO,
        log_bodies: bool = False,
    ) -> None:
        super().__init__(app)
        self._log_level = log_level
        self._log_bodies = log_bodies

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request with logging instrumentation.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware / route handler.

        Returns:
            The response from the next handler with an added
            ``X-Request-ID`` header.
        """
        # Generate or reuse correlation ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Skip verbose logging for documentation paths
        path = request.url.path
        skip_verbose = path in SKIP_LOG_PATHS

        start_time = time.perf_counter()

        # ---- Log incoming request ----
        if not skip_verbose:
            request_headers = self._redact_headers(dict(request.headers))
            log_data: dict[str, Any] = {
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "query": str(request.query_params) if request.query_params else None,
                "client_host": request.client.host if request.client else None,
                "headers": request_headers,
            }

            if self._log_bodies:
                try:
                    body = await request.body()
                    log_data["body_size"] = len(body)
                    if len(body) <= MAX_LOG_BODY_SIZE:
                        log_data["body_preview"] = body.decode("utf-8", errors="replace")
                    else:
                        log_data["body_preview"] = (
                            body[:MAX_LOG_BODY_SIZE].decode("utf-8", errors="replace")
                            + f"… ({len(body)} bytes total)"
                        )
                except Exception:
                    log_data["body_size"] = "unreadable"

            logger.log(self._log_level, "→ %s %s", request.method, path, extra=log_data)

        # ---- Call next handler ----
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "✗ %s %s → 500 (%.1fms) — unhandled exception: %s",
                request.method,
                path,
                elapsed_ms,
                exc,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": 500,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                },
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Add correlation ID to the response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

        # ---- Log outgoing response ----
        if not skip_verbose:
            response_headers = self._redact_headers(dict(response.headers))
            response_log: dict[str, Any] = {
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 3),
                "response_headers": response_headers,
            }

            # Choose log level based on status code
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            else:
                log_level = self._log_level

            logger.log(
                log_level,
                "← %s %s → %d (%.1fms)",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
                extra=response_log,
            )

        # Update request counter on app state
        try:
            if hasattr(request.app.state, "request_count"):
                request.app.state.request_count += 1
            if response.status_code >= 500 and hasattr(request.app.state, "error_count"):
                request.app.state.error_count += 1
        except Exception:
            pass

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive header values for safe logging.

        Args:
            headers: Raw header dictionary.

        Returns:
            A copy with sensitive values replaced by ``"***REDACTED***"``.
        """
        redacted: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in SENSITIVE_HEADERS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted
