"""
Rate Limiting Middleware for MedVision-AI.

Provides sliding-window rate limiting per client IP and per API key,
with configurable limits and graceful HTTP 429 responses.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Paths exempt from rate limiting
EXEMPT_PATHS: set[str] = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/api/v1/ready",
}


@dataclass
class RateLimitRule:
    """A single rate-limiting rule.

    Attributes:
        name: Human-readable name for the rule.
        max_requests: Maximum number of requests allowed in the window.
        window_seconds: Sliding window duration in seconds.
    """

    name: str
    max_requests: int
    window_seconds: int


@dataclass
class ClientBucket:
    """Token-bucket state for a single client.

    Attributes:
        timestamps: List of request timestamps within the current window.
    """

    timestamps: list[float] = field(default_factory=list)

    def record_request(self, now: float) -> None:
        """Add a request timestamp to the bucket.

        Args:
            now: Current UNIX timestamp.
        """
        self.timestamps.append(now)

    def cleanup(self, now: float, window_seconds: int) -> None:
        """Remove timestamps that have fallen outside the window.

        Args:
            now: Current UNIX timestamp.
            window_seconds: Window duration in seconds.
        """
        cutoff = now - window_seconds
        self.timestamps = [ts for ts in self.timestamps if ts > cutoff]

    @property
    def request_count(self) -> int:
        """Return the current number of recorded requests."""
        return len(self.timestamps)


class RateLimitStore:
    """In-memory rate limit state store.

    Maintains per-client token buckets and supports configurable
    rate-limit rules.  Periodic cleanup prevents unbounded memory
    growth.

    Args:
        rules: A list of :class:`RateLimitRule` instances to enforce.
        cleanup_interval: Seconds between automatic cleanup sweeps.
    """

    def __init__(
        self,
        rules: list[RateLimitRule],
        cleanup_interval: int = 60,
    ) -> None:
        self._rules = rules
        self._cleanup_interval = cleanup_interval
        self._buckets: dict[str, dict[str, ClientBucket]] = defaultdict(dict)
        self._last_cleanup: float = time.time()

    def is_allowed(self, client_id: str, rule: RateLimitRule) -> tuple[bool, int, int]:
        """Check whether a request from *client_id* is allowed under *rule*.

        Uses a sliding-window counter approach.

        Args:
            client_id: The client identifier (IP or API key).
            rule: The rate limit rule to check.

        Returns:
            A tuple of ``(allowed, remaining, retry_after_seconds)``.
        """
        now = time.time()
        self._maybe_cleanup(now)

        bucket = self._buckets[rule.name].setdefault(client_id, ClientBucket())
        bucket.cleanup(now, rule.window_seconds)

        if bucket.request_count >= rule.max_requests:
            # Calculate when the oldest request in the window will expire
            oldest = min(bucket.timestamps) if bucket.timestamps else now
            retry_after = int(oldest + rule.window_seconds - now) + 1
            remaining = 0
            return False, remaining, max(retry_after, 1)

        bucket.record_request(now)
        remaining = rule.max_requests - bucket.request_count
        return True, remaining, 0

    def _maybe_cleanup(self, now: float) -> None:
        """Run cleanup if the interval has elapsed.

        Args:
            now: Current UNIX timestamp.
        """
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Find the largest window across all rules
        max_window = max((r.window_seconds for r in self._rules), default=60)
        cutoff = now - max_window

        for rule_name, clients in self._buckets.items():
            stale_clients: list[str] = []
            for client_id, bucket in clients.items():
                bucket.timestamps = [ts for ts in bucket.timestamps if ts > cutoff]
                if not bucket.timestamps:
                    stale_clients.append(client_id)
            for client_id in stale_clients:
                del clients[client_id]

        self._last_cleanup = now


# -----------------------------------------------------------------------
# Default rules
# -----------------------------------------------------------------------

DEFAULT_RULES: list[RateLimitRule] = [
    RateLimitRule(name="global", max_requests=100, window_seconds=60),
    RateLimitRule(name="prediction", max_requests=30, window_seconds=60),
    RateLimitRule(name="report", max_requests=10, window_seconds=60),
]

# Map URL prefixes to the rule that should be applied
PATH_RULE_MAP: dict[str, str] = {
    "/api/v1/predict": "prediction",
    "/api/v1/report": "report",
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces per-client rate limits.

    Applies sliding-window rate limiting based on client IP address
    (or ``X-API-Key`` header if present).  Different rate limits
    can be configured for different API path prefixes.

    When a client exceeds the rate limit, the middleware returns an
    HTTP 429 response with ``Retry-After`` and
    ``X-RateLimit-*`` headers.

    Args:
        app: The ASGI application to wrap.
        rules: Rate limit rules to enforce.
        enabled: Whether rate limiting is active.
    """

    def __init__(
        self,
        app: Any,
        rules: list[RateLimitRule] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._rules = rules or DEFAULT_RULES
        self._store = RateLimitStore(self._rules)
        self._rules_by_name: dict[str, RateLimitRule] = {
            r.name: r for r in self._rules
        }

        logger.info(
            "Rate limiting %s — rules: %s",
            "enabled" if self._enabled else "disabled",
            [(r.name, r.max_requests, r.window_seconds) for r in self._rules],
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request through the rate limiter.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware / route handler.

        Returns:
            The response from the next handler, or a 429 response
            if the client has exceeded the rate limit.
        """
        # Skip if disabled
        if not self._enabled:
            return await call_next(request)

        path = request.url.path

        # Exempt paths from rate limiting
        if path in EXEMPT_PATHS:
            return await call_next(request)

        # Allow OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Identify the client
        client_id = self._get_client_id(request)

        # Determine which rule(s) to apply
        rule_names = self._get_applicable_rules(path)

        # Check each applicable rule
        for rule_name in rule_names:
            rule = self._rules_by_name.get(rule_name)
            if rule is None:
                continue

            allowed, remaining, retry_after = self._store.is_allowed(client_id, rule)

            if not allowed:
                logger.warning(
                    "Rate limit exceeded — client=%s, rule=%s, path=%s",
                    client_id,
                    rule_name,
                    path,
                )
                return self._build_429_response(rule, remaining, retry_after)

        # All rules passed — proceed with the request
        response = await call_next(request)

        # Add rate limit headers to the response
        for rule_name in rule_names:
            rule = self._rules_by_name.get(rule_name)
            if rule is None:
                continue
            _, remaining, _ = self._store.is_allowed(client_id, rule)
            response.headers[f"X-RateLimit-Limit-{rule_name}"] = str(rule.max_requests)
            response.headers[f"X-RateLimit-Remaining-{rule_name}"] = str(remaining)
            response.headers[f"X-RateLimit-Window-{rule_name}"] = f"{rule.window_seconds}s"

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_client_id(request: Request) -> str:
        """Determine the client identifier for rate limiting.

        Uses the ``X-API-Key`` header if present, otherwise falls
        back to the client IP address.

        Args:
            request: The incoming HTTP request.

        Returns:
            A string identifying the client.
        """
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain (original client)
            return f"ip:{forwarded.split(',')[0].strip()}"

        if request.client:
            return f"ip:{request.client.host}"

        return "ip:unknown"

    @staticmethod
    def _get_applicable_rules(path: str) -> list[str]:
        """Determine which rate limit rules apply to a given path.

        Args:
            path: The request URL path.

        Returns:
            A list of rule names that should be checked.
        """
        rules: list[str] = ["global"]  # Global rule always applies

        for prefix, rule_name in PATH_RULE_MAP.items():
            if path.startswith(prefix):
                rules.append(rule_name)

        return rules

    @staticmethod
    def _build_429_response(
        rule: RateLimitRule,
        remaining: int,
        retry_after: int,
    ) -> JSONResponse:
        """Build an HTTP 429 Too Many Requests response.

        Args:
            rule: The rate limit rule that was exceeded.
            remaining: Remaining requests in the window (should be 0).
            retry_after: Seconds until the client can retry.

        Returns:
            A JSON response with 429 status and rate limit headers.
        """
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded for rule '{rule.name}'. "
                    f"Maximum {rule.max_requests} requests per "
                    f"{rule.window_seconds} seconds."
                ),
                "status": "rate_limited",
                "rule": rule.name,
                "limit": rule.max_requests,
                "window_seconds": rule.window_seconds,
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rule.max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(time.time()) + retry_after),
            },
        )
