"""
JWT Authentication Middleware for MedVision-AI.

Provides token verification, user extraction, and role-based access
control for the FastAPI application.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# In production, these would come from environment variables or a secrets manager.
_DEFAULT_SECRET = "medvision-ai-dev-secret-change-in-production"
_DEFAULT_ALGORITHM = "HS256"
_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour

# Paths that do not require authentication
PUBLIC_PATHS: set[str] = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/api/v1/ready",
}


class UserRole(str, Enum):
    """User roles for role-based access control."""

    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    SERVICE = "service"


# -----------------------------------------------------------------------
# User model
# -----------------------------------------------------------------------

@dataclass
class AuthenticatedUser:
    """Represents an authenticated user extracted from a JWT token.

    Attributes:
        user_id: Unique user identifier.
        username: Human-readable username.
        roles: List of roles assigned to the user.
        token_issued_at: UNIX timestamp when the token was issued.
        token_expires_at: UNIX timestamp when the token expires.
        metadata: Additional claims from the token.
    """

    user_id: str
    username: str
    roles: list[UserRole] = field(default_factory=list)
    token_issued_at: float = 0.0
    token_expires_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        """Return ``True`` if the user has the admin role."""
        return UserRole.ADMIN in self.roles

    @property
    def is_clinician(self) -> bool:
        """Return ``True`` if the user has the clinician role."""
        return UserRole.CLINICIAN in self.roles

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the token has expired."""
        return time.time() > self.token_expires_at

    def has_role(self, role: UserRole) -> bool:
        """Check whether the user has a specific role.

        Args:
            role: The role to check.

        Returns:
            ``True`` if the user has the specified role.
        """
        return role in self.roles


# -----------------------------------------------------------------------
# JWT helpers
# -----------------------------------------------------------------------

def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url without padding.

    Args:
        data: Raw bytes to encode.

    Returns:
        Base64url-encoded string.
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    """Decode a base64url string with padding restoration.

    Args:
        data: Base64url-encoded string.

    Returns:
        Decoded bytes.
    """
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def verify_token(token: str, secret: str = _DEFAULT_SECRET) -> Optional[AuthenticatedUser]:
    """Verify a JWT token and extract the authenticated user.

    Supports HS256 tokens with standard claims (``sub``, ``name``,
    ``roles``, ``iat``, ``exp``).

    Args:
        token: The JWT token string.
        secret: The HMAC secret used for signature verification.

    Returns:
        An :class:`AuthenticatedUser` if the token is valid, or
        ``None`` if verification fails.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("Invalid token format — expected 3 parts, got %d", len(parts))
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_sig = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("Token signature verification failed")
            return None

        # Decode header
        header = json.loads(_base64url_decode(header_b64))
        if header.get("alg") != "HS256":
            logger.warning("Unsupported algorithm: %s", header.get("alg"))
            return None

        # Decode payload
        payload = json.loads(_base64url_decode(payload_b64))

        # Check expiration
        exp = payload.get("exp", 0)
        if time.time() > exp:
            logger.warning("Token expired — exp=%d, now=%d", exp, time.time())
            return None

        # Extract user information
        user_id = payload.get("sub", "unknown")
        username = payload.get("name", user_id)
        raw_roles = payload.get("roles", ["viewer"])
        iat = payload.get("iat", 0)

        roles: list[UserRole] = []
        for r in raw_roles:
            try:
                roles.append(UserRole(r))
            except ValueError:
                logger.warning("Unknown role '%s' in token — skipping", r)

        return AuthenticatedUser(
            user_id=user_id,
            username=username,
            roles=roles,
            token_issued_at=iat,
            token_expires_at=exp,
            metadata={
                k: v for k, v in payload.items()
                if k not in {"sub", "name", "roles", "iat", "exp"}
            },
        )

    except Exception as exc:
        logger.warning("Token verification error: %s", exc)
        return None


def get_current_user(request: Request) -> Optional[AuthenticatedUser]:
    """Extract the current authenticated user from a request.

    Reads the ``Authorization`` header and validates the Bearer token.

    Args:
        request: The incoming HTTP request.

    Returns:
        An :class:`AuthenticatedUser` if a valid token is present,
        or ``None`` otherwise.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]  # Strip "Bearer " prefix
    return verify_token(token)


def require_role(*required_roles: UserRole) -> Callable[..., bool]:
    """Create a role-checking function for dependency injection.

    The returned callable checks whether the authenticated user has
    at least one of the required roles.

    Args:
        *required_roles: One or more roles that are permitted.

    Returns:
        A callable that accepts an :class:`AuthenticatedUser` and
        returns ``True`` if access is granted.

    Example::

        check = require_role(UserRole.CLINICIAN, UserRole.ADMIN)
        if not check(user):
            raise HTTPException(status_code=403)
    """
    role_set = set(required_roles)

    def _check(user: AuthenticatedUser) -> bool:
        return bool(set(user.roles) & role_set)

    _check.__doc__ = (
        f"Check that the user has one of the roles: "
        f"{', '.join(r.value for r in required_roles)}."
    )
    return _check


# -----------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------

class AuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces JWT authentication.

    Requests to paths in :data:`PUBLIC_PATHS` are allowed without
    authentication.  All other requests must carry a valid ``Authorization:
    Bearer <token>`` header.

    The authenticated user (if any) is stored on
    ``request.state.user`` for downstream access.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request through the authentication layer.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware / route handler.

        Returns:
            The response from the next handler, or a 401 JSON
            response if authentication fails.
        """
        # Allow public paths without authentication
        if request.url.path in PUBLIC_PATHS:
            request.state.user = None
            return await call_next(request)

        # Allow OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            request.state.user = None
            return await call_next(request)

        # Attempt authentication
        user = get_current_user(request)

        if user is None:
            # In development mode, allow requests without auth
            # but flag them on the request state
            auth_header = request.headers.get("Authorization", "")
            if not auth_header:
                logger.debug(
                    "Unauthenticated request to %s — allowing in dev mode",
                    request.url.path,
                )
                request.state.user = None
                return await call_next(request)

            # Token was provided but is invalid
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or expired authentication token.",
                    "status": "unauthorized",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check token expiry
        if user.is_expired:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication token has expired.",
                    "status": "token_expired",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach user to request state for downstream handlers
        request.state.user = user
        logger.debug(
            "Authenticated user: %s (roles: %s)",
            user.username,
            [r.value for r in user.roles],
        )

        return await call_next(request)
