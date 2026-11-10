"""
JSON Web Token (JWT) authentication for the RAG Academy telecom QA system.

Implements HS256-signed JWT access tokens using only the Python standard library
(hmac, hashlib, base64, json, time, datetime) -- no external dependencies.

WHAT: JWT access tokens that carry scoped identity claims (key_id, scopes, role)
and expire after a configurable window. Replaces the single shared bearer token
with per-client, expiring JWT tokens for least-privilege access.

WHY: A single shared bearer token (API_TOKEN) is a security risk: once leaked it
cannot be revoked without changing it everywhere, and it grants the same access to
every caller. JWT tokens are per-client, time-limited, and scoped, so a compromised
token has limited blast window and automatic expiry.

WHERE IT SITS IN THE DATA FLOW: verify_jwt_token() is a FastAPI dependency that can
be used alongside or instead of verify_api_key(). The /auth/token endpoint in
src/api/main.py issues JWTs after validating API key credentials.

WHAT IT CALLS: src.config.get_settings (jwt_secret, jwt_algorithm, jwt_expiry_seconds),
standard library only (hmac, hashlib, base64, json, time, datetime, typing).

WHAT BREAKS IF REMOVED: No JWT authentication -- callers must use API key auth
(verify_api_key) instead, which works but lacks per-token expiry and rotation.

DESIGN:
- JWT = base64url(header) + "." + base64url(payload) + "." + base64url(signature)
- Header: {"alg": "HS256", "typ": "JWT"}
- Payload: {"sub": key_id, "scopes": [...], "exp": unix_ts, "iat": unix_ts}
- Signature: HMAC-SHA256(header + "." + payload, jwt_secret)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from src.config import get_settings

logger = logging.getLogger(__name__)


# --- base64url helpers (URL-safe base64 without padding) ----------------------

def _b64url_encode(data: bytes) -> str:
    """Base64url-encode bytes without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    """Base64url-decode a string, adding padding as needed."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


# --- JWT functions -----------------------------------------------------------

def _sign(header_b64: str, payload_b64: str, secret: str) -> str:
    """HMAC-SHA256 sign header.payload with secret, return base64url signature."""
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return _b64url_encode(signature)


def _verify_signature(header_b64: str, payload_b64: str, signature_b64: str, secret: str) -> bool:
    """Constant-time verify HMAC-SHA256 signature."""
    expected = _sign(header_b64, payload_b64, secret)
    return hmac.compare_digest(expected, signature_b64)


def _json_encode(obj: Dict[str, Any]) -> str:
    """Compact JSON encoding with sorted keys for deterministic output."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret: Optional[str] = None,
) -> str:
    """
    Create a JWT access token.

    WHAT: Builds a JWT with header {alg: HS256, typ: JWT} and payload containing
    the provided data plus iat (issued at) and exp (expiration) claims.

    WHY: Lets callers obtain a time-limited, scoped token that can be used for
    authenticated requests instead of the long-lived API key.

    INPUTS:
        data: Dict of claims to include (e.g. {"sub": "key1", "scopes": ["ask"]})
        expires_delta: Optional timedelta; defaults to jwt_expiry_seconds setting
        secret: Optional secret override; defaults to JWT_SECRET env var

    OUTPUTS: str -- the encoded JWT token string
    SIDE EFFECTS: None -- pure function
    FAILURE MODES: None for valid inputs
    """
    settings = get_settings()
    secret = secret if secret is not None else settings.jwt_secret
    if not secret:
        raise ValueError("JWT_SECRET must be configured to create tokens")

    now = int(time.time())
    if expires_delta is not None:
        expire = now + int(expires_delta.total_seconds())
    else:
        expire = now + settings.jwt_expiry_seconds

    payload = dict(data)
    payload["iat"] = now
    payload["exp"] = expire

    header_b64 = _b64url_encode(_json_encode({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload_b64 = _b64url_encode(_json_encode(payload).encode("utf-8"))
    signature_b64 = _sign(header_b64, payload_b64, secret)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_access_token(token: str, secret: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT access token and return its payload.

    WHAT: Validates the JWT signature (HMAC-SHA256) and expiration (exp claim).
    Returns the payload dict if valid, None if invalid or expired.

    WHY: Lets callers verify a JWT token without revealing why it failed
    (timing-safe comparison prevents oracle attacks).

    INPUTS:
        token: The JWT token string
        secret: Optional secret override; defaults to JWT_SECRET setting

    OUTPUTS: Optional[Dict] -- the decoded payload if valid, None otherwise
    SIDE EFFECTS: None -- pure function
    FAILURE MODES: Returns None for malformed, expired, or incorrectly signed tokens
    """
    settings = get_settings()
    secret = secret if secret is not None else settings.jwt_secret
    if not secret:
        return None

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Constant-time signature verification
        if not _verify_signature(header_b64, payload_b64, signature_b64, secret):
            return None

        payload_json = _b64url_decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)

        # Check expiration
        exp = payload.get("exp")
        if exp is not None and int(time.time()) > int(exp):
            return None

        return payload
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, AttributeError):
        return None


def get_current_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT access token and return user info dict.

    WHAT: Verifies the JWT token and maps its claims to a user-info dict
    compatible with the existing get_current_user() return shape.

    WHY: Lets callers authenticate with a JWT instead of a raw API key, while
    returning the same user-info shape that the rest of the system expects.

    INPUTS: token: The JWT token string
    OUTPUTS: Optional[Dict] -- user info dict or None if invalid
    SIDE EFFECTS: None -- pure function (calls get_settings)
    """
    payload = verify_access_token(token)
    if payload is None:
        return None

    key_id = payload.get("sub", "")
    scopes = payload.get("scopes", [])
    if not isinstance(scopes, list):
        scopes = []

    # Map scopes to role using the same logic as auth.py
    from src.security.auth import role_for_scopes

    role = role_for_scopes(scopes)

    return {
        "key_id": key_id,
        "role": role,
        "scopes": scopes,
        "access_level": {"guest": 0, "employee": 1, "manager": 2, "admin": 3}[role],
        "can_ingest": "ingest" in scopes or "admin" in scopes,
        "can_delete": "delete" in scopes or "admin" in scopes,
        "authenticated": True,
    }


__all__ = [
    "create_access_token",
    "verify_access_token",
    "get_current_user_from_token",
]