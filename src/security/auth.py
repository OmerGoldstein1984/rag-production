"""
Scoped API key authentication and server-side role-based authorization.

Threat model: callers present a bearer token (API key). We verify it in constant
time against configured keys, derive the caller's role from the key's scopes
(server-side -- never from a client header), and enforce least-privilege access
to endpoints via `require_scope(scope)` dependencies.

All secret handling avoids logging the raw key; only key_id and redacted hints
appear in logs/audit events.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import get_settings


# Map a role to the scopes it is allowed to use (least privilege).
_ROLE_SCOPES = {
    "guest": {"ask"},
    "employee": {"ask"},
    "manager": {"ask", "ingest"},
    "admin": {"ask", "ingest", "delete", "admin"},
}

# Scope combinations -> role (most privileged match wins).
_SCOPE_ROLES = [
    ({"admin"}, "admin"),
    ({"delete"}, "manager"),
    ({"ingest"}, "manager"),
    ({"ask"}, "employee"),
    (set(), "guest"),
]


def role_for_scopes(scopes: List[str]) -> str:
    """Map a key's scopes to the most privileged matching role."""
    scope_set = set(scopes)
    for required, role in _SCOPE_ROLES:
        if required.issubset(scope_set):
            return role
    return "guest"


def scope_allows(role: str, required_scope: str) -> bool:
    """True if `role` is permitted to perform an action requiring `required_scope`."""
    return required_scope in _ROLE_SCOPES.get(role, set())


@dataclass
class ApiKey:
    """One configured API key.

    WHAT: key_id (public id), key_hash (sha256 of secret), scopes (allowed
    actions), created_at, expires_at, revoked flag.
    WHY: Scoped keys let us rotate/revoke individual keys and enforce least
    privilege per key.
    """

    key_id: str
    key_hash: str
    scopes: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: Optional[str] = None
    revoked: bool = False

    def is_valid(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at:
            try:
                if datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc):
                    return False
            except ValueError:
                return False
        return True

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def load_api_keys() -> Dict[str, ApiKey]:
    """Load API keys from the API_KEYS env var ("key_id:secret,...").

    Returns a dict keyed by key_id. Raw secrets are never logged.
    """
    settings = get_settings()
    keys: Dict[str, ApiKey] = {}
    for entry in settings.api_keys or []:
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        key_id, secret = entry.split(":", 1)
        key_id = key_id.strip()
        secret = secret.strip()
        if not key_id or not secret:
            continue
        keys[key_id] = ApiKey(
            key_id=key_id,
            key_hash=_hash_secret(secret),
            scopes=["ask"],  # least privilege default; per-key scopes configurable later
        )
    return keys


class _KeyStore:
    """In-memory key store, reloaded on each verification so rotation/revocation
    takes effect without restart (small config, acceptable cost)."""

    def verify(self, key_id: str, secret: str) -> Optional[ApiKey]:
        key = load_api_keys().get(key_id)
        if key is None or not key.is_valid():
            return None
        expected = key.key_hash.encode("utf-8")
        provided = hashlib.sha256(secret.encode("utf-8")).hexdigest().encode("utf-8")
        if hmac.compare_digest(expected, provided):
            return key
        return None


_key_store = _KeyStore()
bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(status_code=401, detail=detail)


class AuthorizationError(HTTPException):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=403, detail=detail)


async def verify_api_key(
    authorization: Optional[str] = Header(None),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> ApiKey:
    """Verify a bearer API key and return the authenticated ApiKey.

    Fail-closed: when keys are configured / required and no valid key is
    presented, raise 401. The raw token is never logged.
    """
    settings = get_settings()
    # Require auth only when keys are configured; with no keys configured the
    # system is in permissive dev mode (callers are treated as guests).
    required = bool(settings.api_keys)

    # Extract the bearer token from either header style.
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):].strip()

    if not token:
        if required:
            raise AuthenticationError("Missing bearer API key")
        # Dev mode: no keys configured and not required -> unauthenticated guest.
        return ApiKey(key_id="dev-guest", key_hash="", scopes=["ask"])

    if not settings.api_keys:
        return ApiKey(key_id="dev-guest", key_hash="", scopes=["ask"])

    key: Optional[ApiKey] = None
    # Compound token "key_id:secret".
    if ":" in token:
        kid, secret = token.split(":", 1)
        key = _key_store.verify(kid.strip(), secret.strip())
    else:
        # Single-secret mode: try every configured key.
        for entry in settings.api_keys or []:
            entry = entry.strip()
            if ":" not in entry:
                continue
            kid, sec = entry.split(":", 1)
            candidate = _key_store.verify(kid.strip(), token.strip())
            if candidate is not None:
                key = candidate
                break

    if key is None:
        logging.getLogger(__name__).warning("Failed API key authentication attempt")
        raise AuthenticationError("Invalid API key")

    return key


def require_scope(scope: str):
    """Dependency factory: require the authenticated key to have `scope`."""

    def checker(key: ApiKey = Depends(verify_api_key)) -> ApiKey:
        if not key.has_scope(scope):
            raise AuthorizationError(f"Requires scope '{scope}'")
        return key

    return checker


def get_current_user(key: ApiKey = Depends(verify_api_key)) -> Dict:
    """Return a user-info dict derived server-side from the verified key.

    The X-Access-Level header is intentionally ignored: roles come only from the
    verified key's scopes, so a caller cannot self-assert admin.
    """
    role = role_for_scopes(key.scopes)
    return {
        "key_id": key.key_id,
        "role": role,
        "scopes": key.scopes,
        "access_level": {"guest": 0, "employee": 1, "manager": 2, "admin": 3}[role],
        "can_ingest": key.has_scope("ingest"),
        "can_delete": key.has_scope("delete"),
        "authenticated": True,
    }
