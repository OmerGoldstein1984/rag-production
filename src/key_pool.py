"""
NVIDIA API key pool with rotation, failover, and expiry tracking.

Design rules:
- Multiple keys come from env only: NVIDIA_API_KEYS="key1,key2,key3" (or the
  single NVIDIA_API_KEY for back-compat). Full keys are NEVER logged or stored.
- We only ever persist/log a *fingerprint* (first 4 + last 4 chars).
- Round-robin selection spreads usage; on an auth failure (401/403) the key is
  marked bad and the next key is tried (bounded by pool size).
- Expiry is tracked in a local registry file (JSON) keyed by fingerprint,
  because NVIDIA keys do not expose their expiry over the API. The registry
  is gitignored — it is local operational metadata, not source.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_REGISTRY_PATH = "data/key_registry.json"


def fingerprint(key: str) -> str:
    """Non-secret identifier for a key: first4...last4. Never reversible to the full key."""
    k = (key or "").strip()
    if len(k) < 12:
        return "fp:" + "*" * max(len(k), 4)
    return f"{k[:4]}...{k[-4:]}"


@dataclass
class KeyEntry:
    fingerprint: str
    status: str = "active"  # active | bad | expired
    expires_at: Optional[str] = None  # ISO date, from registry
    last_error: str = ""

    def to_public_dict(self) -> dict:
        # Never expose the full key — fingerprint only.
        return {
            "fingerprint": self.fingerprint,
            "status": self.status,
            "expires_at": self.expires_at,
            "last_error": self.last_error,
        }


@dataclass
class KeyPool:
    """Thread-safe pool of API keys with round-robin rotation and failover."""

    keys: List[str] = field(default_factory=list)
    entries: List[KeyEntry] = field(default_factory=list)
    _idx: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("KeyPool requires at least one API key")
        if not self.entries:
            self.entries = [KeyEntry(fingerprint=fingerprint(k)) for k in self.keys]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_env(cls, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> "KeyPool":
        """Build a pool from NVIDIA_API_KEYS (comma-separated) or NVIDIA_API_KEY."""
        raw = os.environ.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY") or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "No NVIDIA API keys configured. Set NVIDIA_API_KEYS='k1,k2,...' "
                "(comma-separated) or NVIDIA_API_KEY for a single key."
            )
        pool = cls(keys=list(dict.fromkeys(keys)))  # dedupe, keep order
        pool.apply_registry(Path(registry_path))
        return pool

    def apply_registry(self, path: Path) -> None:
        """Attach expiry/status metadata from the local registry (by fingerprint)."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for rec in data.get("keys", []):
            fp = rec.get("fingerprint")
            for e in self.entries:
                if e.fingerprint == fp:
                    e.expires_at = rec.get("expires_at")
                    if rec.get("status"):
                        e.status = rec["status"]

    # -- selection ---------------------------------------------------------

    def _next_active(self, start: int, advance: bool) -> Optional[int]:
        """Raw-index round-robin over entries; return index of an active entry or None."""
        n = len(self.entries)
        if n == 0:
            return None
        begin = start % n
        for step in range(n):
            i = (begin + step) % n
            if self.entries[i].status == "active":
                if advance:
                    self._idx = (i + 1) % n
                else:
                    self._idx = i
                return i
        return None

    def current_key(self) -> str:
        """Return an active key and advance round-robin to the next entry."""
        with self._lock:
            i = self._next_active(self._idx, advance=True)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.keys[i]

    def peek_key(self) -> str:
        """Return the active key at the current position without advancing."""
        with self._lock:
            i = self._next_active(self._idx, advance=False)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.keys[i]

    def current_entry(self) -> KeyEntry:
        with self._lock:
            i = self._next_active(self._idx, advance=False)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.entries[i]

    def rotate(self) -> Optional[str]:
        """Advance to the next active key. Returns None only if no active keys remain."""
        with self._lock:
            i = self._next_active(self._idx, advance=True)
            if i is None:
                return None
            return self.keys[i]

    def mark_bad(self, key: str, error: str = "") -> Optional[str]:
        """Mark the given key bad and return the next usable key (or None)."""
        fp = fingerprint(key)
        with self._lock:
            for e in self.entries:
                if e.fingerprint == fp and e.status == "active":
                    e.status = "bad"
                    e.last_error = error[:200]
        return self.rotate()

    def mark_expired(self, key: str) -> None:
        fp = fingerprint(key)
        with self._lock:
            for e in self.entries:
                if e.fingerprint == fp:
                    e.status = "expired"

    # -- introspection -----------------------------------------------------

    def status(self) -> List[dict]:
        with self._lock:
            return [e.to_public_dict() for e in self.entries]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for e in self.entries if e.status == "active")


# -- registry helpers (used by key_monitor) --------------------------------

def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {"keys": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(entries: List[dict], path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"keys": entries}, indent=2), encoding="utf-8")


def days_until_expiry(expires_at: str, now: Optional[datetime] = None) -> int:
    """Days until expiry (negative if already expired)."""
    now = now or datetime.now(timezone.utc)
    exp = datetime.fromisoformat(expires_at)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return (exp - now).days
