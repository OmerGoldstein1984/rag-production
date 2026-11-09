"""
NVIDIA API key pool with rotation, failover, and expiry tracking.

BUSINESS PROBLEM SOLVED: Lets the RAG system call NVIDIA NIM safely with multiple
API keys, rotating among them to spread load, automatically failing over when a
key returns 401/403 (auth failure), and tracking key expiry locally since NVIDIA
does not expose expiry dates over its API. Full keys are never logged or stored --
only a fingerprint (first4...last4) -- so credentials never leak into logs or the
registry file.

WHERE IT SITS IN THE DATA FLOW: Sits between settings (src/config.py) and the LLM
provider (providers/llm.py). The provider calls KeyPool.current_key() to obtain an
active key for each NVIDIA request, and calls mark_bad()/mark_expired() when a key
fails, so the next request uses a different key. The registry file
(data/key_registry.json, gitignored) holds local expiry metadata.

WHAT CALLS IT: providers/llm.py builds a KeyPool via KeyPool.from_env() and calls
current_key()/mark_bad()/mark_expired(). src/key_monitor.py uses the registry
helpers (load_registry/save_registry) but not the KeyPool class itself.

WHAT IT CALLS: standard library only (json, os, threading, dataclasses, datetime,
pathlib, typing). No network access in this module.

WHAT BREAKS IF REMOVED: providers/llm.py cannot obtain an NVIDIA key, so all
production (non-mock) generation fails. There would be no key rotation, no
failover on auth errors, and no expiry tracking -- a single compromised or expired
key would take down the whole LLM path.

Design rules (preserved):
- Multiple keys come from env only: NVIDIA_API_KEYS="key1,key2,key3" (or the
  single NVIDIA_API_KEY for back-compat). Full keys are NEVER logged or stored.
- We only ever persist/log a *fingerprint* (first 4 + last 4 chars).
- Round-robin selection spreads usage; on an auth failure (401/403) the key is
  marked bad and the next key is tried (bounded by pool size).
- Expiry is tracked in a local registry file (JSON) keyed by fingerprint,
  because NVIDIA keys do not expose their expiry over the API. The registry
  is gitignored — it is local operational metadata, not source.

# GAP: Key rotation & failover. Learners from web/automation often call a single
API key directly; here a pool rotates keys and isolates failures so one bad key
does not take down the service -- a production resilience pattern.
# GAP: Secret fingerprints vs full keys. We never log or store the full key,
only an irreversible first4...last4 fingerprint, so logs and the registry are
safe to inspect.
# BUSINESS: Protects uptime (failover when a key dies), cost (spread usage across
keys instead of burning one), and trust (credentials never appear in logs).
# INTERVIEW: Say it out loud -- "The key pool rotates keys round-robin, marks a
key bad on 401/403, and tries the next key, bounded by pool size; full keys are
never logged, only fingerprints."
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
    """Non-secret identifier for a key: first4...last4. Never reversible to the full key.

    WHAT: Turns an API key into a short, safe-to-log fingerprint: "ABCD...WXYZ" for
    keys longer than 12 chars, or "fp:****" for shorter ones.
    WHY: Lets us identify and track a key in logs, the registry, and error messages
    without ever exposing the full secret.
    INPUTS: key: str -- the full NVIDIA API key (never logged elsewhere).
    OUTPUTS: str fingerprint; example: "nvap...1234" or "fp:****".
    SIDE EFFECTS: None -- pure function.
    FAILURE MODES: None for normal input; an empty/None key yields "fp:****"
    (no crash).
    ALTERNATIVE: A cryptographic hash (SHA-256) of the key -- rejected because a
    hash is still reversible given the small key space, and first4...last4 is
    intentionally human-readable for ops while remaining non-reversible.
    # GAP: Secret redaction / fingerprinting. A learner may think logging "the key
    ID" is harmless; here we deliberately show only first4...last4 so a leaked log
    or registry file cannot be used to reconstruct the actual key.
    # BUSINESS: Protects credentials -- fingerprints in logs/registry are useless
    to an attacker, preventing key theft via log exfiltration.
    # INTERVIEW: Say it out loud -- "We fingerprint keys (first4...last4) and never
    log or store the full key, so logs and the registry stay safe."
    """
    k = (key or "").strip()                       # normalize: treat None as "", drop surrounding whitespace
    if len(k) < 12:
        return "fp:" + "*" * max(len(k), 4)       # too short to split meaningfully; mask it entirely
    return f"{k[:4]}...{k[-4:]}"                  # first 4 and last 4 chars only; middle is hidden


@dataclass
class KeyEntry:
    """One tracked API key: its fingerprint, status, optional expiry, and last error.

    WHAT: A lightweight record describing a single key in the pool -- its
    fingerprint (never the full key), whether it is active/bad/expired, an optional
    expiry date from the registry, and the last error seen for it.
    WHY: The pool needs per-key status to decide which keys are eligible for use
    and to surface expiry/health info through status().
    INPUTS: fingerprint (str), status (str, default "active"), expires_at
    (Optional[str], ISO date), last_error (str, default "").
    OUTPUTS: None (data holder); to_public_dict() returns a fingerprint-only dict
    safe for API responses, e.g. {"fingerprint": "nvap...1234", "status": "active",
    "expires_at": "2026-12-31", "last_error": ""}.
    SIDE EFFECTS: None.
    FAILURE MODES: None -- plain data.
    ALTERNATIVE: A dict or namedtuple -- rejected because a dataclass gives clear
    field names, defaults, and a generated __init__ with minimal boilerplate.
    # GAP: Data-holding objects vs plain dicts. Learners often reach for dicts;
    a dataclass documents the shape, provides defaults, and is self-documenting.
    # BUSINESS: Protects trust -- status() returns fingerprints only, never full
    keys, so the API response surface never leaks credentials.
    # INTERVIEW: Say it out loud -- "KeyEntry holds fingerprint-only metadata
    (status, expiry, last error); to_public_dict() strips out anything secret."
    """

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
    """Thread-safe pool of API keys with round-robin rotation and failover.

    WHAT: Holds a list of NVIDIA API keys plus per-key KeyEntry status, and hands
    out one active key at a time in round-robin order. When a key fails with 401/
    403 it is marked bad and the next key is tried, bounded by pool size.
    WHY: Production LLM traffic must not die because one key is revoked, expired,
    or rate-limited; rotating spreads load and failover keeps answers flowing.
    INPUTS: keys (List[str], full keys -- never logged), entries (List[KeyEntry]),
    _idx (current round-robin position), _lock (threading.Lock for thread safety).
    OUTPUTS: None (stateful object); current_key() returns a key string.
    SIDE EFFECTS: current_key()/peek_key()/rotate()/mark_bad()/mark_expired()
    mutate entries and _idx under the lock.
    FAILURE MODES: __post_init__ raises ValueError if no keys are provided.
    current_key()/peek_key()/rotate() raise RuntimeError when no active keys
    remain (all marked bad or expired).
    ALTERNATIVE: A simple round-robin iterator with no failover -- rejected
    because a dead key would then take down generation until manually replaced.
    # GAP: Thread safety (locking). A learner from single-threaded scripts may not
    realize an API server handles many requests concurrently; the _lock guarantees
    two requests never pick the same key slot or corrupt _idx at the same time.
    # GAP: Round-robin + bounded failover. Instead of trying all keys forever on
    every request, we advance one slot per successful call and stop after a full
    circle when all keys are bad -- bounding the work per request.
    # BUSINESS: Protects uptime (one dead key doesn't kill the LLM path) and cost
    (usage spread across keys avoids burning through a single key's quota).
    # INTERVIEW: Say it out loud -- "Round-robin key selection with a lock; on a
    401/403 we mark the key bad and try the next, bounded by pool size, raising
    only when every key is bad or expired."
    """

    keys: List[str] = field(default_factory=list)            # full API keys (never logged)
    entries: List[KeyEntry] = field(default_factory=list)    # per-key status mirroring keys
    _idx: int = 0                                            # round-robin cursor
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # serializes all pool access

    def __post_init__(self) -> None:
        """Validate and initialize the pool after dataclass construction.

        WHAT: Ensures at least one key was provided, and builds a KeyEntry for
        each key if none were supplied.
        WHY: A pool with zero keys is useless; fail fast at construction rather
        than at the first request.
        INPUTS: self (keys, entries).
        OUTPUTS: None (mutates self.entries).
        SIDE EFFECTS: Populates self.entries with KeyEntry objects derived from
        self.keys.
        FAILURE MODES: Raises ValueError("KeyPool requires at least one API key")
        when keys is empty.
        ALTERNATIVE: Allow an empty pool and return None from current_key() --
        rejected because calling code would then need None checks everywhere;
        crashing at construction is clearer.
        """
        if not self.keys:
            raise ValueError("KeyPool requires at least one API key")
        # Build one KeyEntry per key (fingerprint only) when none were provided.
        if not self.entries:
            self.entries = [KeyEntry(fingerprint=fingerprint(k)) for k in self.keys]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_env(cls, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> "KeyPool":
        """Build a pool from NVIDIA_API_KEYS (comma-separated) or NVIDIA_API_KEY.

        WHAT: Reads NVIDIA_API_KEYS (or the single-key NVIDIA_API_KEY for
        back-compat) from the environment, deduplicates while preserving order,
        constructs a KeyPool, and attaches expiry/status metadata from the local
        registry.
        WHY: Centralizes key sourcing in one place so providers don't read env
        vars directly, and deduplication prevents the same key from appearing
        twice in the rotation.
        INPUTS: registry_path (str|Path, default data/key_registry.json).
        OUTPUTS: A KeyPool instance.
        SIDE EFFECTS: Reads os.environ; reads the registry file from disk.
        FAILURE MODES: Raises ValueError if no NVIDIA keys are configured at all.
        ALTERNATIVE: Read keys from a secrets manager or config file -- rejected
        because env vars keep secrets out of git and match container deployment.
        # GAP: Deduplication preserving order. `list(dict.fromkeys(keys))` removes
        duplicate keys while keeping first-seen order, so the same key isn't
        assigned two rotation slots.
        # BUSINESS: Protects cost and uptime -- duplicate keys would waste a
        rotation slot and a missing key fails fast with a clear message.
        # INTERVIEW: Say it out loud -- "from_env reads comma-separated keys from
        NVIDIA_API_KEYS, dedupes them, and loads per-key expiry from the local
        registry."
        """
        raw = os.environ.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY") or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "No NVIDIA API keys configured. Set NVIDIA_API_KEYS='k1,k2,...' "
                "(comma-separated) or NVIDIA_API_KEY for a single key."
            )
        # Dedupe while preserving order so the same key doesn't get two slots.
        pool = cls(keys=list(dict.fromkeys(keys)))
        pool.apply_registry(Path(registry_path))
        return pool

    def apply_registry(self, path: Path) -> None:
        """Attach expiry/status metadata from the local registry (by fingerprint).

        WHAT: Reads data/key_registry.json and, for each registry entry whose
        fingerprint matches one of our keys, updates that key's expires_at and
        status. Missing or malformed registry files are ignored silently.
        WHY: Key expiry dates come from an out-of-band operational process (the
        key_monitor CLI / manual registration), not from NVIDIA's API; this
        merges that local metadata into the running pool.
        INPUTS: path (Path) to the registry JSON file.
        OUTPUTS: None (mutates self.entries in place).
        SIDE EFFECTs: Reads the registry file from disk; mutates KeyEntry objects.
        FAILURE MODES: A missing file returns early; malformed JSON or unreadable
        file is caught and ignored (the pool keeps running with in-memory
        defaults).
        ALTERNATIVE: Require a valid registry and crash if absent -- rejected
        because the registry is optional operational metadata; the system must
        start even before anyone has registered expiry dates.
        # GAP: Defensive file I/O. The registry may not exist yet (first run) or
        may be corrupt; we catch JSONDecodeError and OSError and continue rather
        than crashing the API on a missing optional file.
        # BUSINESS: Protects uptime -- a missing or corrupt registry file never
        prevents the RAG system from serving.
        # INTERVIEW: Say it out loud -- "apply_registry merges local expiry
        metadata into the pool, ignoring a missing or corrupt registry so startup
        never depends on it."
        """
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
        """Raw-index round-robin over entries; return index of an active entry or None.

        WHAT: Walks the entries circularly starting at `start`, returning the
        index of the first entry whose status is "active". When advance=True it
        moves the cursor past the selected key (so the next call picks a
        different key); when advance=False it just peeks without moving on.
        WHY: Round-robin spreads requests across keys; the advance flag lets
        current_key() consume a slot while peek_key()/current_entry() just look.
        INPUTS: start (int cursor), advance (bool).
        OUTPUTS: Optional[int] -- index of an active entry, or None if none are
        active.
        SIDE EFFECTS: Mutates self._idx when advance=True.
        FAILURE MODES: Returns None (no exception) when every entry is bad or
        expired; callers must handle None.
        ALTERNATIVE: Pick a random active key -- rejected because round-robin is
        deterministic and fairer for load spreading.
        # GAP: Bounded circular scan. The loop runs at most n steps (one full
        circle) -- it never spins forever, so a request costs O(pool size) even
        when every key is bad.
        # BUSINESS: Protects latency and uptime -- bounded work per request, and
        no key is starved forever because the cursor advances.
        # INTERVIEW: Say it out loud -- "_next_active scans the ring once for an
        active key, advancing the cursor only when we actually consume a key."
        """
        n = len(self.entries)
        if n == 0:
            return None
        begin = start % n
        for step in range(n):  # at most one full circle -- bounded work
            i = (begin + step) % n
            if self.entries[i].status == "active":
                if advance:
                    self._idx = (i + 1) % n   # move past this key for next time
                else:
                    self._idx = i             # peek: remember position without consuming
                return i
        return None

    def current_key(self) -> str:
        """Return an active key and advance round-robin to the next entry.

        WHAT: Picks the next active key in the ring, advances the cursor, and
        returns the full key string for the caller to use in an NVIDIA request.
        WHY: The provider needs a fresh key per call, and advancing ensures the
        next caller gets a different key, spreading load across the pool.
        INPUTS: None.
        OUTPUTS: str -- the full API key.
        SIDE EFFECTS: Mutates self._idx under the lock.
        FAILURE MODES: Raises RuntimeError when no active keys remain.
        ALTERNATIVE: Return a random key each time -- rejected because round-robin
        is deterministic, fairer, and easier to reason about in ops.
        # GAP: Concurrency on shared state. The lock serializes cursor movement
        so two concurrent requests never receive the same key or corrupt _idx.
        # BUSINESS: Protects uptime and cost -- spreading usage across keys
        avoids exhausting any single key's quota.
        # INTERVIEW: Say it out loud -- "current_key hands out the next active key
        and advances the cursor, raising only when every key is bad or expired."
        """
        with self._lock:
            i = self._next_active(self._idx, advance=True)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.keys[i]

    def peek_key(self) -> str:
        """Return the active key at the current position without advancing.

        WHAT: Like current_key() but leaves the cursor in place; used when the
        caller wants to inspect the current key without consuming a rotation slot.
        WHY: Lets code check which key would be used next without side effects.
        INPUTS: None.
        OUTPUTS: str -- the full API key.
        SIDE EFFECTS: None (cursor unchanged).
        FAILURE MODES: Raises RuntimeError when no active keys remain.
        ALTERNATIVE: Always advance and accept the side effect -- rejected because
        some callers genuinely need read-only inspection.
        """
        with self._lock:
            i = self._next_active(self._idx, advance=False)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.keys[i]

    def current_entry(self) -> KeyEntry:
        """Return the KeyEntry for the current active key without advancing.

        WHAT: Provides the full metadata (fingerprint, status, expires_at,
        last_error) for the key that current_key() would return next.
        WHY: Callers may want to inspect a key's health before using it.
        INPUTS: None.
        OUTPUTS: KeyEntry for the active key at the cursor.
        SIDE EFFECTS: None.
        FAILURE MODES: Raises RuntimeError when no active keys remain.
        ALTERNATIVE: Return the entry at _idx regardless of status -- rejected
        because an inactive entry would mislead the caller.
        """
        with self._lock:
            i = self._next_active(self._idx, advance=False)
            if i is None:
                raise RuntimeError("No active API keys in pool — all marked bad/expired.")
            return self.entries[i]

    def rotate(self) -> Optional[str]:
        """Advance to the next active key. Returns None only if no active keys remain.

        WHAT: Moves the cursor to the next active key and returns it, without
        being tied to a specific request.
        WHY: Used by mark_bad() to immediately hand the caller a replacement key
        after a failure, so the caller can retry without a separate lookup.
        INPUTS: None.
        OUTPUTS: Optional[str] -- the next active key, or None if none remain.
        SIDE EFFECTS: Mutates self._idx under the lock.
        FAILURE MODES: Returns None when all keys are bad or expired.
        ALTERNATIVE: Raise instead of returning None -- rejected because mark_bad()
        wants to degrade gracefully when the pool is exhausted.
        """
        with self._lock:
            i = self._next_active(self._idx, advance=True)
            if i is None:
                return None
            return self.keys[i]

    def mark_bad(self, key: str, error: str = "") -> Optional[str]:
        """Mark the given key bad and return the next usable key (or None).

        WHAT: Finds the entry matching the key's fingerprint, flips its status
        from "active" to "bad", records the last error (truncated to 200 chars),
        then returns the next usable key via rotate().
        WHY: When NVIDIA returns 401/403 for a key, we must stop using it and
        immediately hand the caller a working key so the request can retry.
        INPUTS: key (str, the full key -- only its fingerprint is used), error
        (str, optional diagnostic).
        OUTPUTS: Optional[str] -- next usable key, or None if the pool is now
        exhausted.
        SIDE EFFECTS: Mutates the matching KeyEntry under the lock; advances the
        cursor via rotate().
        FAILURE MODES: Returns None if no active keys remain after marking.
        ALTERNATIVE: Mark bad and return nothing, forcing a separate current_key()
        call -- rejected because returning a replacement key lets the caller retry
        in one step.
        # GAP: Failover after auth failure. Instead of propagating a 401 to the
        user, the pool transparently swaps to another key -- a production
        resilience pattern worth naming out loud in an interview.
        # BUSINESS: Protects uptime -- a revoked or rate-limited key does not
        turn into user-facing 500s.
        # INTERVIEW: Say it out loud -- "mark_bad flips the key to bad, records
        the error, and rotate() hands back the next usable key, or None if the
        pool is exhausted."
        """
        fp = fingerprint(key)
        with self._lock:
            for e in self.entries:
                if e.fingerprint == fp and e.status == "active":
                    e.status = "bad"
                    e.last_error = error[:200]  # keep error messages bounded
        return self.rotate()

    def mark_expired(self, key: str) -> None:
        """Mark the given key expired (no replacement returned)."""
        fp = fingerprint(key)
        with self._lock:
            for e in self.entries:
                if e.fingerprint == fp:
                    e.status = "expired"

    # -- introspection -----------------------------------------------------

    def status(self) -> List[dict]:
        """Return fingerprint-only public status for every entry.

        WHAT: Builds a list of public dicts (fingerprint, status, expires_at,
        last_error) for all entries -- safe to return over an API because it
        never includes full keys.
        WHY: Lets an operator or monitoring endpoint see pool health without
        leaking credentials.
        INPUTS: None.
        OUTPUTS: List[dict]; example:
        [{"fingerprint": "nvap...1234", "status": "active", "expires_at": "2026-12-31", "last_error": ""}].
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Return the raw KeyEntry objects -- rejected because they
        couple callers to the internal dataclass and could tempt exposing keys.
        # BUSINESS: Protects credentials -- status() is safe to expose because
        to_public_dict() strips everything but the fingerprint.
        # INTERVIEW: Say it out loud -- "status() returns fingerprint-only dicts
        so the API never leaks full keys."
        """
        with self._lock:
            return [e.to_public_dict() for e in self.entries]

    def active_count(self) -> int:
        """Count entries whose status is "active".

        WHAT: Returns how many keys are currently eligible for use.
        WHY: A quick health signal -- zero active keys means the LLM path is down.
        INPUTS: None.
        OUTPUTS: int.
        SIDE EFFECTS: None.
        FAILURE MODES: None.
        ALTERNATIVE: Compute on demand from status() -- rejected because a direct
        count is cheaper and clearer.
        """
        with self._lock:
            return sum(1 for e in self.entries if e.status == "active")


# -- registry helpers (used by key_monitor) --------------------------------

def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict:
    """Load the key registry JSON, or {"keys": []} if it does not exist.

    WHAT: Reads data/key_registry.json and returns its parsed contents.
    WHY: key_monitor and KeyPool.apply_registry need the local expiry metadata;
    treating a missing file as empty lets the system start before any key is
    registered.
    INPUTS: path (str|Path, default data/key_registry.json).
    OUTPUTS: dict; example {"keys": [{"fingerprint": "nvap...1234", "expires_at":
    "2026-12-31", ...}]} or {"keys": []}.
    SIDE EFFECTS: Reads the registry file from disk.
    FAILURE MODES: A missing file returns {"keys": []}; a present but corrupt
    file raises json.JSONDecodeError (callers that need tolerance wrap it, e.g.
    KeyPool.apply_registry at src/key_pool.py:235).
    ALTERNATIVE: Crash when the file is absent -- rejected because the registry
    is optional until an operator registers expiry dates.
    # GAP: Defensive defaults for optional files. Missing config/registry files
    return empty structures rather than crashing, so startup isn't gated on
    optional operational metadata.
    """
    p = Path(path)
    if not p.exists():
        return {"keys": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(entries: List[dict], path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
    """Persist the key registry JSON, creating parent directories as needed.

    WHAT: Writes {"keys": entries} to the registry file, pretty-printed, creating
    the data/ directory if it does not exist.
    WHY: key_monitor.register_key() uses this to record a key's declared expiry;
    the directory creation means the first registration works on a fresh system.
    INPUTS: entries (List[dict]), path (str|Path).
    OUTPUTS: None.
    SIDE EFFECTS: Creates parent directories and writes the registry file to disk.
    FAILURE MODES: Raises OSError if the directory cannot be created or the file
    cannot be written (e.g. permissions).
    ALTERNATIVE: Append to a log file -- rejected because JSON lets us load,
    update, and rewrite the whole registry atomically enough for a single-writer
    ops tool.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"keys": entries}, indent=2), encoding="utf-8")


def days_until_expiry(expires_at: str, now: Optional[datetime] = None) -> int:
    """Days until expiry (negative if already expired).

    WHAT: Parses an ISO expiry date and returns the whole-day difference versus
    now (UTC).
    WHY: key_monitor.expiring_entries() uses this to decide which keys are due
    for an alert (expires within warn_days, or already expired).
    INPUTS: expires_at (str, ISO date), now (Optional[datetime], defaults to now
    in UTC -- injectable for tests).
    OUTPUTS: int days; negative means already expired.
    SIDE EFFECTS: None (pure function).
    FAILURE MODES: Raises ValueError if expires_at is not parseable by
    datetime.fromisoformat.
    ALTERNATIVE: Return a timedelta -- rejected because callers compare against
    an integer warn_days threshold, so whole days are more convenient.
    # GAP: Timezone-aware date math. Naive datetimes are treated as UTC to avoid
    silent mismatches between local time and expiry dates stored in UTC.
    # BUSINESS: Protects against surprise key expiry -- accurate day counts mean
    alerts fire before a key actually stops working.
    # INTERVIEW: Say it out loud -- "days_until_expiry returns whole days until
    expiry (negative if expired), treating naive dates as UTC."
    """
    now = now or datetime.now(timezone.utc)
    exp = datetime.fromisoformat(expires_at)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return (exp - now).days
