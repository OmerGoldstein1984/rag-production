"""
src.security.rate_limit - distributed sliding-window rate limiter.

THREAT: The original in-memory RateLimiter resets on restart, is not shared
across uvicorn workers (each worker multiplies the limit by worker count), has
no client eviction (unbounded memory), no per-endpoint limits, and no
Retry-After header on 429.

CONTROL:
- Sliding-window counter keyed by (client_id, endpoint).
- Redis backend when REDIS_URL is configured (shared across workers, survives
  restarts); in-memory fallback otherwise.
- Per-endpoint limits: stricter on /ingest and /agent/ask, looser on /ask/search.
- Retry-After header on 429.
- Client eviction: oldest clients evicted when the client table exceeds a cap.
- Thread-safe.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LimitRule:
    max_requests: int
    window_seconds: int


# Default rules per endpoint path. Clients get the endpoint-specific limit.
DEFAULT_RULES: Dict[str, LimitRule] = {
    "/ask": LimitRule(max_requests=30, window_seconds=60),
    "/ask/stream": LimitRule(max_requests=30, window_seconds=60),
    "/search": LimitRule(max_requests=30, window_seconds=60),
    "/ingest": LimitRule(max_requests=5, window_seconds=60),
    "/documents/upload": LimitRule(max_requests=5, window_seconds=60),
    "/documents/reindex": LimitRule(max_requests=3, window_seconds=60),
    "/documents/": LimitRule(max_requests=10, window_seconds=60),
    "/agent/ask": LimitRule(max_requests=10, window_seconds=60),
    "/feedback": LimitRule(max_requests=20, window_seconds=60),
    # Default catch-all.
    "_default": LimitRule(max_requests=30, window_seconds=60),
}

MAX_CLIENTS = 10_000  # bound the in-memory client table


class RateLimitExceeded(Exception):
    """Raised when a client is over its limit. Carries retry_after seconds."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class RateLimiter:
    """Sliding-window rate limiter.

    Use Redis when available (pass a redis.Redis client); otherwise falls back to
    a thread-safe in-memory store (per-process only).
    """

    def __init__(self, rules: Optional[Dict[str, LimitRule]] = None,
                 redis_client=None, max_clients: int = MAX_CLIENTS) -> None:
        self.rules = dict(rules) if rules else dict(DEFAULT_RULES)
        self._redis = redis_client
        self._lock = threading.Lock()
        # client_id -> {endpoint -> list of timestamps}
        self._requests: Dict[str, Dict[str, List[float]]] = OrderedDict()
        self.max_clients = max_clients

    # --- rule selection ------------------------------------------------------

    def _rule_for(self, endpoint: str) -> LimitRule:
        rule = self.rules.get(endpoint)
        if rule is None:
            rule = self.rules.get("_default")
        return rule or LimitRule(max_requests=30, window_seconds=60)

    # --- in-memory backend ---------------------------------------------------

    def _evict_if_needed(self) -> None:
        while len(self._requests) > self.max_clients:
            self._requests.popitem(last=False)

    def _prune(self, timestamps: List[float], now: float, window: int) -> List[float]:
        cutoff = now - window
        return [t for t in timestamps if t >= cutoff]

    def is_allowed_memory(self, client_id: str, endpoint: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        rule = self._rule_for(endpoint)
        now = time.time()
        with self._lock:
            by_endpoint = self._requests.get(client_id)
            if by_endpoint is None:
                by_endpoint = {}
                self._requests[client_id] = by_endpoint
                self._evict_if_needed()
            timestamps = self._prune(by_endpoint.get(endpoint, []), now, rule.window_seconds)
            if len(timestamps) >= rule.max_requests:
                # compute retry_after from oldest timestamp in window
                if timestamps:
                    retry_after = int(timestamps[0] + rule.window_seconds - now) + 1
                else:
                    retry_after = rule.window_seconds
                # keep the pruned list (do not add current request)
                by_endpoint[endpoint] = timestamps
                return False, max(1, retry_after)
            timestamps.append(now)
            by_endpoint[endpoint] = timestamps
            return True, 0

    # --- public API ----------------------------------------------------------

    def is_allowed(self, client_id: str, endpoint: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if self._redis is not None:
            return self._is_allowed_redis(client_id, endpoint)
        return self.is_allowed_memory(client_id, endpoint)

    def _is_allowed_redis(self, client_id: str, endpoint: str) -> tuple[bool, int]:
        import redis as _redis  # type: ignore
        rule = self._rule_for(endpoint)
        now = time.time()
        window_start = now - rule.window_seconds
        key = f"ratelimit:{client_id}:{endpoint}"
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, rule.window_seconds + 1)
        results = pipe.execute()
        count = results[1]
        if count >= rule.max_requests:
            # Remove the just-added entry since we're rejecting.
            try:
                self._redis.zrem(key, str(now))
            except Exception:
                pass
            # retry_after from the oldest member's score
            oldest = self._redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = int(oldest[0][1] + rule.window_seconds - now) + 1
            else:
                retry_after = rule.window_seconds
            return False, max(1, retry_after)
        return True, 0

    def get_stats(self) -> dict:
        with self._lock:
            total_blocked = 0
            active_clients = len(self._requests)
            for by_endpoint in self._requests.values():
                for timestamps in by_endpoint.values():
                    total_blocked += max(0, len(timestamps) - self._rule_for("").max_requests)
        return {
            "max_clients": self.max_clients,
            "active_clients": active_clients,
            "rules": {k: {"max_requests": v.max_requests, "window_seconds": v.window_seconds}
                      for k, v in self.rules.items()},
        }


# Module-level singleton (wired from settings at startup).
rate_limiter: Optional[RateLimiter] = None


def init_rate_limiter(settings=None, redis_client=None) -> RateLimiter:
    global rate_limiter
    if settings is None:
        from src.config import get_settings as _gs
        settings = _gs()
    rules = dict(DEFAULT_RULES)
    # Allow env overrides via settings attributes if present.
    rate_limiter = RateLimiter(rules=rules, redis_client=redis_client,
                               max_clients=getattr(settings, "max_clients", MAX_CLIENTS))
    return rate_limiter
