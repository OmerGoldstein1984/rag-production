"""
tests/conftest.py - Shared pytest fixtures and test isolation.

Resets module-level global state in src.api.main between tests so that
tests don't leak rate-limiter counters, query cache, or pipeline state
into each other.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import src.api.main as main  # noqa: E402
from src.security.rate_limit import RateLimiter, DEFAULT_RULES  # noqa: E402


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset module-level globals before each test to prevent cross-test
    state leakage (rate limiter counters, query cache, pipeline state)."""
    # Save current state (in case a test monkeypatches and wants undo)
    main.rate_limiter = RateLimiter(rules=dict(DEFAULT_RULES), max_clients=1000)
    main.query_cache = main.QueryCache(max_size=200)
    main.feedback_store = []
    main.rag_pipeline = None
    yield
    # Teardown: reset again so a test that modified state doesn't leak
    main.rate_limiter = RateLimiter(rules=dict(DEFAULT_RULES), max_clients=1000)
    main.query_cache = main.QueryCache(max_size=200)
    main.feedback_store = []
    main.rag_pipeline = None
