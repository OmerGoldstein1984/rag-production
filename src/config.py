"""
Environment-driven configuration for the RAG system.

BUSINESS PROBLEM SOLVED: Centralizes every runtime setting (LLM provider choice,
API keys, base URL, model, timeouts, retries, SMTP alerting, API bearer token) so
that secrets and tunable values live exclusively in environment variables -- never
in source code. This keeps credentials out of git and makes the same code deployable
across dev, test, and prod by changing only the environment (twelve-factor style).

WHERE IT SITS IN THE DATA FLOW: The foundation layer. get_settings() (src/config.py:127)
is read once at startup and threaded into every component that needs it:
src/key_pool.py (API keys), src/key_monitor.py (registry/expiry tool),
src/observability.py (log level), providers/llm.py (provider factory), and
src/api/main.py (startup/auth). Per docs/learn/_plan.md:28 it is the first
dependency -- no module can start without the settings this file defines.

WHAT CALLS IT: src/api/main.py, providers/llm.py, src/key_monitor.py,
src/observability.py, plus the test suite. Practically every runtime module.

WHAT IT CALLS: Only the standard library (os, dataclasses, typing). No network,
no filesystem beyond reading env vars. It does not call into any project module.

WHAT BREAKS IF REMOVED: The system cannot start. Each component reads its keys,
timeouts, base URLs, and log level from Settings; without get_settings() there is
no validated source for NVIDIA_API_KEY, LLM_PROVIDER, timeouts, or the API token,
so providers fail to construct, the pool has no keys, and the API has no auth
token to enforce.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"


class MissingConfigError(RuntimeError):
    """Raised when required configuration is absent from the environment.

    WHAT: An exception signalling that a value the system must have (for example
    NVIDIA_API_KEY when LLM_PROVIDER=nvidia) is not present in the environment.
    WHY: Fail-fast -- it is far better to crash at startup with a clear, actionable
    message than to fail mysteriously mid-request when an API call is attempted
    without credentials.
    INPUTS: A descriptive message string passed to the RuntimeError base class.
    OUTPUTS: None -- raising this aborts the current call stack.
    SIDE EFFECTS: If uncaught at startup (src/api/main.py imports get_settings
    before serving), the process exits and the API never comes up.
    FAILURE MODES: A deployment missing NVIDIA_API_KEY/NVIDIA_API_KEYS crashes
    immediately with this error instead of returning 500s on every request.
    ALTERNATIVE: Return a dummy Settings object and let NVIDIA calls fail later --
    rejected, because silent failure wastes user time and hides misconfiguration.
    # GAP: Fail-fast validation at startup versus lazy validation. Learners from
    web/automation backgrounds often expect code to "just run with defaults";
    here a missing required secret is treated as a deployment bug, not a runtime
    surprise, because a RAG system without an LLM key cannot answer anything.
    # BUSINESS: Protects uptime and user trust -- a clear startup error beats
    intermittent 500s; also prevents accidentally shipping without credentials.
    # INTERVIEW: Say it out loud -- "Required secrets are validated at startup;
    if NVIDIA_API_KEY is missing we crash fast with a clear message rather than
    failing silently per request."
    """


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables.

    WHAT: An immutable bundle of every runtime configuration value for the RAG
    system: LLM provider, NVIDIA keys/base URL/model/timeouts/retries, request
    timeout, log level, SMTP alerting credentials, key-expiry warning window, and
    the optional API bearer token.
    WHY: Immutability (frozen=True) guarantees that once settings are loaded at
    startup no module can silently mutate them mid-request, which keeps behavior
    predictable across threads and requests.
    INPUTS: None (all values are read from os.environ inside get_settings(),
    src/config.py:127).
    OUTPUTS: A Settings instance; example: Settings(llm_provider="mock",
    nvidia_api_keys=(), log_level="INFO").
    SIDE EFFECTS: None -- pure data holder; all side effects happen in get_settings()
    when env vars are read and validated.
    FAILURE MODES: A field with the wrong type (e.g. SMTP_PORT not an integer)
    raises ValueError inside get_settings() (src/config.py:193-212), not here.
    ALTERNATIVE: A mutable dict or a TOML/YAML config file -- rejected because
    env vars keep secrets out of git and are the standard for containerized
    deployment (docs/learn/_plan.md:13).
    # GAP: Immutable configuration. A learner used to editing a config object at
    runtime may not realize frozen dataclasses prevent accidental mutation, which
    matters in a multi-threaded API where one request must not change another's
    timeouts or log level.
    # BUSINESS: Protects uptime and consistency -- immutable settings mean every
    request sees the same timeouts, keys, and log level, avoiding Heisenbugs.
    # INTERVIEW: Say it out loud -- "Settings is frozen so no module can change
    config mid-flight; all values come from env vars at startup."
    """

    llm_provider: str = "mock"                       # "mock" (dev/tests) or "nvidia" (production)
    nvidia_api_key: Optional[str] = None             # single key, kept for backward compatibility
    nvidia_api_keys: tuple = ()                      # multi-key pool from NVIDIA_API_KEYS (comma-separated)
    nvidia_base_url: str = DEFAULT_NVIDIA_BASE_URL   # NVIDIA NIM endpoint
    nvidia_model: str = DEFAULT_NVIDIA_MODEL         # model name used for generation
    nvidia_timeout_s: float = 60.0                   # per-call timeout to NVIDIA (seconds)
    nvidia_max_retries: int = 3                      # max retries on transient NVIDIA failures
    request_timeout_s: float = 60.0                  # HTTP request timeout used elsewhere
    log_level: str = "INFO"                          # logging verbosity
    # key-expiry alerting (email) -- SMTP config for key expiry alerts
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True
    alert_email_to: str = ""                         # recipient of expiry alerts
    key_warn_days: int = 14                          # days before expiry to start warning
    # Optional API bearer token. If set, mutating endpoints require it.
    api_token: str = ""                              # auth token for mutating API endpoints

    @property
    def is_real_provider(self) -> bool:
        """True when a real LLM provider (nvidia) is configured, not mock."""
        return self.llm_provider != "mock"


def get_settings() -> Settings:
    """Load settings from env vars, validating required secrets.

    WHAT: Reads all configuration from os.environ, parses and type-converts each
    value, validates that required secrets are present, and returns a frozen
    Settings instance.
    WHY: Environment variables are the single source of truth, so no secret is
    ever committed to source code and the same image can be deployed to any
    environment by changing env vars alone.
    INPUTS: None (reads os.environ directly).
    OUTPUTS: A Settings instance; example: Settings(llm_provider="mock",
    nvidia_api_keys=(), nvidia_base_url="https://integrate.api.nvidia.com/v1",
    log_level="INFO").
    SIDE EFFECTS: Raises MissingConfigError (src/config.py:40) if LLM_PROVIDER=nvidia
    with no key, or if an unsupported provider is named; raises ValueError if a
    numeric env var (ports, timeouts) cannot be parsed.
    FAILURE MODES: Missing NVIDIA key with provider=nvidia -> crash at startup
    with a clear message (src/config.py:175-189). Bad integer in SMTP_PORT or
    NVIDIA_TIMEOUT_S -> ValueError crash. Unsupported provider -> crash.
    ALTERNATIVE: Read a config file (YAML/TOML) or a secrets manager -- rejected
    because env vars are simpler, keep secrets out of git, and match container
    deployment (docs/learn/_plan.md:13).
    # GAP: Env-var-only configuration (twelve-factor). Learners from web/automation
    often expect a checked-in config file; here there is none (.env is gitignored),
    so every deployment must export the right vars or the app crashes fast.
    # GAP: Type conversion of env strings (float/int/bool) -- every env var is a
    string, so get_settings() casts timeouts, ports, and the TLS flag, and a bad
    value crashes at startup rather than producing weird behavior later.
    # BUSINESS: Protects against credential leakage (secrets never in code) and
    against starting a system that cannot answer questions (missing LLM key).
    # INTERVIEW: Say it out loud -- "All config comes from env vars; required
    secrets are validated at startup with clear errors, and everything is
    immutable after load."
    """
    # --- Read and type-convert every env var. Each os.environ.get returns a
    # string (or None), so we strip, lower-case, or cast to float/int/bool as
    # needed. Defaults are used when the operator did not set the variable. ---
    provider = os.environ.get("LLM_PROVIDER", "mock").strip().lower()
    api_key = os.environ.get("NVIDIA_API_KEY")
    api_keys_raw = os.environ.get("NVIDIA_API_KEYS", "")
    api_keys = tuple(k.strip() for k in api_keys_raw.split(",") if k.strip())
    base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).strip()
    model = os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL).strip()
    timeout = float(os.environ.get("NVIDIA_TIMEOUT_S", "60"))
    max_retries = int(os.environ.get("NVIDIA_MAX_RETRIES", "3"))
    req_timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "60"))
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # --- Fail-fast validation: refuse to start if required secrets are missing. ---
    if provider == "nvidia" and not (api_key and api_key.strip()) and not api_keys:
        raise MissingConfigError(
            "LLM_PROVIDER=nvidia but no NVIDIA key is set. "
            "Export NVIDIA_API_KEYS='key1,key2,...' (comma-separated, recommended) or "
            "NVIDIA_API_KEY for a single key — never in source code. "
            "For local development without a key, set LLM_PROVIDER=mock explicitly."
        )

    # Reject any provider other than the two we support (mistyped providers fail
    # loud instead of silently behaving like "mock").
    if provider not in ("nvidia", "mock"):
        raise MissingConfigError(
            f"Unsupported LLM_PROVIDER={provider!r}. Supported: 'nvidia', 'mock'."
        )

    # Assemble the frozen Settings, casting SMTP and key-warn env vars to their
    # proper types (int/bool). api_token gates mutating API endpoints.
    return Settings(
        llm_provider=provider,
        nvidia_api_key=api_key.strip() if api_key else None,
        nvidia_api_keys=api_keys,
        nvidia_base_url=base_url.rstrip("/"),
        nvidia_model=model,
        nvidia_timeout_s=timeout,
        nvidia_max_retries=max_retries,
        request_timeout_s=req_timeout,
        log_level=log_level,
        smtp_host=os.environ.get("SMTP_HOST", ""),
        smtp_port=int(os.environ.get("SMTP_PORT", "587")),
        smtp_user=os.environ.get("SMTP_USER", ""),
        smtp_password=os.environ.get("SMTP_PASSWORD", ""),
        smtp_from=os.environ.get("SMTP_FROM", ""),
        smtp_use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
        alert_email_to=os.environ.get("ALERT_EMAIL_TO", ""),
        key_warn_days=int(os.environ.get("KEY_WARN_DAYS", "14")),
        api_token=os.environ.get("API_TOKEN", ""),
    )
