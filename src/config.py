"""Environment-driven configuration for the RAG system.

Secrets come only from environment variables. .env is gitignored.
Fail-fast: provider=nvidia without NVIDIA_API_KEY raises MissingConfigError.
Mock is used only when LLM_PROVIDER=mock is set explicitly (dev/tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"


class MissingConfigError(RuntimeError):
    """Raised when required configuration is absent from the environment."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    llm_provider: str = "mock"
    nvidia_api_key: Optional[str] = None          # single key (back-compat)
    nvidia_api_keys: tuple = ()                    # multi-key pool (NVIDIA_API_KEYS, comma-separated)
    nvidia_base_url: str = DEFAULT_NVIDIA_BASE_URL
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    nvidia_timeout_s: float = 60.0
    nvidia_max_retries: int = 3
    request_timeout_s: float = 60.0
    log_level: str = "INFO"
    # key-expiry alerting (email)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True
    alert_email_to: str = ""
    key_warn_days: int = 14
    # Optional API bearer token. If set, mutating endpoints require it.
    api_token: str = ""

    @property
    def is_real_provider(self) -> bool:
        return self.llm_provider != "mock"


def get_settings() -> Settings:
    """Load settings from env vars, validating required secrets."""
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

    if provider == "nvidia" and not (api_key and api_key.strip()) and not api_keys:
        raise MissingConfigError(
            "LLM_PROVIDER=nvidia but no NVIDIA key is set. "
            "Export NVIDIA_API_KEYS='key1,key2,...' (comma-separated, recommended) or "
            "NVIDIA_API_KEY for a single key — never in source code. "
            "For local development without a key, set LLM_PROVIDER=mock explicitly."
        )

    if provider not in ("nvidia", "mock"):
        raise MissingConfigError(
            f"Unsupported LLM_PROVIDER={provider!r}. Supported: 'nvidia', 'mock'."
        )

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
