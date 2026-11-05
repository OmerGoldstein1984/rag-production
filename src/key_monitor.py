"""
NVIDIA API key expiry monitor.

NVIDIA keys do not expose their expiry date over the API, so we keep a local
registry (data/key_registry.json, gitignored) mapping each key's *fingerprint*
to a declared expiry date. This tool:

  1. loads the registry,
  2. finds keys expiring within --warn-days (or already expired / failing a
     live probe),
  3. emails an alert via SMTP (credentials from env only).

Full keys are never read, logged, or emailed — only fingerprints (first4...last4).

Usage:
  python -m src.key_monitor --warn-days 14            # check + email
  python -m src.key_monitor --register nvapi-XXXX --expires 2026-12-31  # record a key's expiry
  python -m src.key_monitor --probe                    # live-check keys against the API
  python -m src.key_monitor --dry-run                  # print what would be emailed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.key_pool import (
    DEFAULT_REGISTRY_PATH,
    days_until_expiry,
    fingerprint,
    load_registry,
    save_registry,
)

logger = logging.getLogger(__name__)

DEFAULT_WARN_DAYS = 14


# --- registry -------------------------------------------------------------

def register_key(key: str, expires_at: str, label: str = "", registry_path: str = DEFAULT_REGISTRY_PATH) -> dict:
    """Record (or update) a key's declared expiry in the local registry."""
    reg = load_registry(registry_path)
    entry = {
        "fingerprint": fingerprint(key),
        "label": label,
        "expires_at": expires_at,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    reg["keys"] = [k for k in reg.get("keys", []) if k.get("fingerprint") != entry["fingerprint"]]
    reg["keys"].append(entry)
    save_registry(reg["keys"], registry_path)
    return entry


# --- checks ---------------------------------------------------------------

def expiring_entries(registry_path: str = DEFAULT_REGISTRY_PATH, warn_days: int = DEFAULT_WARN_DAYS) -> List[dict]:
    """Return registry entries that are expired or expire within warn_days."""
    reg = load_registry(registry_path)
    out = []
    for k in reg.get("keys", []):
        exp = k.get("expires_at")
        if not exp:
            continue
        d = days_until_expiry(exp)
        if d <= warn_days:
            k = dict(k)
            k["days_until_expiry"] = d
            out.append(k)
    return out


def probe_key(base_url: str, key: str, timeout_s: float = 15.0) -> bool:
    """Live-check a key with a cheap GET /models. Returns True if it works."""
    import httpx

    try:
        r = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )
        return r.status_code == 200
    except Exception as e:
        logger.warning("probe failed: %s", e)
        return False


# --- email ----------------------------------------------------------------

def smtp_settings_from_env() -> dict:
    return {
        "host": os.environ.get("SMTP_HOST", ""),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from": os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "")),
        "to": os.environ.get("ALERT_EMAIL_TO", ""),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
    }


def build_alert(subject: str, body: str, from_addr: str = "", to_addr: str = "") -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr or "key-monitor@localhost"
    msg["To"] = to_addr or "owner@localhost"
    msg.set_content(body)
    return msg


def send_alert(subject: str, body: str, settings: Optional[dict] = None) -> None:
    """Send an alert email via SMTP. Credentials come from env; never logged."""
    s = settings or smtp_settings_from_env()
    if not s["host"] or not s["to"]:
        raise RuntimeError("SMTP_HOST and ALERT_EMAIL_TO must be set to send alerts.")
    msg = build_alert(subject, body, from_addr=s["from"], to_addr=s["to"])
    with smtplib.SMTP(s["host"], s["port"], timeout=30) as server:
        if s["use_tls"]:
            server.starttls()
        if s["user"]:
            server.login(s["user"], s["password"])
        server.send_message(msg)
    logger.info("alert email sent to %s", s["to"])


def format_alert(entries: List[dict], warn_days: int) -> tuple[str, str]:
    subject = f"[rag-project] NVIDIA API key expiring soon ({len(entries)} key(s))"
    lines = [
        f"The following NVIDIA API key(s) expire within {warn_days} days (or are already expired):",
        "",
    ]
    for k in entries:
        d = k.get("days_until_expiry")
        state = "EXPIRED" if (d is not None and d < 0) else f"expires in {d} day(s)"
        lines.append(
            f"  - key {k.get('fingerprint')} ({k.get('label') or 'no label'}): {state} "
            f"[expiry {k.get('expires_at')}]"
        )
    lines += [
        "",
        "Action: create a new NVIDIA API key, add it to NVIDIA_API_KEYS,",
        "register its expiry (`python -m src.key_monitor --register <key> --expires YYYY-MM-DD`),",
        "and remove the old key from the pool.",
        "",
        "(Only key fingerprints are shown — full keys are never stored or logged.)",
    ]
    return subject, "\n".join(lines)


# --- CLI ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA API key expiry monitor")
    parser.add_argument("--warn-days", type=int, default=DEFAULT_WARN_DAYS)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--register", metavar="KEY", help="Register a key's declared expiry")
    parser.add_argument("--expires", metavar="YYYY-MM-DD", help="Expiry date for --register")
    parser.add_argument("--label", default="", help="Human label for --register")
    parser.add_argument("--probe", action="store_true", help="Live-probe keys from NVIDIA_API_KEYS")
    parser.add_argument("--dry-run", action="store_true", help="Print the alert instead of emailing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.register:
        if not args.expires:
            parser.error("--register requires --expires YYYY-MM-DD")
        entry = register_key(args.register, args.expires, args.label, args.registry)
        print(f"registered {entry['fingerprint']} expires {entry['expires_at']}")
        # wipe the key from the process env echo
        return 0

    entries = expiring_entries(args.registry, args.warn_days)

    if args.probe:
        raw = os.environ.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY") or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        for k in keys:
            ok = probe_key(base_url, k)
            fp = fingerprint(k)
            status = "OK" if ok else "FAILING"
            print(f"probe {fp}: {status}")
            if not ok:
                entries.append({"fingerprint": fp, "label": "live-probe", "expires_at": None,
                                "days_until_expiry": None, "probe": "failing"})

    if not entries:
        print("no keys expiring within warn window")
        return 0

    subject, body = format_alert(entries, args.warn_days)
    print(body)
    if args.dry_run:
        print("(dry-run: no email sent)")
        return 0
    send_alert(subject, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
