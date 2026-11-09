"""
NVIDIA API key expiry monitor.

BUSINESS PROBLEM SOLVED: NVIDIA API keys expire, but NVIDIA does not expose
expiry dates over its API, so a key can silently stop working and take down the
LLM path. This tool keeps a local registry (data/key_registry.json, gitignored)
mapping each key's fingerprint to a declared expiry date, finds keys expiring
soon (or already expired / failing a live probe), and emails an alert via SMTP
so operators rotate keys before they break. Full keys are never read, logged, or
emailed -- only fingerprints (first4...last4).

WHERE IT SITS IN THE DATA FLOW: Operational sidecar to src/key_pool.py. It reads
and writes the same key registry that KeyPool.apply_registry() loads at startup
(src/key_pool.py:235). It is run manually or on a schedule (e.g. cron), not as
part of the request path -- it is a CLI tool invoked via `python -m src.key_monitor`.

WHAT CALLS IT: Operators/CI run it directly:
  python -m src.key_monitor --warn-days 14        # check + email
  python -m src.key_monitor --register <key> --expires 2026-12-31
  python -m src.key_monitor --probe                # live-check keys against the API
  python -m src.key_monitor --dry-run              # print what would be emailed
It is also exercised by tests (tests/test_key_monitor.py if present).

WHAT IT CALLS: src.key_pool (DEFAULT_REGISTRY_PATH, days_until_expiry,
fingerprint, load_registry, save_registry). Standard library: argparse, json,
logging, os, smtplib, sys, datetime, email.message, pathlib, typing; and httpx
(imported lazily inside probe_key for the live API check). No project module
calls key_monitor at runtime.

WHAT BREAKS IF REMOVED: There is no automated warning before an NVIDIA key
expires. The first symptom would be live 401s/403s in production, which
key_pool.py handles via failover -- but if all keys expire unnoticed, the LLM
path goes down with no prior alert.

# GAP: Out-of-band expiry tracking. Since the provider API never tells us when a
key expires, we maintain our own registry and a separate CLI tool to monitor it
-- a learner must understand that not all operational data comes from the API
itself.
# GAP: CLI tool as operational artifact. This module is a runnable script
(`if __name__ == "__main__"`), not a library imported by the API; learners from
web backgrounds may expect everything to be imported, but ops tools often run as
CLI entry points.
# BUSINESS: Protects uptime and user trust -- advance expiry alerts prevent
unexpected LLM outages; credential safety (fingerprints only) protects against
key theft.
# INTERVIEW: Say it out loud -- "key_monitor watches a local registry of key
expiry dates, probes keys live against the NVIDIA API, and emails an SMTP alert
when a key expires within the warn window, showing only fingerprints."
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

# Allow `python -m src.key_monitor` to import src.key_pool regardless of the
# current working directory by putting the project root on sys.path. This is a
# CLI-script convenience; library code normally relies on installed packages.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.key_pool import (
    DEFAULT_REGISTRY_PATH,
    days_until_expiry,
    fingerprint,
    load_registry,
    save_registry,
)

logger = logging.getLogger(__name__)   # module-level logger for probe warnings

DEFAULT_WARN_DAYS = 14


# --- registry -------------------------------------------------------------

def register_key(key: str, expires_at: str, label: str = "", registry_path: str = DEFAULT_REGISTRY_PATH) -> dict:
    """Record (or update) a key's declared expiry in the local registry.

    WHAT: Loads the registry, removes any existing entry with the same
    fingerprint, appends a new entry (fingerprint, label, expires_at,
    registered_at timestamp), saves it, and returns the entry.
    WHY: NVIDIA doesn't expose expiry dates, so an operator records each key's
    declared expiry here; the registry is the single source key_monitor and
    KeyPool.apply_registry() both read.
    INPUTS: key (str, full key -- only its fingerprint is stored), expires_at
    (str, ISO date), label (str, optional human label), registry_path (str).
    OUTPUTS: dict; example {"fingerprint": "nvap...1234", "label": "prod-key-1",
    "expires_at": "2026-12-31", "registered_at": "2026-11-09T..."}.
    SIDE EFFECTS: Reads and writes the registry JSON file to disk.
    FAILURE MODES: Raises OSError/json errors if the registry file cannot be read
    or written (propagated from load_registry/save_registry).
    ALTERNATIVE: Store expiry in a database -- rejected because a single JSON
    file is simple, gitignored, and sufficient for a single-writer ops tool.
    # GAP: Upsert semantics. The filter-then-append pattern replaces an existing
    entry for the same fingerprint rather than duplicating it -- important so
    re-registering a key doesn't grow the registry forever.
    # BUSINESS: Protects uptime -- accurate expiry records mean alerts fire before
    a key stops working.
    # INTERVIEW: Say it out loud -- "register_key upserts a fingerprint + expiry
    into the local registry, never storing the full key."
    """
    reg = load_registry(registry_path)
    entry = {
        "fingerprint": fingerprint(key),     # never store the full key
        "label": label,
        "expires_at": expires_at,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    # Drop any prior entry for this fingerprint so re-registering upserts, not
    # duplicates.
    reg["keys"] = [k for k in reg.get("keys", []) if k.get("fingerprint") != entry["fingerprint"]]
    reg["keys"].append(entry)
    save_registry(reg["keys"], registry_path)
    return entry


# --- checks ---------------------------------------------------------------

def expiring_entries(registry_path: str = DEFAULT_REGISTRY_PATH, warn_days: int = DEFAULT_WARN_DAYS) -> List[dict]:
    """Return registry entries that are expired or expire within warn_days.

    WHAT: Loads the registry and, for each entry with an expires_at date, uses
    days_until_expiry() to compute how many days remain; keeps entries whose
    days_until_expiry is within warn_days (including negative, i.e. already
    expired).
    WHY: This is the core detection logic -- it decides which keys need an
    alert before they break.
    INPUTS: registry_path (str), warn_days (int, default 14).
    OUTPUTS: List[dict] -- copies of matching entries each with an added
    "days_until_expiry" int.
    SIDE EFFECTS: Reads the registry file from disk.
    FAILURE MODES: Propagates load_registry errors if the registry is corrupt.
    Entries without expires_at are skipped (no crash).
    ALTERNATIVE: Compute expiry from key creation date -- rejected because NVIDIA
    keys have no creation-date API, so we rely on operator-declared expiry.
    # GAP: Negative-day semantics. days_until_expiry returns negative for expired
    keys; `d <= warn_days` catches both "expiring soon" and "already expired" in
    one comparison.
    # BUSINESS: Protects uptime -- catching expired and soon-to-expire keys lets
    operators rotate before the LLM path fails.
    # INTERVIEW: Say it out loud -- "expiring_entries returns every registry key
    whose expiry is within the warn window, or already expired, with days left
    attached."
    """
    reg = load_registry(registry_path)
    out = []
    for k in reg.get("keys", []):
        exp = k.get("expires_at")
        if not exp:
            continue                            # skip entries with no expiry date
        d = days_until_expiry(exp)
        if d <= warn_days:
            k = dict(k)                          # copy so we don't mutate the registry
            k["days_until_expiry"] = d
            out.append(k)
    return out


def probe_key(base_url: str, key: str, timeout_s: float = 15.0) -> bool:
    """Live-check a key with a cheap GET /models. Returns True if it works.

    WHAT: Calls GET {base_url}/models with a Bearer token for the given key and
    returns True only on HTTP 200.
    WHY: A live probe detects keys that are expired or revoked even if the local
    registry says they are still valid -- registry dates are declared, not
    authoritative.
    INPUTS: base_url (str), key (str -- used in an Authorization header, never
    logged), timeout_s (float, default 15).
    OUTPUTS: bool -- True on HTTP 200, False otherwise.
    SIDE EFFECTS: Makes a real HTTP request to NVIDIA; logs a warning on failure.
    FAILURE MODES: Any exception (network, timeout, non-200) returns False and
    logs the error -- the probe never crashes the monitor.
    ALTERNATIVE: Call a heavier endpoint -- rejected because /models is cheap,
    requires auth, and reliably indicates whether the key works.
    # GAP: Synthetic probing vs registry dates. A declared expiry can be wrong;
    a live probe is ground truth. Learners should understand the difference
    between "declared metadata" and "observed reality".
    # BUSINESS: Protects uptime -- a probe catches revoked keys before users hit
    401s.
    # INTERVIEW: Say it out loud -- "probe_key does a live GET /models with the
    key's bearer token; 200 means the key works, anything else is treated as
    failing."
    """
    import httpx

    try:
        r = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )
        return r.status_code == 200
    except Exception as e:
        logger.warning("probe failed: %s", e)   # httpx error details, not the key
        return False


# --- email ----------------------------------------------------------------

def smtp_settings_from_env() -> dict:
    """Read SMTP connection and credential settings from environment variables.

    WHAT: Pulls SMTP_HOST, PORT, USER, PASSWORD, FROM, TO, and USE_TLS from
    os.environ with sensible defaults (port 587, TLS on).
    WHY: Credentials come only from env, never source code, so the alert email
    can be configured per deployment without touching code.
    INPUTS: None (reads os.environ).
    OUTPUTS: dict with keys host, port, user, password, from, to, use_tls.
    SIDE EFFECTS: None.
    FAILURE MODES: None -- missing vars yield empty/defaults; the caller
    (send_alert) validates that host and to are present.
    ALTERNATIVE: Pass SMTP settings as arguments -- rejected because env keeps
    secrets out of code and matches the rest of the config approach.
    """
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
    """Build an EmailMessage with the given subject/body/from/to.

    WHAT: Constructs a MIME email message; defaults the From/To addresses when
    not supplied.
    WHY: Separates message construction from sending, so send_alert and the
    dry-run path can both reuse it (dry-run prints the body without sending).
    INPUTS: subject (str), body (str), from_addr (str), to_addr (str).
    OUTPUTS: EmailMessage.
    SIDE EFFECTS: None.
    FAILURE MODES: None.
    ALTERNATIVE: Build the message inline inside send_alert -- rejected because
    build_alert is independently tested and reused.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr or "key-monitor@localhost"
    msg["To"] = to_addr or "owner@localhost"
    msg.set_content(body)
    return msg


def send_alert(subject: str, body: str, settings: Optional[dict] = None) -> None:
    """Send an alert email via SMTP. Credentials come from env; never logged.

    WHAT: Validates SMTP host and recipient, builds the message, connects to the
    SMTP server, optionally starts TLS, logs in if a user is configured, and
    sends the message.
    WHY: This is the actual notification path that tells operators a key is
    expiring; credentials are read from env at send time and never logged.
    INPUTS: subject (str), body (str), settings (Optional[dict] -- defaults to
    smtp_settings_from_env()).
    OUTPUTS: None.
    SIDE EFFECTS: Makes a real SMTP connection and sends an email; logs an info
    message on success.
    FAILURE MODES: Raises RuntimeError if SMTP_HOST or ALERT_EMAIL_TO is unset;
    smtplib errors propagate if the SMTP server is unreachable or auth fails.
    ALTERNATIVE: Send via an email API (SendGrid, SES) -- rejected because SMTP
    is universal, needs no extra SDK, and is configured via env like the rest of
    the system.
    # BUSINESS: Protects trust -- operators get timely expiry alerts so keys are
    rotated before breaking; credentials never appear in logs.
    # INTERVIEW: Say it out loud -- "send_alert validates required SMTP settings,
    connects over SMTP with optional TLS and login, sends the alert, and logs
    that it was sent -- credentials come only from env."
    """
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
    """Build the subject and body text for the key-expiry alert email.

    WHAT: Renders a subject line and a body listing each expiring key's
    fingerprint, label, state (expired / days left), and expiry date, plus
    action instructions. Only fingerprints appear -- never full keys.
    WHY: A consistent, readable alert body that operators can act on, with the
    security property that no full key is ever emailed.
    INPUTS: entries (List[dict] from expiring_entries()), warn_days (int).
    OUTPUTS: tuple[str, str] -- (subject, body).
    SIDE EFFECTS: None.
    FAILURE MODES: None for well-formed entries; missing fields render as None/
    "no label".
    ALTERNATIVE: Send HTML or JSON -- rejected because plain text is simplest,
    readable in any mail client, and easy to scan during an incident.
    # BUSINESS: Protects trust and uptime -- clear, fingerprint-only alerts tell
    operators exactly which keys to rotate and by when.
    # INTERVIEW: Say it out loud -- "format_alert builds a subject + body listing
    each expiring key's fingerprint and state, with rotation instructions, never
    exposing full keys."
    """
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
    """CLI entry point for the NVIDIA API key expiry monitor.

    WHAT: Parses command-line flags and dispatches: --register records a key's
    expiry; otherwise it computes expiring entries, optionally live-probes the
    keys, and either emails or dry-runs the alert.
    WHY: A single command gives operators everything they need: register keys,
    check expiry, probe live keys, and preview/send alerts.
    INPUTS: None (reads sys.argv via argparse).
    OUTPUTS: int process exit code (0 on success).
    SIDE EFFECTS: Configures logging, reads/writes the registry file, makes live
    HTTP probes to NVIDIA, and may send an SMTP email.
    FAILURE MODES: parser.error() exits with code 2 on bad args; register_key or
    send_alert exceptions propagate (nonzero exit). Probe failures are caught
    inside probe_key and do not fail the run.
    ALTERNATIVE: Make this an importable library with subcommands -- rejected
    because it is an ops tool run directly by operators/Cron, not part of the
    request path.
    # GAP: CLI dispatch with side effects. main() mixes argument parsing, file
    I/O, network calls, and email sending -- a learner should see that ops CLI
    tools are allowed to have side effects, unlike pure request handlers.
    # BUSINESS: Protects uptime -- the CLI is the operator's primary way to
    detect and act on expiring keys before they break the LLM path.
    # INTERVIEW: Say it out loud -- "main registers keys, checks expiry, probes
    live keys, and sends/dry-runs the alert, returning 0 on success."
    """
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

    # --register path: record a key's expiry and exit.
    if args.register:
        if not args.expires:
            parser.error("--register requires --expires YYYY-MM-DD")
        entry = register_key(args.register, args.expires, args.label, args.registry)
        print(f"registered {entry['fingerprint']} expires {entry['expires_at']}")
        # Don't echo the full key back to the terminal; fingerprint only.
        return 0

    # Find keys expiring within the warn window (or already expired).
    entries = expiring_entries(args.registry, args.warn_days)

    # Optional live probe: check each configured key against the NVIDIA API and
    # fold failing keys into the alert list.
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

    # Nothing to report.
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
