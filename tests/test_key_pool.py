"""
Tests for the API key pool, provider failover, and the expiry monitor.

All mocked — no network, and full keys never appear in logs/registry output.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.key_pool import KeyPool, days_until_expiry, fingerprint, load_registry, save_registry
from src import key_monitor
from providers.llm_nvidia import NVIDIAProvider, ProviderAuthError

K1 = "nvapi-aaaaaaaaaaaaaaaaaaaa01"
K2 = "nvapi-bbbbbbbbbbbbbbbbbbb02"
K3 = "nvapi-ccccccccccccccccccc03"


# --- key pool -------------------------------------------------------------

def test_fingerprint_hides_full_key():
    fp = fingerprint(K1)
    assert K1 not in fp
    assert fp.startswith("nvap...") and fp.endswith("aa01")


def test_pool_from_env_multi_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEYS", f"{K1}, {K2} ,{K3}")
    pool = KeyPool.from_env(registry_path=tmp_path / "none.json")
    assert len(pool.keys) == 3
    assert pool.active_count() == 3


def test_pool_from_env_single_key_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEYS", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", K1)
    pool = KeyPool.from_env(registry_path=tmp_path / "none.json")
    assert pool.keys == [K1]


def test_pool_requires_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEYS", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        KeyPool.from_env(registry_path=tmp_path / "none.json")


def test_round_robin_rotation():
    pool = KeyPool(keys=[K1, K2, K3])
    seen = {pool.current_key() for _ in range(6)}
    assert seen == {K1, K2, K3}


def test_mark_bad_rotates_and_skips():
    pool = KeyPool(keys=[K1, K2, K3])
    first = pool.current_key()
    nxt = pool.mark_bad(first, "HTTP 401")
    assert nxt is not None and nxt != first
    assert pool.active_count() == 2
    # marking all bad -> no active
    pool.mark_bad(K2)
    pool.mark_bad(K3)
    assert pool.active_count() == 0
    with pytest.raises(RuntimeError):
        pool.current_key()


def test_status_never_leaks_full_key():
    pool = KeyPool(keys=[K1, K2])
    pool.mark_bad(K1, "boom")
    status = pool.status()
    blob = json.dumps(status)
    assert K1 not in blob and K2 not in blob
    assert any(s["fingerprint"] == fingerprint(K1) and s["status"] == "bad" for s in status)


def test_registry_roundtrip_and_expiry_attach(tmp_path):
    reg = tmp_path / "reg.json"
    exp = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
    save_registry([{"fingerprint": fingerprint(K1), "expires_at": exp}], reg)
    pool = KeyPool(keys=[K1, K2])
    pool.apply_registry(reg)
    entry = next(e for e in pool.entries if e.fingerprint == fingerprint(K1))
    assert entry.expires_at == exp


def test_days_until_expiry():
    future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    assert days_until_expiry(future) >= 4
    assert days_until_expiry(past) < 0


# --- provider failover ----------------------------------------------------

def make_pool_provider(handler, keys=(K1, K2)):
    pool = KeyPool(keys=list(keys))
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return NVIDIAProvider(client=client, key_pool=pool, backoff_base_s=0.0, backoff_max_s=0.0), pool


def ok_payload(text="ok"):
    return {
        "model": "m",
        "choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_provider_failover_on_401_then_success():
    seen_auth = []

    def handler(request):
        seen_auth.append(request.headers.get("authorization"))
        if len(seen_auth) == 1:
            return httpx.Response(401, json={"error": "bad key"}, request=request)
        return httpx.Response(200, json=ok_payload("recovered"), request=request)

    p, pool = make_pool_provider(handler)
    result = p.chat([{"role": "user", "content": "hi"}])
    assert result.text == "recovered"
    assert len(seen_auth) == 2
    assert seen_auth[0] == f"Bearer {K1}" and seen_auth[1] == f"Bearer {K2}"
    # first key marked bad in the pool
    assert any(e.fingerprint == fingerprint(K1) and e.status == "bad" for e in pool.entries)


def test_provider_auth_error_when_all_keys_bad_no_leak():
    def handler(request):
        return httpx.Response(401, json={"error": f"invalid {K1}"}, request=request)

    p, pool = make_pool_provider(handler, keys=(K1, K2))
    with pytest.raises(ProviderAuthError) as exc_info:
        p.chat([{"role": "user", "content": "hi"}])
    assert K1 not in str(exc_info.value) and K2 not in str(exc_info.value)
    assert pool.active_count() == 0


def test_provider_without_pool_401_raises_immediately():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad"}, request=request)

    transport = httpx.MockTransport(handler)
    p = NVIDIAProvider(api_key=K1, client=httpx.Client(transport=transport))
    with pytest.raises(ProviderAuthError):
        p.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # no retry without a pool


# --- expiry monitor -------------------------------------------------------

def test_expiring_entries_within_window(tmp_path):
    reg = tmp_path / "reg.json"
    soon = (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat()
    far = (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat()
    gone = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    save_registry([
        {"fingerprint": fingerprint(K1), "expires_at": soon},
        {"fingerprint": fingerprint(K2), "expires_at": far},
        {"fingerprint": fingerprint(K3), "expires_at": gone},
    ], reg)
    hits = key_monitor.expiring_entries(str(reg), warn_days=14)
    fps = {h["fingerprint"] for h in hits}
    assert fingerprint(K1) in fps and fingerprint(K3) in fps
    assert fingerprint(K2) not in fps


def test_register_key_writes_registry(tmp_path, monkeypatch):
    reg = tmp_path / "reg.json"
    entry = key_monitor.register_key(K1, "2027-01-01", label="primary", registry_path=str(reg))
    assert entry["fingerprint"] == fingerprint(K1)
    assert K1 not in reg.read_text(encoding="utf-8")  # full key never stored
    data = load_registry(reg)
    assert data["keys"][0]["expires_at"] == "2027-01-01"


def test_format_alert_contains_fingerprint_not_key(tmp_path):
    reg = tmp_path / "reg.json"
    soon = (datetime.now(timezone.utc) + timedelta(days=3)).date().isoformat()
    save_registry([{"fingerprint": fingerprint(K1), "expires_at": soon, "label": "main"}], reg)
    entries = key_monitor.expiring_entries(str(reg), warn_days=14)
    subject, body = key_monitor.format_alert(entries, 14)
    assert fingerprint(K1) in body
    assert K1 not in body and K1 not in subject


def test_send_alert_uses_smtp_and_never_logs_password(tmp_path, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host
            sent["port"] = port
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def starttls(self):
            sent["tls"] = True
        def login(self, user, password):
            sent["login"] = (user, password)
        def send_message(self, msg):
            sent["subject"] = msg["Subject"]
            sent["to"] = msg["To"]
            sent["body"] = msg.get_content()

    monkeypatch.setattr(key_monitor.smtplib, "SMTP", FakeSMTP)
    settings = {
        "host": "smtp.example.com", "port": 587, "user": "u", "password": "secret-pw",
        "from": "alerts@example.com", "to": "me@example.com", "use_tls": True,
    }
    key_monitor.send_alert("test subject", "key fp expiring", settings)
    assert sent["host"] == "smtp.example.com"
    assert sent["login"] == ("u", "secret-pw")
    assert sent["to"] == "me@example.com"
    assert "secret-pw" not in sent["body"] and "secret-pw" not in sent["subject"]


def test_send_alert_requires_host_and_to():
    with pytest.raises(RuntimeError):
        key_monitor.send_alert("s", "b", {"host": "", "to": "", "port": 587, "use_tls": True})


def test_monitor_cli_dry_run(tmp_path, monkeypatch, capsys):
    reg = tmp_path / "reg.json"
    soon = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()
    save_registry([{"fingerprint": fingerprint(K1), "expires_at": soon}], reg)
    monkeypatch.setattr(sys, "argv", ["key_monitor", "--registry", str(reg), "--warn-days", "14", "--dry-run"])
    rc = key_monitor.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert fingerprint(K1) in out and K1 not in out
    assert "dry-run" in out
