# Runbook / Deployment

## Local dev (no key)
```bash
pip install -r requirements.txt
cp .env.example .env          # leave LLM_PROVIDER=mock
python -m pytest               # 103 tests, no network/key
python scripts/demo.py         # guided end-to-end demo (mock)
uvicorn src.api.main:app --port 8000     # API
streamlit run ui/streamlit_app.py        # UI on :8501
```

## Real NVIDIA
```bash
export LLM_PROVIDER=nvidia
export NVIDIA_API_KEYS="key1,key2"     # multi-key pool w/ failover
export NVIDIA_MODEL=meta/llama-3.1-70b-instruct
# optional alerts before a key expires:
python -m src.key_monitor --register <key> --expires 2026-12-31 --label primary
python -m src.key_monitor --warn-days 14          # emails via SMTP_* env
python -m src.evaluation.run_eval --data-dir data/raw --output evaluation/results.md
python scripts/demo.py --real
```

## Docker
```bash
docker compose up --build        # api :8000, ui :8501; volumes persist Chroma/uploads
# healthcheck is python-based (curl not in slim image); .env never baked in (CI asserts)
```

## Ops endpoints
`/health` (store/cache stats) · `/provider` (active model, is_real) · `/metrics` (counters, latency p50/p95, tokens, error rate) · `/traces` (per-stage timings) · `/documents/status` (ingest log).

## Troubleshooting
- `MissingConfigError` at startup → set `NVIDIA_API_KEYS` (or `LLM_PROVIDER=mock` for dev).
- 401s → key revoked/expired; pool rotates; register new key expiry; check `/provider`.
- Slow first run → embedding/reranker models download once to HF cache.
- Retrieval looks stale after uploads → `POST /documents/reindex` (admin).
