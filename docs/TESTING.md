# Testing strategy

All tests run with **no API key and no network** (LLM calls mocked at the HTTP/provider layer). CI: pytest + import smoke + docker build. 103 tests.

## Unit
- chunker (3 strategies), loaders (real pypdf / stdlib docx; never-fabricate invariant), cleaner
- retrieval: BM25, RRF (stable keys), hybrid, reranker passthrough
- metrics: hit/recall/precision/MRR/nDCG/AP
- key pool: rotation, mark-bad, fingerprint secrecy, registry
- provider: auth header, retries on 429/5xx/timeout, failover on 401 across keys, streaming deltas, structured JSON validation+retry, key never leaked
- agent: allow-list rejection, bad-args rejection, admin gating, timeout, repeat cap, step cap
- observability: redaction, JSON logs, traces, metrics

## Integration (TestClient, mocked provider)
- `/ask` (answer+sources+provider+usage), `/ask/stream` (token events + done event), `/agent/ask` (tool trace), `/documents` upload/list/delete cascade, `/metrics` `/traces` `/provider`, token auth (401 without/with wrong token), ingest path guard (400 on traversal)
- end-to-end pipeline on a temp corpus

## Failure tests
missing/invalid key (fail-fast), 401 all-keys-bad, timeout, 429 exhaustion, malformed structured output, failed ingest recorded, corrupt PDF → honest empty (no fabrication), bad upload type/size/path.

## Optional real integration
Set `LLM_PROVIDER=nvidia` + `NVIDIA_API_KEYS` and run `python -m src.evaluation.run_eval --data-dir data/raw` for the measured golden-set report. CI never uses paid keys.
