---
name: aurelius-gateway-hardening
title: Aurelius Gateway Hardening
description: Security-hardened FastAPI gateway patterns: input validation, response sanitization, rate limiting (memory + Redis), health probes, observability, and defense-in-depth middleware stack.
---

## WHEN TO USE

Load this skill when security-hardening a FastAPI gateway, adding input validation,
rate limiting, observability, or response sanitization to an existing service.

## PRINCIPLES

1. **Defense in depth** — validate early, sanitize late
2. **Fail closed** — on error, fall back to safe defaults (rate limiter failure -> allow-all)
3. **Observability** — expose every rejection and error as a metric
4. **Configuration via env** — no secrets in repo; all knobs tunable at deploy time
5. **Backward compatibility** — add endpoints/features without breaking existing clients

## FASTAPI SECURITY MIDDLEWARE STACK

Order matters (outer -> inner):

1. Content-Type guard (reject non-JSON bodies)
2. Host allow-list envelope check
3. Request size limit middleware
4. X-Request-ID injection (traceability)
5. Input parameter validation (pydantic + manual range checks)
6. Rate limiting per IP (token bucket; Redis or in-memory)
7. Response sanitization (LLM completion cleanup)
8. Security headers (CSP, HSTS prod-only, X-Frame-Options: DENY, etc.)
9. Prometheus metrics hook (counters + gauges)

## INPUT VALIDATION (chat/completions)

Field limits (conservative defaults):

| Field | Range |
|-------|-------|
| temperature | 0.0 - 2.0 |
| top_p | 0.01 - 1.0 |
| repetition_penalty | 1.0 - 2.0 |
| max_tokens | 1 - 32768 |
| messages aggregate token budget | ~4 chars/token heuristic |

Enforcement strategy:
- Pydantic schema for required fields + basic types
- Manual validator function for range enforcement
- Increment `aurelius_validation_failures_total` on rejection
- Return 400 with `error.code = "validation_failed"` and field-level diagnostics

## RESPONSE SANITIZATION

Target: LLM completion text only (do not alter user prompts).

Implementation:
```python
def _sanitize_completion(text: str) -> str:
    # Strip non-ASCII (prevents control char leakage)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Collapse repeated whitespace
    return ' '.join(text.split())
```

Apply to every chunk in streaming mode and once to full response in non-streaming.

## RATE LIMITING — PLUGGABLE BACKENDS

**Interface** (`gateway/rate_limit.py`):
```python
class RateLimiter(Protocol):
    def __call__(self, identifier: str) -> bool: ...

class MemoryRateLimiter:
    # token bucket stored in dict[ip] -> {"tokens": int, "ts": float}
    ...

class RedisRateLimiter:
    # distributed token bucket via atomic Lua script
    ...
```

**Factory** `get_rate_limiter()`:
- Reads `AURELIUS_RATE_LIMIT` (default 120), `AURELIUS_RATE_WINDOW` (default 60)
- If `AURELIUS_RATE_LIMIT_REDIS_URL` set -> RedisRateLimiter
- Else -> MemoryRateLimiter
- On Redis init failure -> log warning, fall back to None (allow-all)

**Middleware usage**:
```python
if rate_limiter and not rate_limiter(ip):
    metrics.record_rate_limit_rejection()
    return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
```

## METRICS EXPOSURE

Counters (thread-safe) in `gateway/metrics_middleware.py`:

- `aurelius_rate_limit_rejected_total` – requests blocked by rate limiter
- `aurelius_validation_failures_total` – parameter validation failures

Gauges:

- `aurelius_rate_limiter_backend` – 0 = memory, 1 = redis (set dynamically)

Snapshot API:

- `MetricsCollector.snapshot()` -> dict consumed by Prometheus text exporter
- `MetricsCollector.set_rate_limiter_backend(name: str)` – called by gateway at startup

## HEALTH & READINESS

**Endpoints**

| Endpoint | Purpose | Status codes |
|----------|---------|--------------|
| `GET /health` | Liveness probe | 200 = server up |
| `GET /health/ready` | Readiness probe | 200 = engine loaded; 503 = initializing |
| `GET /healthz` | Legacy alias | mirrors `/health` |

**Kubernetes probes** (in `deployment/helm/aurelius/templates/deployment.yaml`):
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
readinessProbe:
  httpGet:
    path: /health/ready
    port: http
  initialDelaySeconds: 10
  periodSeconds: 5
```

**Docker HEALTHCHECK** (in Dockerfile):
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1
```

## ENVIRONMENT VARIABLES

| Variable | Default | Purpose |
| `AURELIUS_SERVING_PROFILE` | `production` | Serving preset: `production` (full-featured) or `single-gpu` (throughput-optimized, no speculative decode, smaller batches) |
| `AURELIUS_BATCH_SIZE_MAX` | `32` (prod), `16` (single-gpu) | Maximum batch size for static batch endpoint and vLLM `max_num_seqs` |
| `AURELIUS_GPU_MEM_UTIL` | `0.90` (prod), `0.85` (single-gpu) | Fraction of GPU memory allocated to vLLM KV cache |
| `AURELIUS_USE_CUDA_GRAPHS` | `auto` | Enable CUDA graphs for kernel fusion: `always`, `never`, `auto` |
| `AURELIUS_SPECULATIVE_DECODING` | profile‑dependent | Explicitly enable/disable speculative decoding (overrides profile) |
| `AURELIUS_KV_CACHE_STRATEGY` | `standard` (single-gpu) / `auto` (prod) | KV cache backend; `standard` = plain paged attention |

|----------|---------|---------|
| `AURELIUS_MAX_REQUEST_SIZE` | `1048576` (1 MiB) | Max JSON body size |
| `AURELIUS_ALLOWED_HOSTS` | `*` | Comma-separated host allow-list |
| `AURELIUS_RATE_LIMIT` | `120` | Requests per window per IP |
| `AURELIUS_RATE_WINDOW` | `60` | Rate limit window in seconds |
| `AURELIUS_RATE_LIMIT_REDIS_URL` | unset | If set, enables distributed rate limiting |
| `AURELIUS_RATE_LIMIT_PREFIX` | `rl:` | Redis key prefix for limiter |
| `AURELIUS_ENV` | `development` | Set `production` to enable HSTS header |

## REFERENCE FILES

- `gateway/aurelius_api.py` — hardened FastAPI app (validation, sanitization, middleware)
- `gateway/metrics_middleware.py` — thread-safe metrics collector + Redis rate limiter factory
- `gateway/rate_limit.py` — pluggable rate-limit backends (memory & Redis)
- `deployment/helm/aurelius/templates/deployment.yaml` — K8s probes
- `k8s/aurelius-deployment.yaml` — standalone K8s manifest
- `deployment/compose*.yaml` — Docker Compose with HEALTHCHECK
- `README.md` — "Serving & Deployment" section documents all knobs

## PITFALLS

- **Import cycles**: Put lazy imports inside `describe()` methods (see `src/model/interface_framework.py`)
- **Stale `src.agent` imports**: Canonical package is `agent` at repo root; keep `src/agent/__init__.py` as a re-export shim
- **Rate limiter memory leak**: `MemoryRateLimiter` prunes buckets > 60 s old on every check
- **Token budgeting heuristic**: Conservative ~4 chars/token avoids context overruns; adjust if tokenizer changes
- **HSTS in dev**: Only set `Strict-Transport-Security` when `AURELIUS_ENV=production`
- **Response sanitization**: Apply to LLM text only; never to user prompts

## VERIFICATION CHECKLIST

- [ ] `ruff check gateway/` → 0 issues
- [ ] `python -m compileall gateway/` → 0 errors
- [ ] `pytest tests/serving/ -q` → all pass
- [ ] `curl http://localhost:8000/metrics` contains `aurelius_rate_limit_rejected_total` and `aurelius_validation_failures_total`
- [ ] Set `AURELIUS_RATE_LIMIT_REDIS_URL`, restart -> `/metrics` shows `aurelius_rate_limiter_backend 1`
- [ ] Submit out-of-range `temperature=3` -> 400, counter increments
- [ ] `docker build .` succeeds; `docker run` health status becomes `healthy`
- [ ] `kubectl describe deployment aurelius` shows readiness probe on `/health/ready`

## EXTENSIBILITY

**Add a new rate-limit backend:**


### Add a serving profile

1. Define profile defaults in `_load_engine()` (see `gateway/aurelius_api.py`)
2. Document in README "Serving Profiles" and update the env‑var table
3. Consider adding a dedicated CLI flag (`--profile`) for convenience

### Add a new batch endpoint

1. Add a Pydantic request model with `prompts: list[str]`
2. Ensure tokenizer is cached at startup (`_tokenizer` global)
3. Implement `generate_batch(input_ids_list, ...)` on the engine object
4. Wire route to call `_engine_obj.generate_batch(...)` and sanitize outputs
5. Add unit test in `tests/serving/test_batch_endpoint.py`


1. Implement `RateLimiter` protocol in `gateway/rate_limit.py`
2. Extend `get_rate_limiter()` factory branch
3. Call `METRICS.set_rate_limiter_backend("<name>")` after instantiation
4. Add unit test in `tests/gateway/`

**Add a new security header:**

1. Insert into `security_headers_middleware()` list in `gateway/aurelius_api.py`
2. Document in README "Serving & Deployment" section
3. Add feature-flag guard if header should be dev-only

## CHANGELOG

- 2026-05-13 — Initial implementation: validation, sanitization, rate limiting (memory+Redis), health/readiness endpoints, Prometheus counters, Helm/K8s probes, README overhaul
- 2026-05-13 — Added serving profiles (`single-gpu`), batch endpoint `/v1/batch/completions`, env vars for batch/torch tuning; updated engine loader 3‑tuple return; tests updated.
