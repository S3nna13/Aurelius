---
name: aurelius-rate-limit-redis
title: Distributed Rate Limiting with Redis for Aurelius
description: Pluggable Redis-backed rate limiter using atomic Lua scripts; drop-in replacement for in-memory token bucket. Auto-selected via AURELIUS_RATE_LIMIT_REDIS_URL.
---

## WHEN TO USE

Load this skill when you need distributed rate limiting across multiple Aurelius API
replicas (K8s, Docker Compose scale‑out), or when you want to replace the default
in‑memory token bucket with a shared backend without changing middleware code.

---

## RATE LIMITER ARCHITECTURE

### Core protocol
```python
from typing import Protocol

class RateLimiter(Protocol):
    """Callable rate limiter returns True if request is allowed, False if blocked."""
    def __call__(self, identifier: str) -> bool: ...
```

### Backends

**MemoryRateLimiter** (default)
- In‑process dict: `{ip: {"tokens": float, "ts": timestamp}}`
- Refill: proportional to elapsed time
- Prune: remove stale entries > 60 s on every call (prevents memory leak)
- Pros: zero dependencies, fast
- Cons: not shared across process replicas

**RedisRateLimiter** (distributed)
- Single Lua script (`EVAL`) — atomic token bucket operations
- Redis keys: `rl:{ip}` (prefix configurable via `AURELIUS_RATE_LIMIT_PREFIX`)
- Script returns `1` (allow) or `0` (deny)
- Pros: consistent limits cluster‑wide; survives pod restart
- Cons: Redis dependency; small network overhead

---

## FACTORY & CONFIGURATION

**Factory function** `get_rate_limiter()` reads environment:

| Env var | Meaning |
|---------|---------|
| `AURELIUS_RATE_LIMIT` | Burst capacity (tokens per window). Default: `120` |
| `AURELIUS_RATE_WINDOW` | Window size in seconds. Default: `60` |
| `AURELIUS_RATE_LIMIT_REDIS_URL` | If set (e.g. `redis://redis:6379/0`), factory returns `RedisRateLimiter` |
| `AURELIUS_RATE_LIMIT_PREFIX` | Redis key prefix; default `rl:` |

**Fallback behavior**
- Redis connection failure -> logs warning, returns `None` (allow‑all, fail‑open)
- Gateway middleware treats `None` as "no rate limiting" — useful for debugging

**Backend indicator**
After instantiation the gateway calls:
```python
METRICS.set_rate_limiter_backend("redis" if using_redis else "memory")
```
This sets Prometheus gauge `aurelius_rate_limiter_backend` (0=memory, 1=redis).

---

## LUA SCRIPT LOGIC (RedisRateLimiter)

```lua
-- KEYS[1] = rate-limit key (e.g. "rl:1.2.3.4")
-- ARGV[1] = rate (tokens per window)
-- ARGV[2] = window (seconds)
-- ARGV[3] = now (timestamp)
-- ARGV[4] = requested (always 1 for per-request limiting)

local key = KEYS[1]
local rate = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(bucket[1]) or rate
local ts = tonumber(bucket[2]) or now

-- refill
local elapsed = now - ts
local refill = (elapsed / window) * rate
tokens = math.min(rate, tokens + refill)

if tokens < requested then
    return 0  -- denied
else
    tokens = tokens - requested
    redis.call("HMSET", key, "tokens", tokens, "ts", now)
    redis.call("EXPIRE", key, math.ceil(window * 2))
    return 1  -- allowed
end
```

Invoked from Python via:
```python
allowed = self._redis.eval(lua_script, 1, key, rate, window, now, 1)
```

---

## INTEGRATION POINTS

**gateway/aurelius_api.py**
```python
from gateway.rate_limit import get_rate_limiter

_rate_limiter: RateLimiter | None = None

@app.on_event("startup")
def _init_rate_limiter() -> None:
    global _rate_limiter
    _rate_limiter = get_rate_limiter()
    METRICS.set_rate_limiter_backend(
        "redis" if isinstance(_rate_limiter, RedisRateLimiter) else "memory"
    )
```

Middleware callback:
```python
if _rate_limiter and not _rate_limiter(ip):
    METRICS.record_rate_limit_rejection()
    return JSONResponse(...)
```

**Metrics** (`gateway/metrics_middleware.py`)
- Counter: `aurelius_rate_limit_rejected_total`
- Gauge: `aurelius_rate_limiter_backend` (0=momory, 1=redis)

---

## TESTING STRATEGY

Unit test `MemoryRateLimiter`:
1. Call repeatedly to exceed burst limit → expect False after tokens exhausted
2. Sleep past refill window → expect True after tokens replenish
3. Ensure stale entries are pruned (internal dict size bounded)

Integration test `RedisRateLimiter`:
1. Spin up Redis container (pytest fixture with `redis` package)
2. Burst 150 requests with limit 120 → expect ~30 rejections
3. Verify Lua script atomicity by concurrent requests (threadpool)
4. Check Redis key TTL is set (~2× window)

**Smoke test** (manual):
```bash
# Memory (default)
curl http://localhost:8000/health
# Flood
for i in $(seq 1 130); do curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions; done | grep -c 429
# Should see ~10 rejections if rate=120

# Redis
export AURELIUS_RATE_LIMIT_REDIS_URL=redis://localhost:6379/0
pytest tests/gateway/test_rate_limit_redis.py -k test_distributed_burst_consistent_across_workers
```

---

## DEPLOYMENT

**Docker Compose** (`deployment/compose*.yaml`):
```yaml
services:
  aurelius:
    environment:
      AURELIUS_RATE_LIMIT: "120"
      AURELIUS_RATE_WINDOW: "60"
      # AURELIUS_RATE_LIMIT_REDIS_URL: redis://redis:6379/0  # uncomment to enable
```

**Kubernetes / Helm** (environment block):
```yaml
env:
  - name: AURELIUS_RATE_LIMIT
    value: "120"
  - name: AURELIUS_RATE_WINDOW
    value: "60"
  - name: AURELIUS_RATE_LIMIT_REDIS_URL
    value: "redis://redis-headless:6379/0"
```

**Redis sizing**:
- Single node sufficient for moderate traffic (< 10 k RPM)
- For higher scale, use Redis Cluster; Lua script remains single-key so client-side sharding by IP prefix may be needed
- TTL: `EXPIRE key <2×window>`; Redis auto-eviction keeps memory bounded

---

## PITFALLS & TROUBLESHOOTING

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No rejections despite high traffic | Using `MemoryRateLimiter` in multi‑pod setup | Set `AURELIUS_RATE_LIMIT_REDIS_URL` |
| All requests denied immediately | Redis URL malformed or unreachable | Check connectivity; fallback is allow‑all but logs warning |
| Memory grows unbounded in MemoryRateLimiter | Prune interval not reached or bug | Ensure `_prune_stale()` called on every `__call__` |
| Lua script error | Redis version < 2.6 (no `EVAL`) | Upgrade Redis; all modern versions support `EVAL` |
| Stale backend indicator on `/metrics` | `set_rate_liminder_backend()` not called at startup | Verify startup event hook in `aurelius_api.py` |

---

## PERFORMANCE

- **Memory backend**: ~200 ns per call (dict lookup + float math)
- **Redis backend**: ~0.3–0.6 ms per call (network round-trip + Lua)
- Benchmark with `pytest tests/gateway/test_rate_limit_perf.py` (exists)
- Consider connection pooling (`redis.ConnectionPool`) for high QPS deployments

---

## REFERENCES

- Implementation: `gateway/rate_limit.py`, `gateway/metrics_middleware.py`, `gateway/aurelius_api.py`
- Original commit: `059423fd feat(gateway): health probes, Redis rate limiting, observability`
