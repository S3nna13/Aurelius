# Aurelius — Agent Architecture Guide

## Project Overview

Aurelius is a 1.395B-parameter decoder-only LLM (targeting 2.7B) with a full-stack architecture spanning four layers:

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| **Rust Engine** | `crates/`, `tools/` | Rust | High-performance data engine, search, tokenization, vector similarity, caching, auth sessions |
| **Python Backend** | `src/`, `configs/` | Python 3.12 | Model training, inference, API server, CLI |
| **Node.js BFF** | `middle/` | TypeScript | API gateway for frontend, auth, file serving, WebSocket, SSE, cron scheduler |
| **Frontend** | `frontend/` | TypeScript/React | Mission Control dashboard, agent management, chat, analytics |

## Architecture Flow

```
Browser/Frontend
    ↓ (HTTP/WS)
Node.js BFF (middle/ - port 3001)
    ↓ (HTTP)
Rust Gateway (api-gateway/ - port 8081) [optional]
    ↓ (HTTP)
Python API Server (src/serving/ - port 8080)
    ↓
Rust Engine (crates/data-engine/)  ←  Node.js NAPI bindings
```

## Rust Crates (11)

| Crate | Type | Purpose |
|-------|------|---------|
| `usage-store` | PyO3 | Python-accessible usage/cost tracking (redb + ulid) |
| `data-engine` | NAPI | Core Node.js data store: agents, activity, notifications, memory, config, logs |
| `token-counter` | NAPI | Approximate token counting, budget computation, truncation |
| `session-manager` | NAPI | User session management with TTL expiry |
| `search-index` | NAPI | Full-text BM25 search with stemming, autocomplete, snippet generation |
| `redis-client` | NAPI | Async Redis client: get/set/hash/list/pubsub, connection pooling |
| `text-processor` | NAPI | Text chunking, stats, keyword extraction, summarization |
| `vector-similarity` | NAPI | Cosine/dot/Euclidean/Manhattan similarity, top-k, normalization |
| `prompt-templates` | NAPI | Template rendering with variables, partials, ChatML/Llama-3 format |
| `json-validator` | NAPI | JSON schema validation, type checking, analysis |
| `uuid-gen` | NAPI | Fast UUID v4/v7 generation, validation, timestamp extraction |
| `api-gateway` | Binary | Standalone reverse proxy with JWT auth, rate limiting, metrics |

## Node.js Middle Layer (17 route modules)

| Route Module | Endpoints | Purpose |
|-------------|-----------|---------|
| `health` | `/health`, `/healthz`, `/readyz`, `/metrics` | Health probes, Prometheus metrics |
| `agents` | `/api/agents/**` | Agent CRUD, state management, command dispatch |
| `activity` | `/api/activity/**` | Activity feed logging and querying |
| `notifications` | `/api/notifications/**` | Notification CRUD, stats, mark-read |
| `config` | `/api/config/**` | Runtime configuration |
| `memory` | `/api/memory/**` | Memory layer management |
| `logs` | `/api/logs/**` | Log querying with level/query filters |
| `chat` | `/api/command`, `/api/events`, `/api/suggestions` | Chat, SSE events, suggestions |
| `models` | `/api/v1/models` | Model listing (proxies to Python API) |
| `plugins` | `/api/plugins` | Plugin management |
| `auth` | `/api/auth/**` | Login, register, API key management |
| `files` | `/api/upload`, `/api/files/**` | File upload/download |
| `sse` | `/api/sse/**` | Server-Sent Events with broadcast |
| `stats` | `/api/stats/**` | Aggregated analytics |
| `system` | `/api/system/**` | System info, env, dependencies |
| `scheduler` | `/api/scheduler/**` | Cron task scheduling |
| `search` | `/api/search/**` | Unified cross-entity search |

## Frontend (23 pages, 49 components, 23 hooks)

| Page | Route | Features |
|------|-------|----------|
| Dashboard | `/` | Metric cards, agent list, activity feed, charts |
| Chat | `/chat` | Streaming chat, WS status, suggestions |
| Analytics | `/analytics` | Requests, latency, status breakdowns |
| Notifications | `/notifications` | Category/priority filters, mark read |
| Skills | `/skills` | Search, activate/deactivate |
| Workflows | `/workflows` | Progress bars, run/stop |
| Memory | `/memory` | Layer browser, search, add entry |
| Tasks | `/tasks` | Cron scheduler, run-now, enable/disable |
| Agents | `/agents` | Comparison table, success rate charts |
| Agent Detail | `/agents/:id` | State control, command exec, metrics |
| Users | `/users` | User list, API key management |
| Logs | `/logs` | Level filter, search, export |
| Health | `/health` | Multi-layer diagnostics |
| API Docs | `/api-docs` | OpenAPI spec viewer |
| Settings | `/settings` | Agent mode, system config, plugins, security |
| Data Explorer | `/data` | Tabbed data browser with DataGrid |
| Login | `/login` | API key authentication |
| 404/500 | `*`, `/500` | Error pages |

## Python Serving Layer (54 modules)

| Module | Purpose |
|--------|---------|
| `api_server.py` | OpenAI-compatible HTTP API with metrics, CORS, graceful shutdown |
| `openapi_spec.py` | OpenAPI 3.1 spec generator + Swagger UI |
| `auth_middleware.py` | API key authentication |
| `cors_middleware.py` | CORS for stdlib HTTPServer |
| `metrics_middleware.py` | Prometheus-format metrics collection |
| `rate_limiter.py` | Token bucket rate limiting |
| `guardrails.py` | Output safety guardrails |
| `streaming.py`, `sse_chat_stream.py` | Streaming inference |

## Development Commands

```bash
make dev          # Python API server
make middle       # Node.js BFF server
make frontend     # Vite dev server
make test         # Python tests
make test-all     # All tests
make docker-up    # All services via Docker Compose
make bootstrap    # One-command full setup
```

## Key Design Decisions

1. **Rust for state management** — The `data-engine` NAPI crate provides in-memory state with dashmap + RwLock, exposed to Node.js. All runtime state (agents, activity, notifications, logs, memory, config) lives here.

2. **Node.js BFF as sole frontend API** — The frontend never talks directly to Python. All API calls go through `middle/` which handles auth, rate limiting, websocket, SSE, file upload, cron scheduling.

3. **Python for model serving only** — The Python API server (`src/serving/api_server.py`) is limited to model inference endpoints (`/v1/chat/completions`, `/v1/models`) with no client-facing business logic.

4. **NAPI over FFI** — Rust crates use `napi-rs` for Node.js integration rather than cffi/pyo3 for Python, because the BFF layer needs fast access to data engine, search, token counting, and session management.

5. **Stateful in-memory** — The data engine is in-memory with optional JSON serialization (`exportJson()`/`importJson()`). For production, add the `redis-client` crate or switch to SQLite via alembic.
