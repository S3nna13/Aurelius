# Aurelius — Frontier AI Research Platform

> 1.395B decoder-only transformer built from scratch — pure PyTorch core, Rust data engine, Node.js BFF, React frontend.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-2024+-dea584.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Every component — transformer core, training pipeline, alignment system, inference engine, API gateway, and frontend — is handwritten. No HuggingFace Transformers, no flash-attn runtime, no bitsandbytes at inference.

---

## Stack

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| Rust Engine | `crates/` | Rust 2024 | Tokenization, search, vector similarity, session management, data engine |
| Python Backend | `src/`, `agent/`, `gateway/` | Python 3.12 | Model, training, inference, alignment, API, CLI |
| Node.js BFF | `middle/` | TypeScript | Auth, rate limiting, WebSocket, SSE, cron, file serving |
| Frontend | `frontend/` | React 19 + TypeScript | Mission Control: dashboard, chat, analytics, admin |

**Data flow:** Browser → Node.js BFF (port 3001) → Python API (port 8080) → AureliusTransformer + Rust NAPI

The frontend never talks directly to Python. All API calls route through the BFF.

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only causal LM |
| Parameters | 1.395B (target) |
| Layers | 24 transformer blocks |
| Hidden dim | 2,048 |
| Attention | Grouped-Query Attention (16 Q heads, 8 KV heads) |
| Head dim | 128 |
| FFN | SwiGLU, d_ff = 5,632 |
| Normalization | Pre-norm RMSNorm |
| Positional encoding | RoPE (θ = 500,000) + YaRN context extension |
| Vocabulary | 8,192 tokens (BPE) |
| Embeddings | Tied input/output |
| KV cache | GQA-compressed; 8 hot-swappable strategies (KIVI, DuoAttention, EVICT, QUEST, Rocket KV, SAGE, TEAL, INT8) |
| MoE | SparseMoELayer — top-2 routing, 8 experts, shared expert, EP load balancing |
| MTP | Multi-Token Prediction (n=2, shared params, staged training) |
| Optimizer | Muon (Newton-Schulz 8+2 steps + Nesterov + RMS rescaling) |
| Checkpoint | safetensors (legacy .pt fallback with deprecation warning) |

---

## Training

- **Full stack:** pretrain → SFT → DPO → GRPO → RLHF — all from scratch
- **Muon optimizer** — hybrid Newton-Schulz orthogonalization (8+2 steps), Nesterov momentum, RMS rescaling; Polar Express (T=6) in flight
- **Liger kernel** — fused RMSNorm, SwiGLU, cross-entropy (~30% throughput uplift)
- **ZClip** — z-score gradient clipping; **BAdam** — block-coordinate fine-tuning
- **Forward replay** — activation checkpointing with selective layer replay
- **Memory-mapped shards** — `.npy` uint16 token arrays, O(log n) shard lookup via `searchsorted`
- **`torch.compile`** — `AureliusTransformer.from_config(config, compile=True)` (CUDA)

**Training configs:** `train_1b.yaml`, `train_2.7b.yaml`, `train_3b.yaml`, `train_moe_5b.yaml`, `yarn_finetune.yaml`

---

## Alignment

**PRAXIS / MOSAIC v2** — 6-signal architecture-aware alignment:
- SteeringRewardCorrespondence (SRC), ExpertSafetyAffinity (ESA), MultiTokenAlignmentHorizon (MTAH)
- PrecisionFusion (Bayesian inverse-variance weighting)
- PRAXISLoss = DAPO + KL penalty + constitutional gate

**Full suite:** REINFORCE++, SAPO, TUR-DPO, AEM, DPO, GRPO, CPO, ORPO, PPO, SimPO, SPIN, KTO, constitutional AI

**MIS-PO** (in flight) — discrete distributional filtering at token level (KL threshold gate) and trajectory level (reward floor), with KL penalty term to keep policy near reference.

---

## Inference & KV Cache

8 hot-swappable KV cache strategies:

| Strategy | Description |
|----------|-------------|
| DuoAttention | Per-head retrieval/streaming classification; JSON config auto-export |
| EVICT (H2O) | Attention-score-based eviction |
| KIVI | INT4/INT8 quantized cache with configurable residual length |
| QUEST | Query-aware sparse KV access |
| Rocket KV | Importance-weighted budget allocation |
| SAGE Attention | SageAttention kernel integration |
| TEAL | Sparsity-based token eviction |
| INT8 Sim | Quantization noise simulation during fine-tuning |

**MTP speculative decoding** (in flight) — draft via MTP heads, single-pass verification → targets 2-3x throughput on long-context sequences.

---

## Agent System

- **ReAct loop** — tool-call parsing, argument validation, budget-bounded termination; AST-walker arithmetic (no dynamic code execution)
- **AbsoluteZero** — self-play curriculum: task proposer + solver in closed feedback loop
- **Planning engine** — workstream DAG, `TaskStatus` / `PlanStatus` StrEnum, `get_workstream(missing_ok)` guard
- **Task scheduler** — cron / interval / delayed jobs; persists to `~/.cache/aurelius/jobs.json`
- **Neuro-symbolic skill** — LLM reasoning on symbolic rule engines
- **Reputation system** — Bayesian multi-agent trust scoring, Sybil resistance
- **13 personas** across 5 domains (GENERAL, CODING, SECURITY, THREAT_INTEL, AGENT); 7 composable facets

---

## Observability

Full production observability stack (`src/observability/`):

| Module | Purpose |
|--------|---------|
| `AgentTelemetry` | High-level facade: audit + metrics + tracing in one call |
| `AuditLogger` | Structured audit trail with retention |
| `EventBus` | In-process async event routing |
| `MetricsCollector` | Counters, histograms, gauges with labels |
| `TraceContext` | W3C-compatible distributed trace propagation |

**SRE metrics** (`src/monitoring/`) — golden signals: latency (p50/p99), error rate, traffic, saturation.
**Prometheus** — `/metrics` endpoint: request counts, latency percentiles, active connecti

### Prometheus Metrics

The `/metrics` endpoint exposes the following counters and gauges:

| Metric | Type | Description |
|--------|------|-------------|
| `aurelius_requests_total` | counter | Total HTTP requests received |
| `aurelius_requests_per_second` | gauge | Current request rate |
| `aurelius_active_connections` | gauge | Concurrent active connections |
| `aurelius_request_duration_ms` | gauge | Request latency p50/p95/p99 |
| `aurelius_uptime_seconds` | gauge | Server uptime |
| `aurelius_http_status_total` | counter | Requests by HTTP status code (label `code`) |
| `aurelius_rate_limit_rejected_total` | counter | Requests rejected by rate limiter |
| `aurelius_validation_failures_total` | counter | Parameter validation failures (out‑of‑range) |
| `aurelius_rate_limiter_backend` | gauge | Rate‑limiter backend (`0`=memory, `1`=redis) |

ons.

---

## Resilience

Production-grade fault-tolerance primitives (`src/resilience/`):

| Pattern | Description |
|---------|-------------|
| `CircuitBreaker` | CLOSED / OPEN / HALF_OPEN FSM; configurable failure threshold + recovery timeout |
| `Bulkhead` | Semaphore-based concurrency cap; isolates subsystem failures |
| `RetryPolicy` | Exponential backoff with jitter, configurable max attempts |
| `RateLimiter` | Token bucket; in-memory (single-node) or Redis (distributed) |
| `Pipeline` | Composable chain: circuit breaker → bulkhead → retry |

---

## Security & Safety

**May 2026 audit — 11 critical, 8 high remediations:**

- Sandbox escape via `object.__subclasses__()` blocked in all in-process execution modules
- SSRF: private/reserved IP blocklist; URL validation moved before `Request()` construction
- Auth middleware default → `require_auth=True` (fail-closed)
- Shell tool: `shell=True` + denylist → `shell=False` + `shlex.split()` + explicit allow-list
- PPO trainer: `prompt_ids` NameError fixed; off-by-one in logit gather corrected
- Constitutional AI: KL divergence argument order corrected (alignment signal was silenced)
- Plugin sandbox: exception → fail-closed `SandboxResult(success=False)`
- CI gates: `continue-on-error: true` removed from all security scan steps

**Ongoing safety:** topology safety (persistent-homology invariants), superposition geometry (polysemanticity detection), 24 adversarial defense modules, jailbreak detector, PII scanner, harm taxonomy (9 categories).

All `torch.load` calls use `weights_only=True`. Container images use non-root users + pinned base-image digests.

---

## API & Serving

**Endpoints:**
- `POST /v1/chat/completions` — streaming + non-streaming, OpenAI-compatible
- `GET /v1/models` — model listing
- `GET /health` — liveness probe
- `GET /metrics` — Prometheus scrape
- `WebSocket /ws` — real-time streaming

**Gateway features:** CSP / HSTS / X-Frame-Options headers, per-IP rate limiting (memory or Redis), 1 MiB / 10 MiB request size limits, `X-Request-ID` tracing, host allow-listing, input parameter validation and response sanitization.

**Health & readiness:**
- `GET /health` — liveness (returns `engine_loaded` flag; 200 only when UP)
- `GET /health/ready` — readiness (waits for model engine initialization)
- Docker `HEALTHCHECK` uses `/health`; Kubernetes probes configure both liveness and readiness.

**Rate limiting backends:**
- **Memory** (default) — in-process token bucket; suitable for single‑instance deployments
- **Redis** (set `AURELIUS_RATE_LIMIT_REDIS_URL`) — distributed Lua‑script token bucket; consistent limits across multiple API replicas

**Deployment targets:** Docker Compose (`deployment/compose.yaml`), Kubernetes (`k8s/`), Helm charts.

---

## Quick Start

```bash
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh         # full setup (Rust + Python + Node)
bash scripts/bootstrap.sh --fast  # skip Rust builds
```

**Prerequisites:** Python 3.12+, Node 22+, Rust 1.81+, npm 10+

### CLI

```bash
aurelius                                      # interactive chat
aurelius chat --persona aurelius-coding       # coding persona
aurelius chat --react --model-path <ckpt>     # ReAct tool-use loop
aurelius serve --port 8080                    # API server
```

### OpenAI Client

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="none")
resp = client.chat.completions.create(
    model="aurelius",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

### Docker

```bash
docker compose up                    # full stack
docker compose up --profile cache    # with Redis
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURELIUS_API_KEY` | — | Single shared API key |
| `AURELIUS_API_KEYS` | — | Multi-key: `id:key:scope1,scope2;...` |
| `AURELIUS_AUTH_ENABLED` | `true` | Require auth on non-loopback |
| `AURELIUS_ALLOWED_HOSTS` | `*` | Comma-separated host allow-list |
| `AURELIUS_RATE_LIMIT` | `60` | Max requests per window per IP |
| `AURELIUS_RATE_WINDOW` | `60` | Rate limit window (seconds) |
| `AURELIUS_RATE_LIMIT_REDIS_URL` | — | Redis URL for distributed rate limiting |
| `AURELIUS_RATE_LIMIT_PREFIX` | `rl:` | Redis key prefix for rate-limit tokens |
| `AURELIUS_MODEL_PATH` | — | Path to checkpoint directory |
| `AURELIUS_VERSION` | `0.1.0` | Version string (visible at `/health`) |

---

## DAIES Scaling Plan

| Phase | Params | Active | Strategy | Status |
|-------|--------|--------|----------|--------|
| v1 | 1.395B | 1.395B | Muon + grad_ckpt, bs=4 | Training in progress |
| v2 | 2.7B | 2.7B | Muon + grad_ckpt, bs=1 | Planned |
| v3 | 3.0B | 3.0B | 8-bit optim + MLX | Planned |
| v4 | ~5B MoE | ~2B | Sparse MoE + expert offload | Planned |
| v5 | 7-14B | 7-14B | bf16 / 4-bit quant | Future |
| v6 | 32B | ~8B MoE | Expert parallelism, distributed | Future |

Dense checkpoints seed MoE experts via `src/model/moe_upcycle.py`. GGUF Q4_K_M export targets 25-35 tok/s on Apple Silicon.

---

## Directory Structure

```
Aurelius/
├── src/
│   ├── model/           # Transformer, GQA, RoPE, SwiGLU, MoE, MTP — 200+ modules
│   ├── training/        # Muon, ZClip, BAdam, curriculum, RLHF trainers
│   ├── alignment/       # PRAXIS/MOSAIC v2, DPO, GRPO, PPO, MIS-PO
│   ├── inference/       # 8 KV cache strategies, speculative decoding, sampling
│   ├── agent/           # ReAct, AbsoluteZero, tool parser, planner
│   ├── persona/         # 13 personas, 7 facets, routing
│   ├── memory/          # MemCoE, semantic + episodic + unified orchestrator
│   ├── retrieval/       # BM25 + dense hybrid + re-ranking
│   ├── safety/          # Jailbreak, topology safety, superposition geometry, PII
│   ├── security/        # GCG adversarial, backdoor scan, MITRE ATT&CK
│   ├── observability/   # Telemetry, audit, event bus, metrics, trace
│   ├── resilience/      # Circuit breaker, bulkhead, retry, rate limiter, pipeline
│   ├── monitoring/      # SRE golden signals
│   ├── interpretability/# SAEs, activation patching, probing
│   ├── quantization/    # AWQ, GPTQ, SmoothQuant, NF4, FP8
│   └── reasoning/       # MCTS, chain-of-thought, structured reasoning
├── agent/               # Planning engine, task scheduler (canonical namespace)
├── gateway/             # FastAPI server + rate limiting + metrics middleware
├── aurelius_cli/        # CLI entry points, pipeline, scheduler commands
├── middle/              # Node.js BFF (TypeScript)
├── frontend/            # React 19 + Vite + TypeScript
├── crates/              # Rust NAPI-rs (11 crates: data-engine, token-counter, search, etc.)
├── k8s/                 # Kubernetes manifests (Deployment, Service)
├── deployment/          # Docker Compose, Helm charts
├── configs/             # Training YAML configs
├── examples/            # Runnable scripts (scheduler, pipeline, SRE metrics)
├── scripts/             # Bootstrap, benchmark, GGUF export, data prep
├── tests/               # 33,000+ tests across all surfaces
├── data/                # Training shards (.npy uint16), tokenizer, corpus
└── checkpoints/         # Saved checkpoints (safetensors)
```

---

## Testing

```bash
make test           # Python backend
make test-cov       # With coverage
make frontend-test  # Vitest
make middle-test    # Node.js BFF
make rust-test      # Rust crates
make test-all       # All surfaces
make ci             # lint + typecheck + security + all tests
```

---

## Entry Points

| Command | Description |
|---------|-------------|
| `aurelius` | Interactive chat CLI |
| `aurelius-cli` | Terminal chat with conversation management |
| `aurelius-shell` | REPL with slash commands |
| `aurelius-api` | Python API server |
| `aurelius-server` | Production serving stack |

---

## Documentation

| Document | Description |
|----------|-------------|
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Architecture card |
| [docs/threat_model.md](docs/threat_model.md) | Security threat model |
| [examples/](examples/) | Runnable example scripts |

---

[MIT License](LICENSE) — Copyright © 2025 Aurelius Systems, Inc.

**GitHub:** [https://github.com/S3nna13/Aurelius](https://github.com/S3nna13/Aurelius)
