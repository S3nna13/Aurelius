# Aurelius — Frontier AI Research Platform

> **1.395B decoder-only transformer** trained from scratch — pure PyTorch core, Rust data engine, Node.js BFF, React frontend.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-2024+-dea584.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What Is Aurelius?

Aurelius is a four-layer full-stack AI research platform built entirely from scratch. Every component — the transformer core, training pipeline, alignment system, inference engine, API server, and frontend — is handwritten. No HuggingFace Transformers, no flash-attn runtime dependency, no bitsandbytes at inference time.

The architecture targets a 1.395B parameter decoder-only transformer with Grouped-Query Attention, SwiGLU FFN, RoPE/YaRN positional encoding, and a Sparse Mixture-of-Experts variant — all implemented in pure PyTorch.

---

## Architecture Overview

Aurelius is organized into four independent layers, each owning a distinct responsibility:

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| **Rust Engine** | `crates/`, `tools/` | Rust 2024 | Data engine, tokenization, search, vector similarity, session management |
| **Python Backend** | `src/`, `agent/`, `gateway/`, `aurelius_cli/` | Python 3.12 | Model training, inference, API server, alignment, CLI |
| **Node.js BFF** | `middle/` | TypeScript | API gateway, auth, WebSocket, SSE, cron scheduling |
| **Frontend** | `frontend/` | TypeScript / React 19 | Mission Control dashboard, chat, analytics, admin |

### Data Flow

```
Browser / Frontend
    │  HTTP / WebSocket
    ▼
Node.js BFF (middle/ — port 3001)
    │  HTTP proxy
    ▼
Python API Server (gateway/ — port 8080)
    │
    ├── Model inference (AureliusTransformer)
    ├── ReAct agentic loop
    └── Rust NAPI bindings (token counting, search, sessions)
```

The frontend **never** talks directly to Python. All API calls route through the Node.js BFF, which handles auth, rate limiting, WebSocket multiplexing, SSE, file upload, and cron scheduling.

---

## Model Architecture

The Aurelius transformer (`src/model/transformer.py`, `src/model/attention.py`) implements:

| Component | Specification |
|-----------|--------------|
| Architecture | Decoder-only, causal language model |
| Parameters | 1.395B (target) |
| Layers | 24 transformer blocks |
| Hidden dim | 2,048 (`d_model`) |
| Attention | Grouped-Query Attention — 16 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, `d_ff = 5,632` |
| Normalization | Pre-norm RMSNorm |
| Position encoding | RoPE (`θ = 500,000`) with YaRN context extension |
| Vocabulary | 8,192 tokens (BPE tokenizer) |
| Tied embeddings | Yes (input/output weight sharing) |
| Dropout | 0.0 (inference-safe) |

### In-Flight Architecture Upgrades (Step 3.5 Flash paper, arXiv:2602.10604)

The following upgrades are being integrated from the Step 3.5 Flash architecture:

| Upgrade | Target file | Status |
|---------|-------------|--------|
| Shared expert in MoE (always-firing alongside top-k routed) | `src/model/moe.py` | In progress |
| S3F1 hybrid attention (3 SWA : 1 full per 4-layer block) | `src/model/transformer.py` | In progress |
| Polar Express Muon (T=6 float16 refinement after Newton-Schulz) | `src/training/muon.py` | In progress |
| Fast-MTP (position-dependent loss reweighting) | `src/model/mtp.py` | In progress |
| Staged MTP training (MTP-1 warmup → clone to MTP-2/3) | `src/training/trainer.py` | In progress |
| EP-level load balancing (ℒ_EP = G∑f_g×p_g) | `src/model/moe.py` | In progress |
| MoE activation clipping + expert norm monitoring | `src/model/moe.py` | In progress |
| MIS-PO alignment (token + trajectory distributional filtering) | `src/alignment/mispo.py` | In progress |
| MTP speculative decoding (draft from MTP heads, verify in one pass) | `src/inference/mtp_speculative.py` | In progress |
| Progressive batch schedule (4k → 8k → 12k → 16k tokens) | `src/training/trainer.py` | In progress |
| Context length schedule (4k → 32k → 128k mid-training) | `src/training/trainer.py` | In progress |

---

## Key Features

### Training Pipeline

- **Muon optimizer** — hybrid Newton-Schulz (8+2 steps) + Nesterov momentum + RMS rescaling; Polar Express iteration (T=6) in flight
- **Full training stack** — pretrain → SFT → DPO → GRPO → RLHF, all from scratch
- **Multi-Token Prediction** — `MTPModule` with `n_predict=2`, shared parameters; staged training protocol in flight
- **Liger kernel integration** — fused RMSNorm, SwiGLU, cross-entropy for ~30% throughput uplift
- **ZClip** — gradient clipping with z-score normalization
- **BAdam** — block-coordinate optimizer for memory-efficient fine-tuning
- **Forward replay** — activation checkpointing with selective replay for memory-efficient backprop
- **`torch.compile` support** — `AureliusTransformer.from_config(config, compile=True)` (requires CUDA)
- **Training shards** — memory-mapped `.npy` uint16 shards via `TokenizedShardDataset`; O(log n) shard lookup

### Alignment (PRAXIS / MOSAIC v2)

- **PRAXIS trainer** — 6-signal MOSAIC v2 architecture-aware alignment framework
  - SteeringRewardCorrespondence (SRC), ExpertSafetyAffinity (ESA), MultiTokenAlignmentHorizon (MTAH)
  - PrecisionFusion (Bayesian inverse-variance weighting), PRAXISLoss (DAPO + KL + constitutional gate)
- **REINFORCE++** — variance-reduced policy gradient with baseline correction
- **SAPO** — Self-Adaptive Policy Optimization with dynamic KL weighting
- **TUR-DPO** — Trust-Uncertainty-Regularized DPO for distribution-aware preference learning
- **AEM** — Adaptive Entropy Masking for controlled generation diversity
- **MIS-PO** (in flight) — discrete distributional filtering at token and trajectory level
- **Full suite** — DPO, GRPO, CPO, ORPO, PPO, SimPO, SPIN, KTO, constitutional AI

### Inference & KV Cache

- **8 KV cache strategies** — DuoAttention, EVICT (H2O-style), KIVI quantization, QUEST attention, RMR, Rocket KV, SAGE attention, TEAL sparsity
- **DuoAttention head classification** — auto-detects retrieval vs. streaming heads; JSON config export at `configs/duo_attention_heads.json`
- **MTP speculative decoding** (in flight) — draft tokens from MTP heads, verify in a single forward pass; expected 2-3x throughput on long sequences
- **KIVI KV cache** — INT4/INT8 quantized KV cache with configurable residual length
- **INT8 KV quantization** — simulates quantization noise during fine-tuning for quantization-aware training
- **OOD pathway** — out-of-distribution detection and routing for robustness

### Agent System

- **ReAct loop** (`src/inference/agentic_loop.py`) — tool-call parsing, argument validation, budget-bounded termination; AST-walker arithmetic evaluator (secure, no dynamic code execution)
- **AbsoluteZero loop** — self-play curriculum: task proposer + solver in closed feedback loop
- **Neuro-symbolic skill** — LLM reasoning grafted onto symbolic rule engines
- **Reputation system** — multi-agent trust scoring with Bayesian update and Sybil resistance
- **13 unified personas** across 5 domains (GENERAL, CODING, SECURITY, THREAT_INTEL, AGENT)
- **7 composable facets** — security, threat_intel, agent_mode, constitution, harm_filter, personality, dialogue

### Memory & Retrieval

- **MemCoE** — Memory-Conditioned Expert gating: routes tokens through memory-specialized MoE experts
- **Unified memory orchestrator** — semantic + episodic + working memory with coherent recall
- **Retrieval pipeline** — end-to-end BM25 + dense hybrid retrieval with re-ranking

### Security & Safety

- **Topology safety** — persistent-homology invariant checking across activation manifolds
- **Superposition geometry** — polysemanticity detection via interference-angle analysis
- **24 security modules** — gradient inversion defense, GCG adversarial search, prompt injection detector, STRIP backdoor scan
- **Jailbreak detection** and output filtering
- **PII scanner** and harm taxonomy (9 categories)
- **SSRF-hardened** HTTP backend (`_validate_backend_url()` called before `Request()` construction)
- All production `torch.load` calls enforce `weights_only=True`
- Container hardening: non-root users and base-image digests (see `SECURITY.md`)

### Serving & Deployment

- **OpenAI-compatible API** — drop-in replacement (`/v1/chat/completions`, `/v1/models`)
- **Prometheus metrics** — `/metrics` endpoint scrapes request counts, latency percentiles, active connections
- **Security headers** — CSP, HSTS, X-Frame-Options, X-Content-Type injected at gateway level
- **Per-IP rate limiting** — configurable via `AURELIUS_RATE_LIMIT` / `AURELIUS_RATE_WINDOW`
- **Request tracing** — `X-Request-ID` header for end-to-end correlation
- **Docker Compose** — `deployment/compose.yaml` for single-command deployment
- **Helm charts** — Kubernetes deployment

### Frontend (Mission Control)

- **Dashboard, chat, analytics, workflows, admin surfaces**
- **Real-time streaming** — WebSocket chat, SSE events, live metrics
- **Backend switcher** — target `Auto`, `mock`, `agentic` backends; Settings persist in localStorage
- **Agent Chat** (default) — routes through `/api/chat/agent`, dispatches to best-fit agent (Coding, Research, General) based on content
- **Model Chat** — sends directly to `/api/chat/completions`

---

## Quick Start

### Prerequisites

```
node >= 22       # Frontend + BFF
npm >= 10        # Package management
rustc >= 1.81    # Rust data engine
python >= 3.12   # Python backend
```

### Installation

```bash
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh
```

For a faster setup that skips the Rust build step:

```bash
bash scripts/bootstrap.sh --fast
```

Or via Makefile:

```bash
make setup-dev
```

### Running the CLI

```bash
aurelius                                               # Interactive chat
aurelius chat                                          # Same as above
aurelius chat --persona aurelius-coding                # Coding persona
aurelius chat --model-path checkpoints/my_run/         # Load trained weights
aurelius chat --react --model-path checkpoints/my_run/ # ReAct tool-use loop
aurelius serve                                         # Start API server
aurelius serve --engine agentic --model-path <ckpt>   # ReAct API backend
aurelius serve --port 8080                             # Custom port
aurelius --version
```

### Developer Utilities

#### Task Scheduling — `agent.task_scheduler`

```python
from agent.task_scheduler import TaskScheduler

sched = TaskScheduler()
sched.schedule_cron("0 2 * * *", daily_backup)   # daily at 2am
sched.schedule_interval(60.0, heartbeat)          # every 60s
sched.schedule_delayed(300, cleanup)              # one-shot in 5min
sched.start()
```

CLI equivalents:
```bash
aurelius schedule cron "0 2 * * *" -- python backup.py
aurelius schedule interval 60 -- python heartbeat.py
aurelius schedule once 300 -- python cleanup.py
aurelius schedule list
aurelius schedule cancel <job_id>
```

Jobs persist to `~/.cache/aurelius/jobs.json` and survive process restarts.

#### Data Pipeline — `aurelius_cli.pipeline_processor`

```python
from aurelius_cli.pipeline_processor import Pipeline

result = (
    Pipeline(items)
    .filter(lambda x: x.active)
    .map(lambda x: x.normalize())
    .sort(key=lambda x: x.priority, reverse=True)
    .head(10)
    .collect()
)
```

CLI one-liner:
```bash
cat users.jsonl | aurelius pipeline \
    --filter "x['active'] == True" \
    --map "x['name'].upper()" \
    --sort "x['score']" --reverse \
    --head 5
```

#### SRE Metrics — `src.monitoring.sre_metrics`

```python
from src.monitoring.sre_metrics import SREMetricsCollector

metrics = SREMetricsCollector(service="aurelius-api")
metrics.record_request(latency_ms=120, success=True)
metrics.record_error(error_type="timeout")
print(metrics.summary())
# {total_requests, error_rate, p50_latency_ms, p99_latency_ms, saturation}
```

```bash
aurelius metrics demo --requests 200 --error-rate 0.05 --latency-mean 120
aurelius metrics demo --requests 200 --json
```

---

## Running the API

```bash
# Python API server (OpenAI-compatible)
aurelius-api --host 127.0.0.1 --port 8080

# Integrated API + web UI
aurelius serve --host 127.0.0.1 --port 8080

# Makefile shortcuts
make dev       # Python API server
make middle    # Node.js BFF
make frontend  # Vite frontend dev server

# Full stack via Docker
docker compose -f deployment/compose.yaml up
```

### OpenAI-Compatible Usage

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="aurelius",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURELIUS_API_KEY` | — | Single API key for server authentication |
| `AURELIUS_API_KEYS` | — | Multi-key store: `id:key:scope1,scope2;...` |
| `AURELIUS_AUTH_ENABLED` | `true` | Require API key on non-loopback interfaces |
| `AURELIUS_ALLOWED_HOSTS` | `*` | Comma-separated host allow-list |
| `AURELIUS_RATE_LIMIT` | `60` | Max requests per window per IP |
| `AURELIUS_RATE_WINDOW` | `60` | Rate limit window in seconds |
| `AURELIUS_VERSION` | `0.1.0` | Server version string (visible at `/health`) |

---

## Entry Points

| Command | Module | Description |
|---------|--------|-------------|
| `aurelius` | `aurelius_cli/main.py` | Interactive chat CLI with persona support |
| `aurelius-cli` | `aurelius_cli/terminal_cli.py` | Terminal chat with full conversation management |
| `aurelius-shell` | `aurelius_cli/aurelius_shell.py` | REPL shell with slash commands |
| `aurelius-api` | `src/backend.py` | Python API server (model inference) |
| `aurelius-server` | `gateway/aurelius_api.py` | Production serving stack |

---

## Security Audit — May 2026

A full-stack security, correctness, and CI/CD audit was performed. **11 critical** and **8 high-severity** findings were remediated.

### Critical Remediations

| ID | File | Issue | Fix |
|----|------|-------|-----|
| C1 | `.github/workflows/deploy.yml` | Registry credentials logged in CI shell | Moved secrets to `env:` block |
| C2 | `.github/workflows/deploy.yml` | `GITHUB_TOKEN` had `write-all` permissions | Scoped to `contents: read / packages: write` |
| C3 | `aurelius/plugin_system.py` | Star-import from `src.agent.plugin_system` | Replaced with explicit named imports + `__all__` |
| C4 | `aurelius/skills_registry.py` | Star-import from `src.agent.skills_registry` | Replaced with explicit named imports + `__all__` |
| C5 | `src/backends/credential_manager.py` | Credential stuck in `REFRESHING` state on error | Removed premature status mutation |
| C6 | `src/agent/react_loop.py` | Lambda passed to `ProcessPoolExecutor` — not serializable, silent crash | Switched to `ThreadPoolExecutor` with direct function reference |
| C7 | `src/agent/tool_registry_dispatcher.py` | `_wall_start` race condition (TOCTOU) | Moved assignment inside `_budget_lock` |
| C8 | `src/alignment/ppo_trainer.py` | `prompt_ids` `NameError` made PPO non-functional | Threaded `prompt_ids` through rollout dict |
| C9 | `src/alignment/ppo_trainer.py` | Off-by-one in logit gather skipped first token's log-prob | Fixed causal slice: `logits[:, prompt_len-1 : prompt_len-1+T, :]` |
| C10 | `src/alignment/constitutional_ai.py` | KL divergence arguments swapped — alignment signal silenced | Corrected argument order in `F.kl_div()` |
| C11 | `src/agent/plugin_sandbox.py` | Sandbox fail-open on exception | Fail-closed: exception returns `SandboxResult(success=False)` |

### High-Severity Remediations

| ID | File | Issue | Fix |
|----|------|-------|-----|
| H1 | `src/backends/circuit_breaker.py` | Dead code after `return True` | Removed 9 unreachable lines |
| H2 | `src/backends/circuit_breaker.py` | `reset()` mutated shared state without lock | Added `with self._lock:` |
| H3 | `src/backends/async_rate_limiter.py` | `reset()` / `get_remaining()` bypassed asyncio lock | Made both methods `async` |
| H4 | `src/alignment/rlvr.py` | NaN risk from zero-std reward normalization | `std.clamp(min=1e-8)` guard |
| H5 | `src/backends/http_backend.py` | SSRF TOCTOU — URL validated after `Request()` construction | Moved validation before object construction |
| H6 | `src/alignment/rlvr.py` + `grpo.py` | Negative KL possible with mismatched distributions | `.clamp(min=0.0)` on all KL terms |
| H7 | `src/alignment/dapo.py` | Asymmetric PPO clip applied uniformly — broke DAPO invariant | `torch.where(advantages >= 0, ...)` for correct clipping |
| H8 | `.github/workflows/ci.yml` | `continue-on-error: true` on Bandit/pip-audit made security gates cosmetic | Removed; CI now fails on security findings |

### CI/CD Hardening

- Ruff auto-fix workflow excluded from `main` branch; version pinned
- `torch`, `flash-attn`, `deepspeed` upper-bounded in `pyproject.toml` to prevent silent major-version breakage
- MCP audit log path set in `configs/config.yaml`
- Auth and rate limiting enabled by default in `.env.example`

---

## DAIES Scaling Plan

Aurelius follows the **DAIES** (Doubling AI Efficiency Simultaneously) scaling philosophy.

| Phase | Params | Strategy | Memory (M1 Pro 32GB) | Status |
|-------|--------|----------|----------------------|--------|
| **v1** | 1.395B | Muon+AdamW, grad_ckpt, bs=4 | ~12.8GB | Architecture complete; training in progress |
| **v2** | 2.7B | Muon+AdamW, grad_ckpt, bs=1 | ~17.3GB | Planned |
| **v3** | 3.0B | 8-bit optim, MLX, grad_ckpt, bs=1 | ~20.5GB | Planned |
| **v4** | 5-6B MoE | Sparse MoE, expert offloading, ~2B active | ~18-20GB | Planned |
| **v5** | 7-14B | bf16 / 4-bit quant | 14-28GB | Future |
| **v6** | 32B | MoE with expert parallelism, distributed | Cluster | Future |

Key principles:
- Every layer is handwritten — no black-box dependencies at runtime
- Memory-budgeted design — each phase fits within 32GB Apple Silicon or a single H100
- MoE upcycle — dense checkpoints seed sparse experts via `src/model/moe_upcycle.py`
- Q4_K_M GGUF export for local serving at 25-35 tok/s

---

## Directory Structure

```
Aurelius/
├── src/                          # Python backend
│   ├── model/                    # Transformer core (GQA, RoPE, SwiGLU, MoE, MTP, SSMs)
│   ├── training/                 # Muon, ZClip, curriculum, RLHF trainers
│   ├── alignment/                # PRAXIS/MOSAIC v2, DPO, GRPO, PPO, MIS-PO (in flight)
│   ├── inference/                # KV cache (DuoAttention, EVICT, KIVI, QUEST, Rocket KV, SAGE, TEAL)
│   │                             #   + MTP speculative decoding (in flight)
│   ├── persona/                  # 13 personas, 7 facets, routing, prompt composition
│   ├── agent/                    # ReAct loop, AbsoluteZero, tool parser, planner
│   ├── chat/                     # ChatML, Llama-3 templates, conversation management
│   ├── serving/                  # OpenAI-compatible API, streaming, metrics
│   ├── safety/                   # Jailbreak detector, topology safety, superposition geometry, PII scanner
│   ├── security/                 # Adversarial defense, backdoor scan, MITRE ATT&CK
│   ├── interpretability/         # Activation patching, SAEs, probing, circuit discovery
│   ├── eval/                     # Benchmarks, scorers, calibration
│   ├── data/                     # Data processing, tokenization, curriculum
│   ├── longcontext/              # KV quantization, StreamingLLM, attention sinks
│   ├── retrieval/                # BM25 + dense hybrid retrieval with re-ranking
│   ├── reasoning/                # MCTS, chain-of-thought, structured reasoning
│   ├── memory/                   # MemCoE, semantic + episodic + unified orchestrator
│   ├── multiagent/               # Multi-agent coordination, Bayesian reputation system
│   ├── quantization/             # AWQ, GPTQ, SmoothQuant, NF4, FP8
│   └── monitoring/               # SRE metrics (golden signals: latency, errors, traffic, saturation)
├── agent/                        # Top-level agent code (canonical namespace)
├── gateway/                      # Python API server (canonical namespace)
├── aurelius_cli/                 # CLI entry points
├── middle/                       # Node.js BFF (TypeScript)
│   └── src/
│       ├── routes/               # API route modules
│       ├── middleware/           # Auth, CORS, rate limiting, logging
│       └── ws/                   # WebSocket hub
├── frontend/                     # React 19 + Vite + TypeScript
│   └── src/
│       ├── pages/                # Dashboard, Chat, Analytics, admin
│       ├── components/           # Shared UI components
│       └── hooks/                # Custom hooks and data helpers
├── crates/                       # Rust workspace (NAPI-rs native bindings for Node.js)
│   ├── data-engine/              # Core in-memory data store (dashmap + RwLock)
│   ├── api-gateway/              # Standalone reverse proxy binary
│   ├── token-counter/            # Approximate token counting
│   ├── session-manager/          # User session management with TTL
│   ├── search-index/             # BM25 full-text search
│   ├── vector-similarity/        # Cosine / dot / Euclidean similarity
│   ├── redis-client/             # Async Redis client
│   ├── text-processor/           # Text chunking, stats, keyword extraction
│   ├── prompt-templates/         # ChatML / Llama-3 template rendering
│   ├── json-validator/           # JSON schema validation
│   └── uuid-gen/                 # UUID v4/v7 generation
├── configs/                      # Training configs (1.4B, 2.7B, 3B, MoE-5B, YaRN fine-tune)
├── scripts/                      # Benchmark, bootstrap, GGUF export, profiling
├── tests/                        # Test suite (33,000+ tests across all surfaces)
├── deployment/                   # Docker, Docker Compose, Helm charts
├── docs/                         # Documentation
├── tools/                        # Rust CLI tools (data-cli, jsonl-merge)
├── data/                         # Training shards (.npy uint16), tokenizer, reference corpus
├── training_data/                # Curated training datasets
├── examples/                     # Runnable scripts (task scheduling, pipeline, SRE metrics)
├── checkpoints/                  # Saved model checkpoints (safetensors format)
├── alembic/                      # Database migrations
└── plugins/                      # Plugin system
```

---

## Testing

```bash
# Python backend
make test                # Run test suite (excludes legacy/)
make test-cov            # With coverage report

# Frontend (Vitest)
make frontend-test

# Node.js BFF
make middle-test

# All surfaces
make test-all

# Rust crates
make rust-test

# Full CI locally
make ci
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [LICENSE](LICENSE) | MIT License |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Model architecture card |
| [docs/dataset_card.md](docs/dataset_card.md) | Dataset documentation |
| [docs/eval_card.md](docs/eval_card.md) | Evaluation methodology |
| [docs/threat_model.md](docs/threat_model.md) | Security threat model |
| [docs/plans/](docs/plans/) | Design docs, handoff notes, implementation plans |

---

## License

Released under the [MIT License](LICENSE).

---

**GitHub:** [https://github.com/S3nna13/Aurelius](https://github.com/S3nna13/Aurelius)

*Built with pure PyTorch. No compromises.*
