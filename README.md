# Aurelius — Frontier AI Coding Platform

> **From 1.3B to 32B parameters** — a four-layer full-stack AI platform with pure-PyTorch transformer core, Rust data engine, Node.js BFF, and React frontend.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-2024+-dea584.svg)](https://www.rust-lang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4+-06b6d4.svg)](https://tailwindcss.com/)
[![License: Aurelius Open License](https://img.shields.io/badge/License-Aurelius%20Open%20License-green.svg)](LICENSE)

---

## Architecture Overview

> ![Aurelius architecture diagram](docs/aurelius-architecture.svg)
> *Open interactive version: [docs/architecture.html](docs/architecture.html)*

Aurelius is organized into four independent layers, each owning a distinct responsibility:

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| **Rust Engine** | `crates/`, `tools/` | Rust 2024 | High-performance data engine, search, tokenization, vector similarity, caching, auth sessions |
| Python Backend | `src/`, `configs/`, `agent/`, `gateway/`, `aurelius_cli/` | Python 3.12 | Model training, inference, API server, CLI |
| **Node.js BFF** | `middle/` | TypeScript | API gateway for frontend, auth, file serving, WebSocket, SSE, cron scheduler |
| **Frontend** | `frontend/` | TypeScript / React 19 | Mission Control dashboard, agent management, chat, analytics |

### Data Flow

```
Browser / Frontend
    |  HTTP / WebSocket
    v
Node.js BFF (middle/ — port 3001)
|    |  HTTP proxy                                    v
v                              v
Rust Engine (crates/data-engine/)  Python API Server (gateway/ — port 8080)
    |                                    |
    | in-memory state (dashmap+RwLock)   | model inference, persona routing
    v                                    v
JSON export / Redis / SQLite            Trained checkpoints (safetensors)
```

The frontend **never** talks directly to Python. All API calls route through the Node.js BFF, which handles auth, rate limiting, WebSocket multiplexing, SSE, file upload, and cron scheduling. The Python server exposes model inference endpoints (`/v1/chat/completions`, `/v1/models`) and can run either a plain generation backend or the ReAct agentic backend via `--engine agentic`.

---

## Key Features & Capabilities

### Model (Pure PyTorch)
- **1.395B decoder-only transformer** — GQA, RoPE/YaRN, SwiGLU, RMSNorm
- **~150 research architecture modules** (43 actively used; 211 archived experiments)
- **Full training pipeline** — pretrain → SFT → DPO → RLHF, all from scratch
- **No HF Transformers / flash-attn / bitsandbytes / DeepSpeed at runtime**

### Agent System
- **ReAct loop** with tool-call parsing, argument validation, budget-bounded termination
- **13 unified personas** across 5 domains (GENERAL, CODING, SECURITY, THREAT_INTEL, AGENT)
- **7 composable facets** — security, threat_intel, agent_mode, constitution, harm_filter, personality, dialogue
- **Multi-step chaining** with timeout controls

### Security & Safety
- **24 security modules** — gradient inversion defense, GCG adversarial search, prompt injection detector, STRIP backdoor scan
- **Jailbreak detection** and output filtering
- **PII scanner** and harm taxonomy (9 categories)
- **SSRF-hardened** HTTP backend
- All production `torch.load` calls now enforce `weights_only=True`. Unit‑test exception noted.
- **Container hardening**: non‑root users and base‑image digests (see `SECURITY.md`).
- **CI security**: Trivy scans for high/critical CVEs on built images.


### Frontend (Mission Control)
- **Dashboard, chat, analytics, workflows, and admin surfaces**
- **Real-time streaming** — WebSocket chat, SSE events, live metrics
- **Shared components and hooks** — React 19 + Vite + TypeScript + Tailwind CSS
- **Backend switcher** — Mission Control Playground and Chat can target `Auto`, `mock`, `vLLM`, or `agentic`; Settings stores the default backend and upstream URLs used by `Auto`
  - **Agent Chat** (default mode) — Routes requests through `/api/chat/agent`, which dispatches to the best-fit agent (Coding, Research, or General) based on message content
  - **Model Chat** — Sends requests directly to `/api/chat/completions`; when backend is `Auto`, the BFF resolves it to the **Default Backend** saved in Settings; explicit backends (`mock`, `vLLM`, `agentic`) override the default
  - **Auto resolution** — When `Auto` is selected, the BFF reads `chat.backend` from config (defaults to `vllm`); invalid backend values are normalized to the config default
  - Backend and mode choices persist in `localStorage` across page refreshes

### Serving & Deployment
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **ReAct agentic backend** — enable tool-using chat loops with `--engine agentic`
- **vLLM paged attention engine** — GPU-accelerated inference with block-wise KV cache management
  - `--engine vllm --model-path <model> --speculative-decoding --tensor-parallel-size N`
  - FP8 / INT8 / AWQ / GPTQ quantization via `--quantization fp8|int8|awq|gptq`
- **Speculative decoding** — Eagle / Medusa heads for 2-3x throughput via `--speculative-decoding`
- **Continuous batching** and paged KV cache
- **GPU Dockerfile** — `deployment/Dockerfile.gpu` with multi-stage Rust + vLLM build
- **Docker Compose GPU** — `deployment/compose.gpu.yaml` for single-command GPU deployment
- **Helm charts** for Kubernetes deployment

```bash
# GPU production deployment
docker compose -f deployment/compose.gpu.yaml up --build

# vLLM inference with speculative decoding
python -m gateway.aurelius_api \
  --engine vllm \
  --model-path aurelius-ai/aurelius-1b \
  --gpu-memory-utilization 0.90 \
  --speculative-decoding \
  --n-spec-tokens 5

# ReAct agentic backend
python -m gateway.aurelius_api \
  --engine agentic \
  --model-path aurelius-ai/aurelius-1b

# Quantized INT8 serving
python -m gateway.aurelius_api \
  --engine vllm \
  --model-path aurelius-ai/aurelius-1b \
  --quantization int8
```

---

## Quick Start

### Prerequisites

```bash
node >= 22       # Frontend + BFF
npm >= 10        # Package management
rustc >= 1.81    # Rust data engine
python >= 3.12   # Python backend
```

### Installation

```bash
# Clone and bootstrap the repo
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh
```

`bootstrap.sh` installs the Python, Node.js, and Rust dependencies used by the
repo and performs the optional native builds. For a faster setup that skips the
Rust build step, use:

```bash
bash scripts/bootstrap.sh --fast
```

If you prefer Makefile targets, `make setup-dev` installs the Python backend,
frontend, middle layer, and pre-commit hooks.

```bash
make setup-dev
```

### Running the CLI

```bash
aurelius                      # Interactive chat (default)
aurelius chat                 # Same as above
aurelius chat --persona aurelius-coding   # Use coding persona
aurelius chat --model-path <ckpt>  # Load trained weights
aurelius chat --react --model-path <ckpt>  # Enable ReAct tool-use loop
aurelius serve                 # Start API server + web UI
aurelius serve --engine agentic --model-path <ckpt>  # ReAct API backend
aurelius serve --port 8080     # Custom API port
aurelius --version             # Print version
```

### Running the API

```bash
# Start the OpenAI-compatible Python server
aurelius-api --host 127.0.0.1 --port 8080

# Start the integrated API + web UI stack
aurelius serve --host 127.0.0.1 --port 8080 --ui-port 7860

# Or use the Makefile shortcuts
make dev        # Python API server only
make middle     # Node.js BFF only
make frontend   # Vite frontend only

# Or everything at once with Docker
docker compose up
docker compose --profile inference up   # with LLM inference
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

---

## Entry Points

| Command | Module | Description |
|---------|--------|-------------|
| `aurelius` | `src/cli/main.py` | Interactive chat CLI with persona support |
| `aurelius-cli` | `src/cli/terminal_cli.py` | Terminal chat with full conversation management |
| `aurelius-shell` | `src/cli/aurelius_shell.py` | REPL shell with slash commands |
| `aurelius-api` | `src/backend.py` | Python API server (model inference) |
| `aurelius-server` | `src/serving/aurelius_api.py` | Production serving stack |

---

## Security Audit — May 2026

A full-stack security, correctness, and CI/CD audit was performed against this codebase. **11 critical** and **8 high-severity** findings were remediated.

### Critical Remediations

| ID | File | Issue | Fix |
|----|------|-------|-----|
| C1 | `.github/workflows/deploy.yml` | Registry credentials logged in CI shell | Moved secrets to `env:` block; scripts use `$ENV_VAR` references |
| C2 | `.github/workflows/deploy.yml` | `GITHUB_TOKEN` had `write-all` permissions | Scoped to `contents: read / packages: write` |
| C3 | `aurelius/plugin_system.py` | `from src.agent.plugin_system import *` star-import | Replaced with explicit named imports + `__all__` |
| C4 | `aurelius/skills_registry.py` | `from src.agent.skills_registry import *` star-import | Replaced with explicit named imports + `__all__` |
| C5 | `src/backends/credential_manager.py` | Credential stuck in `REFRESHING` state on error | Removed premature status mutation; wrapped `NotImplementedError` |
| C6 | `src/agent/react_loop.py` | Lambda passed to `ProcessPoolExecutor` — not picklable, silent crash | Switched to `ThreadPoolExecutor` with direct function reference |
| C7 | `src/agent/tool_registry_dispatcher.py` | `_wall_start` race condition (TOCTOU) | Moved assignment inside `_budget_lock` |
| C8 | `src/alignment/ppo_trainer.py` | `prompt_ids` `NameError` made PPO trainer non-functional | Threaded `prompt_ids` through rollout dict |
| C9 | `src/alignment/ppo_trainer.py` | Off-by-one in logit gather skipped first token's log-prob | Fixed causal slice: `logits[:, prompt_len-1 : prompt_len-1+T, :]` |
| C10 | `src/alignment/constitutional_ai.py` | KL divergence arguments swapped — alignment signal silenced | Corrected to `F.kl_div(log_policy, ref_probs.detach().exp())` |
| C11 | `src/agent/plugin_sandbox.py` | Sandbox fail-open: exception triggered unsandboxed plugin execution | Fail-closed: exception returns `SandboxResult(success=False)` |

### High-Severity Remediations

| ID | File | Issue | Fix |
|----|------|-------|-----|
| H1 | `src/backends/circuit_breaker.py` | Dead code after `return True` in `allow_request()` | Removed 9 unreachable lines |
| H2 | `src/backends/circuit_breaker.py` | `reset()` mutated shared state without lock | Added `with self._lock:` |
| H3 | `src/backends/async_rate_limiter.py` | `reset()` / `get_remaining()` bypassed asyncio lock | Made both methods `async` with `async with self._lock:` |
| H4 | `src/alignment/rlvr.py` | NaN risk from zero-std reward normalization | `std.clamp(min=1e-8)` guard |
| H5 | `src/backends/http_backend.py` | SSRF TOCTOU — URL validated after `Request()` construction | Moved `_validate_backend_url()` before object construction |
| H6 | `src/alignment/rlvr.py` + `grpo.py` | Negative KL possible with mismatched distributions | `.clamp(min=0.0)` on all KL penalty terms |
| H7 | `src/alignment/dapo.py` | Asymmetric PPO clip applied uniformly — broke DAPO invariant | `torch.where(advantages >= 0, ...)` for correct asymmetric clipping |
| H8 | `.github/workflows/ci.yml` | `continue-on-error: true` on Bandit/pip-audit made security gates cosmetic | Removed; CI now fails on security findings |

### CI/CD Hardening

- **`ruff-autofix.yml`**: Auto-fix workflow excluded from `main` branch (was force-pushing directly); Ruff version pinned to `0.9.0`
- **`pyproject.toml`**: Upper-bounded `torch`, `flash-attn`, `deepspeed` to prevent silent major-version breakage
- **`configs/config.yaml`**: MCP audit log path set (was `null`)
- **`.env.example`**: Auth and rate limiting enabled by default

---

## DAIES Scaling Plan (1.3B → 32B+)

Aurelius follows the **DAIES** (Doubling AI Efficiency Simultaneously) scaling philosophy: doubling parameters while optimizing memory, compute, and data efficiency at each step.

| Phase | Params | Strategy | Memory (M1 Pro 32GB) | Status |
|-------|--------|----------|---------------------|--------|
| **v1** | 1.395B | Muon+AdamW, grad_ckpt, bs=4 | ~12.8GB | Shipped |
| **v2** | 2.7B | Muon+AdamW, grad_ckpt, bs=1 | ~17.3GB | Recommended |
| **v3** | 3.0B | 8-bit optim, MLX, grad_ckpt, bs=1 | ~20.5GB | Tight fit |
| **v4** | 5-6B MoE | Sparse MoE, expert offloading, ~2B active | ~18-20GB | Complex |
| **v5** | 7-14B | bf16 or 4-bit quant | 14-28GB | Inference only |
| **v6** | 32B | MoE with expert parallelism, distributed | Cluster | Future |

Key principles:
- **Every layer is handwritten** — no black-box dependencies
- **Memory-budgeted design** — each phase fits within 32GB Apple Silicon or single H100
- **MoE upcycle** — dense checkpoints seed sparse experts
- **Quantization-first inference** — Q4_K_M GGUF for local serving at 25-35 tok/s

---

## Module Directory Structure

```
Aurelius/
+-- src/                          # Python backend (model, alignment, inference, CLI, serving, eval)
|   +-- model/                    # Transformer core (GQA, RoPE, SwiGLU, MoE, SSMs)
|   +-- training/                 # Muon, ZClip, curriculum, RLHF trainers
|   +-- alignment/                # DPO, GRPO, SimPO, ORPO, KTO, SPIN, constitutional AI
|   +-- inference/                # Speculative decoding, batching, paged KV cache
|   +-- persona/                  # 13 personas, 7 facets, routing, prompt composition
|   +-- agent/                    # ReAct loop, tool parser, planner, memory writer
|   +-- chat/                     # ChatML, Llama-3 templates, conversation management
|   +-- cli/                      # CLI entry points (aurelius, aurelius-cli, aurelius-shell)
|   +-- serving/                  # OpenAI-compatible API, streaming, metrics
|   +-- safety/                   # Jailbreak detector, output filter, PII scanner
|   +-- security/                 # Adversarial defense, backdoor scan, MITRE ATT&CK
|   +-- interpretability/         # Activation patching, SAEs, probing, circuit discovery
|   +-- eval/                     # Benchmarks, scorers, calibration
|   +-- data/                     # Data processing, tokenization, curriculum
|   +-- longcontext/              # KV quantization, StreamingLLM, attention sinks
|   +-- retrieval/                # BM25, hybrid search, dense retriever
|   +-- reasoning/                # MCTS, chain-of-thought, structured reasoning
|   +-- memory/                   # Semantic memory, episodic memory, recall
|   +-- tools/                    # Tool definitions, schemas, execution
|   +-- workflow/                 # Workflow engine, DAG execution
|   +-- multiagent/               # Multi-agent coordination, delegation
|   +-- multimodal/               # Vision, audio, multimodal integration
|   +-- quantization/             # AWQ, GPTQ, SmoothQuant, NF4, FP8
|   +-- runtime/                  # Hot reload, compile manager, compute scheduler
|   +-- computer_use/             # Computer use agent (UI navigation)
|   +-- mcp/                      # MCP protocol integration
|   +-- trading/                  # Trading agent capabilities
|   +-- monitoring/               # System monitoring, profiling
|   +-- evaluation/               # Evaluation harness, benchmarks
|   +-- protocol/                 # Protocol definitions, serialization
|   +-- profiling/                # Performance profiling tools
|   +-- optimizers/               # Custom optimizers (Muon, SOAP, etc.)
|   +-- simulation/               # Simulation environments
|   +-- compression/              # Model compression utilities
|   +-- federation/               # Federated learning
|   +-- backends/                 # Backend abstractions
|   +-- deployment/               # Deployment utilities
|   +-- ui/                       # Server-side UI utilities
+-- middle/                       # Node.js BFF (TypeScript)
|   +-- src/
|       +-- routes/               # API route modules (health, agents, activity, eval, retrieval, etc.)
|       +-- middleware/            # Auth, CORS, rate limiting, logging
|       +-- ws/                   # WebSocket hub
|       +-- index.ts              # Express server entry
|       +-- server.ts             # Server configuration
|       +-- engine.ts             # Rust engine bindings
|       +-- config.ts             # Configuration management
|       +-- cache.ts              # Caching layer
|       +-- provider_router.ts    # LLM provider routing
+-- frontend/                     # React 19 + Vite + TypeScript
|   +-- src/
|       +-- pages/                # Dashboard, Chat, Analytics, workflows, and admin surfaces
|       +-- components/           # Shared UI components
|       +-- hooks/                # Custom hooks and data helpers
|       +-- lib/                  # Utilities, API client, stores
+-- crates/                       # Rust workspace (11 crates + 2 tools)
|   +-- data-engine/              # Core Node.js NAPI data store
|   +-- api-gateway/              # Standalone reverse proxy binary
|   +-- token-counter/            # Approximate token counting
|   +-- session-manager/          # User session management with TTL
|   +-- search-index/             # BM25 full-text search
|   +-- vector-similarity/        # Cosine/dot/Euclidean similarity
|   +-- redis-client/             # Async Redis client
|   +-- text-processor/           # Text chunking, stats, keyword extraction
|   +-- prompt-templates/         # ChatML/Llama-3 template rendering
|   +-- json-validator/           # JSON schema validation
|   +-- uuid-gen/                 # UUID v4/v7 generation
|   +-- usage-store/              # PyO3 usage/cost tracking
+-- configs/                      # Training configs (1.4B, 2.7B, 3B, MoE-5B)
+-- scripts/                      # Benchmark, bootstrap, export, profile
+-- tests/                        # Broad test coverage across the stack
+-- deployment/                   # Docker, Docker Compose, Helm charts
+-- docs/                         # Documentation
+-- tools/                        # Rust CLI tools (data-cli, jsonl-merge)
+-- data/                         # Data artifacts
+-- training_data/                # Training datasets
+-- checkpoints/                  # Model checkpoints
+-- alembic/                      # Database migrations
+-- plugins/                      # Plugin system
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture, data flow, agent design, scaling philosophy, security model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |
| [CONFIDENTIAL.md](CONFIDENTIAL.md) | Confidential materials policy |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [FEATURE_AUDIT.md](FEATURE_AUDIT.md) | Feature audit and tracking |
| [EULA.md](EULA.md) | End User License Agreement |
| [LICENSE](LICENSE) | Aurelius Open License |
| [model_card.md](docs/model_card.md) | Model architecture card |
| [dataset_card.md](docs/dataset_card.md) | Dataset documentation |
| [eval_card.md](docs/eval_card.md) | Evaluation methodology |
| [threat_model.md](docs/threat_model.md) | Security threat model |
| [plans/](docs/plans/) | Design docs, handoff notes, implementation plans |
| [executive/](docs/executive/) | Executive overview (PDF) |

---

## Testing

```bash
# Python backend
pytest -q

# Frontend (Vitest)
cd frontend && npm test

# Rust
cd crates/data-engine && cargo test
```

## License

Aurelius is released under the [Aurelius Open License](LICENSE) — free to use, modify, and distribute for any purpose. The Aurelius name and logo are reserved trademarks.

---

**GitHub:** [https://github.com/S3nna13/Aurelius](https://github.com/S3nna13/Aurelius)
*Built with pure PyTorch. No compromises.*
