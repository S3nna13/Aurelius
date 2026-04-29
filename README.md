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

Aurelius is organized into four independent layers, each owning a distinct responsibility:

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| **Rust Engine** | `crates/`, `tools/` | Rust 2024 | High-performance data engine, search, tokenization, vector similarity, caching, auth sessions |
| **Python Backend** | `src/`, `configs/` | Python 3.12 | Model training, inference, API server, CLI |
| **Node.js BFF** | `middle/` | TypeScript | API gateway for frontend, auth, file serving, WebSocket, SSE, cron scheduler |
| **Frontend** | `frontend/` | TypeScript / React 19 | Mission Control dashboard, agent management, chat, analytics |

### Data Flow

```
Browser / Frontend
    |  HTTP / WebSocket
    v
Node.js BFF (middle/ — port 3001)
    |  NAPI-rs FFI (sync)        |  HTTP proxy
    v                              v
Rust Engine (crates/data-engine/)  Python API Server (src/serving/ — port 8080)
    |                                    |
    | in-memory state (dashmap+RwLock)   | model inference, persona routing
    v                                    v
JSON export / Redis / SQLite            Trained checkpoints (safetensors)
```

The frontend **never** talks directly to Python. All API calls route through the Node.js BFF, which handles auth, rate limiting, WebSocket multiplexing, SSE, file upload, and cron scheduling. The Python server only exposes model inference endpoints (`/v1/chat/completions`, `/v1/models`).

---

## Key Features & Capabilities

### Model (Pure PyTorch)
- **1.395B decoder-only transformer** — GQA, RoPE/YaRN, SwiGLU, RMSNorm
- **150+ architecture modules** — MoE, SSMs (Mamba, RWKV, Griffin), diffusion LM head
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

### Frontend (Mission Control)
- **23 pages** — Dashboard, Chat, Analytics, Notifications, Skills, Workflows, Memory, Tasks, Agents, Users, Logs, Health, API Docs, Settings, Data Explorer
- **Real-time streaming** — WebSocket chat, SSE events, live metrics
- **49 components, 23 hooks** — React 19 + Vite + TypeScript + Tailwind CSS

### Serving & Deployment
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **Speculative decoding** — Eagle / Medusa heads for 2-3x throughput
- **Continuous batching** and paged KV cache
- **Docker Compose** — single command for full stack
- **Helm charts** for Kubernetes deployment

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
# Clone and install Python package
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
pip install -e ".[dev]"

# Build Rust data engine
cd crates/data-engine
npm install
npm run build
cd ../..

# Install Node.js BFF
cd middle
npm install
cd ..

# Install frontend
cd frontend
npm install
cd ..
```

### Running the CLI

```bash
aurelius                      # Interactive chat (default)
aurelius chat                 # Same as above
aurelius chat -p aurelius-coding   # Use coding persona
aurelius chat --model-path <ckpt>  # Load trained weights
aurelius serve                 # Start API server + web UI
aurelius serve --port 8080     # Custom API port
aurelius --version             # Print version
```

### Running the API

```bash
# Start Python inference server
aurelius-api

# Start Node.js BFF (in another terminal)
cd middle && npm run dev

# Start frontend (in another terminal)
cd frontend && npm run dev

# Or everything at once
docker compose up
docker compose --profile inference up   # with LLM inference
```

### OpenAI-Compatible Usage

```python
import openai
client = openai.OpenAI(base_url="http://localhost:7870/v1", api_key="none")
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
+-- src/                          # Python backend (~1,720 modules)
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
|       +-- routes/               # 17 route modules (health, agents, activity, etc.)
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
|       +-- pages/                # 23 pages (Dashboard, Chat, Analytics, etc.)
|       +-- components/           # 49 reusable components
|       +-- hooks/                # 23 custom hooks
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
+-- tests/                        # 32,400+ tests across all surfaces
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
# Python backend (32,400+ tests)
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
