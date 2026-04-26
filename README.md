# Aurelius

> **A 1.395B-parameter decoder-only transformer** (scaling to 2.7B+) built entirely in pure PyTorch. No HuggingFace Transformers, no einops, no framework wrappers at runtime. Every algorithm is written from scratch. Talk to it like ChatGPT or Claude, extend it like a research codebase.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4+-06b6d4.svg)](https://tailwindcss.com/)
[![License: Aurelius Open License](https://img.shields.io/badge/License-Aurelius%20Open%20License-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-32%2C400%2B%20passing-brightgreen.svg)]()
[![Security](https://img.shields.io/badge/security-0%20High%20findings-success.svg)]()

---

## What is Aurelius?

Aurelius is a **full-stack AI assistant platform** — from a 1.395B-parameter transformer (scaling to 2.7B/3B/MoE-6B) to a Rust-backed API gateway, to a modern React web dashboard. It is designed for researchers who want full control over every layer of the stack, and for builders who want a deployable conversational AI with memory, agents, tools, and safety guardrails.

### Highlights

- **Pure PyTorch** — No runtime dependencies on HF Transformers, flash-attn, bitsandbytes, PEFT, TRL, or DeepSpeed
- **Unified Persona System** — 13 personas across 5 domains with composable facets (security, threat intel, constitution, harm filter, agent mode)
- **Scaling Roadmap** — 1.4B now, 2.7B dense next, 3B with 8-bit optimizers, 5-6B MoE with expert offloading
- **Family Architecture** — `FamilyManifest` + `ModelVariant` + factory pattern lets base/chat/coding/long-context/agent/safety variants coexist
- **Modern Web UI** — React 19 + Vite + TypeScript + Tailwind CSS with PWA support, dark/light themes, real-time streaming
- **Production-Grade Serving** — OpenAI-compatible HTTP API, WebSocket chat streaming, SSE events, session routing, semantic memory
- **157+ Implementation Cycles** — 1,720+ Python modules across model, training, alignment, inference, eval, data, security, agents, persona

> **Training materials, datasets, and proprietary resources are confidential and internal-only.** See [`CONFIDENTIAL.md`](CONFIDENTIAL.md) for details.

---

## Scaling Roadmap (M1 Pro 32GB)

| Tier | Params | Strategy | Memory | Feasibility |
|------|--------|----------|--------|-------------|
| **Current** | 1.395B | Muon+AdamW, grad_ckpt, bs=4 | ~12.8GB | Shipped |
| **Phase 1** | 2.7B | Muon+AdamW, grad_ckpt, bs=1 | ~17.3GB | Recommended |
| **Phase 2** | 3.0B | 8-bit optim, MLX, grad_ckpt, bs=1 | ~20.5GB | Tight |
| **Phase 3** | 5-6B MoE | Sparse MoE, expert swap, ~2B active | ~18-20GB | Complex |
| **Inference** | 7-14B | bf16 or 4-bit quant | 14-28GB | Inference only |

See [`aurelius_loop_v9.md`](Desktop/aurelius_loop_v9.md) for the full scaling plan with architecture configs, memory budgets, and implementation steps.

---

## Unified Persona System

Aurelius consolidates **7 fragmented persona systems** into one `UnifiedPersona` architecture with composable **facets**:

| Persona | Domain | Temperature | Key Facets |
|---------|--------|-------------|------------|
| `aurelius-general` | GENERAL | 0.7 | constitution, harm_filter |
| `aurelius-coding` | CODING | 0.3 | agent_mode(code), constitution, harm_filter |
| `aurelius-teacher` | GENERAL | 0.8 | personality(teacher), constitution |
| `aurelius-analyst` | GENERAL | 0.2 | personality(analyst), constitution |
| `aurelius-creative` | GENERAL | 1.0 | personality(creative), constitution |
| `aurelius-redteam` | SECURITY | 0.2 | security(offensive), constitution, harm_filter |
| `aurelius-blueteam` | SECURITY | 0.2 | security(defensive), constitution |
| `aurelius-purpleteam`| SECURITY | 0.2 | security(purple), constitution, harm_filter |
| `aurelius-threatintel`| THREAT_INTEL | 0.2 | threat_intel, constitution, harm_filter |
| `aurelius-code` | CODING | 0.3 | agent_mode(code) |
| `aurelius-architect` | AGENT | 0.5 | agent_mode(architect) |
| `aurelius-ask` | GENERAL | 0.7 | agent_mode(ask), personality |
| `aurelius-debug` | CODING | 0.2 | agent_mode(debug), personality |

**Facets** are composable capability attachments that let any persona gain features from any other system:

- `security` — scope enforcement, guardrails, output contracts, workflow stages
- `threat_intel` — CVE/MITRE/actor/IOC classification, response validation
- `agent_mode` — tool gating, response style
- `constitution` — 15-dimension scoring for RLHF/alignment
- `harm_filter` — 9-category harm taxonomy with per-persona thresholds
- `personality` — keyword-triggered personality traits
- `dialogue` — state machine (GREETING → TASK_EXECUTION → CLOSING)

```python
from src.persona import UnifiedPersonaRegistry, PersonaRouter, PromptComposer
from src.persona.builtins import ALL_BUILTINS

registry = UnifiedPersonaRegistry()
for persona in ALL_BUILTINS:
    registry.register(persona)

router = PersonaRouter(registry)
persona = router.route("What is CVE-2024-3094?")  # -> aurelius-threatintel

composer = PromptComposer()
messages = composer.build_messages(persona, "Explain CVE-2024-3094")
```

---

## What's Inside

| Layer | Stack | Highlights |
|---|---|---|
| **Model** | Pure PyTorch | 1.395B decoder-only transformer, GQA, RoPE/YaRN, SwiGLU, RMSNorm, MoE, SSMs (Mamba, RWKV, Griffin), diffusion LM head, 150+ architecture modules |
| **Persona** | Unified system | 13 personas, 7 composable facets, 9 security/threat-intel output contracts, unified routing, prompt composition |
| **Training Framework** | Custom trainers | Muon + AdamW + ZClip, async RL, curriculum, distillation, RLHF, 200+ utilities |
| **Alignment** | From-scratch | DPO, GRPO, SimPO, ORPO, KTO, SPIN, RLOO, Nash-MD, constitutional AI, 150+ modules |
| **Inference** | Optimized | Speculative decoding (Eagle, Medusa), flash prefill, continuous batching, paged KV cache, structured output, MCTS reasoning |
| **Security** | Hardened | Gradient inversion defense, GCG adversarial search, prompt injection detector, STRIP backdoor scan, 24 modules |
| **Agent** | Tool-capable | ReAct loop with timeouts, tool-call parser, argument validation, budget-bounded termination, multi-step chaining |
| **Serving** | Full-stack | OpenAI-compatible API, WebSocket streaming, React dashboard, session memory, tool calling, persona system |
| **Frontend** | React 19 + Vite | Real-time chat, agent management, memory browser, workflow designer, system health, settings, PWA |

---

## Transformer Core

| Hyperparameter | Value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 24 |
| `n_heads` | 16 |
| `n_kv_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `d_ff` | 5632 (SwiGLU) |
| `vocab_size` | 128,000 |
| `max_seq_len` | 8,192 |

- **Positional encoding:** RoPE (theta = 500K) with YaRN context extension
- **Normalization:** RMSNorm
- **Activation:** SwiGLU
- **Attention:** Grouped Query Attention (GQA)
- **Optimizer:** Muon (matrix params) + AdamW (embeddings/norms)
- **Gradient clipping:** ZClip (adaptive z-score anomaly detection)
- **Checkpointing:** Safetensors format with rolling window

---

## Quick Start

### 1. Prerequisites

```bash
node >= 22    # Frontend + BFF
npm >= 10     # Package management
rustc >= 1.81 # Rust data engine
python >= 3.12 # Optional: LLM inference
```

### 2. Build the Rust Data Engine

```bash
cd crates/data-engine
npm install    # installs @napi-rs/cli
npm run build  # compiles Rust -> native .node addon
```

### 3. Start the Gateway (Development)

```bash
cd server
npm install
npm run dev   # starts on http://localhost:7870
```

### 4. Start the Frontend (optional)

```bash
cd frontend
npm install
npm run dev   # starts on http://localhost:5173
```

### 5. Build for Production

```bash
cd frontend && npm run build  # outputs to frontend/dist/
cd server && npm run build    # compiles TypeScript
npm start                     # serves SPA + API on :7870
```

### 6. Docker (Full Stack)

```bash
docker compose up            # gateway only
docker compose --profile inference up  # gateway + Python LLM
```

---

## The `aurelius` CLI

```bash
aurelius                              # interactive chat (default)
aurelius chat                         # same as above
aurelius chat -p aurelius-coding      # use the coding persona
aurelius chat -p aurelius-redteam     # use the red team security persona
aurelius chat -p aurelius-threatintel  # use the threat intel persona
aurelius chat --model-path <ckpt>     # load trained weights
aurelius serve                        # start API server + browser web UI
aurelius serve --port 8080            # custom API port (default: 7870)
aurelius --version                    # print version
aurelius --help                       # full help
```

### Built-in Personas (`-p` flag)

| Persona ID | Domain | Purpose |
|---|---|---|
| `aurelius-general` | GENERAL | General helpful assistant |
| `aurelius-coding` | CODING | Software engineer, code-focused |
| `aurelius-teacher` | GENERAL | Patient educator, step-by-step |
| `aurelius-analyst` | GENERAL | Data & research analyst |
| `aurelius-creative` | GENERAL | Creative writing companion |
| `aurelius-redteam` | SECURITY | Authorized offensive security (lab scope only) |
| `aurelius-blueteam` | SECURITY | Defensive SOC / incident response |
| `aurelius-purpleteam` | SECURITY | Joint offense + defense, detection validation |
| `aurelius-threatintel` | THREAT_INTEL | CVE, MITRE ATT&CK, threat actors, IOCs |
| `aurelius-code` | CODING | Agent mode: write, edit, refactor |
| `aurelius-architect` | AGENT | Agent mode: design, plan, evaluate |
| `aurelius-ask` | GENERAL | Agent mode: answer, explain, document |
| `aurelius-debug` | CODING | Agent mode: trace, isolate, fix |

### Chat Slash Commands

| Command | Description |
|---|---|
| `/help` | List all commands |
| `/reset` | Clear conversation history |
| `/history` | Show conversation so far |
| `/system <prompt>` | Swap the system prompt |
| `/save <id>` | Save conversation |
| `/load <id>` | Restore a saved conversation |
| `/list` | List all saved conversations |
| `/model` | Show loaded model info |
| `/clear` | Clear the screen |
| `/quit` | Exit |

---

## Full-Stack Architecture

```
+--------------------------------------------------------------+
|                      React 19 Frontend                        |
|  (Vite + TypeScript + Tailwind + Zustand + Framer Motion)    |
|  Dashboard . Chat . Playground . Training . Models . Memory   |
|  Agents . Workflows . Skills . Notifications . Logs . Health  |
+------------------------------+-------------------------------+
                               | HTTP REST + WebSocket
+------------------------------v-------------------------------+
|                  Node.js Gateway (Express)                     |
|  Middleware: CORS . Auth . Rate-Limit . Logging . Metrics     |
|  Routes: 20+ REST endpoints . WebSocket Hub (/ws)             |
|  Static: SPA serving from frontend/dist/                      |
+---------------+------------------------------+------------------+
                | napi-rs FFI                  | HTTP proxy
+---------------v---------------+  +-----------v------------------+
|  Rust Data Engine (.node)      |  |  Python LLM (port 8080)     |
|  Agent state . Activity         |  |  Aurelius Transformer       |
|  Notifications . Memory         |  |  Persona System             |
|  Config . Logs . Models         |  |  13 Personas . 7 Facets     |
|  Skills . Workflows             |  |  UnifiedPersona Router       |
|  Training Runs . Prefs          |  |  PromptComposer             |
+---------------------------------+  +-----------------------------+
```

---

## API Usage

### OpenAI-Compatible Endpoint

```python
import openai

client = openai.OpenAI(base_url="http://localhost:7870/v1", api_key="none")

response = client.chat.completions.create(
    model="aurelius",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:7870/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aurelius","messages":[{"role":"user","content":"Hello"}]}'
```

### Key REST Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/modes` | GET | List agent modes |
| `/api/config` | GET / POST | Runtime configuration |
| `/api/memory/entries` | GET | Memory entries with layer/filter/search |
| `/api/logs` | GET | System logs (500-record ring buffer) |
| `/api/notifications/preferences` | GET / POST | Notification preferences |
| `/api/events` | SSE | Real-time system events |
| `/ws` | WebSocket | Bi-directional chat streaming |

---

## Running Tests

### Backend (Python)

```bash
pytest -q                              # Full suite (32,400+ tests)
pytest -q tests/alignment/test_grpo.py  # Focused module
pytest -q src/persona/               # Persona system tests
```

### Frontend (TypeScript / Vitest)

```bash
cd frontend
npm test              # run once
npm run test:watch    # watch mode
npm run test:coverage  # with coverage
```

---

## Repository Structure

```
Aurelius/
+-- src/
|   +-- persona/          # Unified persona system (13 personas, 7 facets, 9 contracts)
|   +-- model/            # Transformer core, attention, SSMs, 150+ modules
|   +-- training/         # Training framework: Muon, ZClip, curriculum, RLHF
|   +-- alignment/        # DPO, GRPO, SimPO, ORPO, KTO, constitutional AI
|   +-- inference/         # Speculative decoding, batching, caching, 200+ modules
|   +-- eval/             # Benchmarks, scorers, calibration, 100+ modules
|   +-- data/             # Data processing, tokenization, curriculum
|   +-- interpretability/  # Activation patching, SAEs, probing
|   +-- security/          # Adversarial defense, backdoor scan, MITRE ATT&CK
|   +-- agent/             # ReAct loop, tool parser, planner, memory writer
|   +-- chat/              # ChatML, Llama-3 templates, conversation management
|   +-- longcontext/       # KV quantization, attention sinks, StreamingLLM
|   +-- retrieval/         # BM25, hybrid search, dense retriever
|   +-- safety/            # Jailbreak detector, output filter, PII scanner
|   +-- serving/           # API server, web UI, streaming, session router
|   +-- cli/               # aurelius terminal command
|   +-- ... (20+ surfaces)
+-- server/               # Node.js API Gateway (Express + TypeScript)
+-- crates/               # Rust crates (data-engine, search-index, vector-similarity, etc.)
+-- frontend/              # React 19 + Vite + TypeScript + Tailwind
+-- configs/              # Training configs (1.4B, 2.7B, 3B, MoE-5B, curriculum)
+-- scripts/              # Benchmark, bootstrap, export, profile scripts
+-- tests/                 # 32,400+ tests across all surfaces
+-- deployment/            # Docker, docker-compose, Helm charts
+-- docs/                 # Documentation (confidential)
```

---

## Security

- **0 High / 0 Critical** findings (continuously monitored with `bandit`)
- Opt-in API authentication (`require_auth` defaults to `false`)
- API key and session token support
- SSRF-hardened HTTP backend
- Prompt injection detection and sanitization
- Gradient inversion defense and model fingerprinting
- Full CVE ledger: AUR-SEC-2026-0001 through AUR-SEC-2026-0019 (all closed)

---

## License

Aurelius is released under the **[Aurelius Open License](LICENSE)**.

> **Free to use, modify, and distribute** — for any purpose, personal or commercial. The underlying Aurelius architecture remains the intellectual property of the authors. The Aurelius name and logo are reserved trademarks.

See [`LICENSE`](LICENSE) and [`EULA.md`](EULA.md) for full terms. Training datasets, proprietary configurations, checkpoints, and experiment logs are **confidential and internal-only** — see [`CONFIDENTIAL.md`](CONFIDENTIAL.md).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on code style (Black formatter, Ruff linter), test requirements (every module ships with 10-16 tests), security review, and cycle-based development workflow.

---

**GitHub:** [https://github.com/S3nna13/Aurelius](https://github.com/S3nna13/Aurelius)

*Built with pure PyTorch. No compromises.*