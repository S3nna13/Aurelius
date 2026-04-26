# Aurelius

> **A 1.395B-parameter decoder-only transformer** built entirely in pure PyTorch. No HuggingFace Transformers, no einops, no framework wrappers at runtime. Every algorithm is written from scratch. Talk to it like ChatGPT or Claude, extend it like a research codebase.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4+-06b6d4.svg)](https://tailwindcss.com/)
[![License: Aurelius Open License](https://img.shields.io/badge/License-Aurelius%20Open%20License-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-32%2C400%2B%20passing-brightgreen.svg)]()
[![Security](https://img.shields.io/badge/security-0%20High%20findings-success.svg)]()

---

## 🚀 What is Aurelius?

Aurelius is a **full-stack AI assistant platform** — from a 1.395B-parameter transformer to a production-ready API server, to a modern React web dashboard. It is designed for researchers who want full control over every layer of the stack, and for builders who want a deployable conversational AI with memory, agents, tools, and safety guardrails.

### Highlights

- **Pure PyTorch** — No runtime dependencies on HF Transformers, flash-attn, bitsandbytes, PEFT, TRL, or DeepSpeed
- **Family Architecture** — `FamilyManifest` + `ModelVariant` + factory pattern lets base / chat / coding / long-context / retrieval / agent / safety variants coexist under a stable contract
- **Modern Web UI** — React 19 + Vite + TypeScript + Tailwind CSS with PWA support, dark/light themes, and real-time streaming
- **Production-Grade Serving** — OpenAI-compatible HTTP API, WebSocket chat streaming, SSE events, session routing, and semantic memory
- **157+ Implementation Cycles** — Continuously expanded with 1,720+ Python modules across model, training, alignment, inference, eval, data, security, agents, and more

> **🔒 Training materials, datasets, and proprietary resources are confidential and internal-only.** See [`CONFIDENTIAL.md`](CONFIDENTIAL.md) for details. The architecture, inference engine, API, and web UI remain open under the Aurelius Open License.

---

## 📦 What's Inside

| Layer | Stack | Highlights |
|---|---|---|
| **Model** | Pure PyTorch | 1.395B decoder-only transformer, GQA, RoPE/YaRN, SwiGLU, RMSNorm, MoE, SSMs (Mamba, RWKV, Griffin), diffusion LM head, 150+ architecture modules |
| **Training Framework** | Custom trainers | Muon + AdamW + ZClip, async RL (double-sided IS), Shampoo, SOAP, GaLore, ReLoRA, DoRA, gradient checkpointing, curriculum, distillation, RLHF, 200+ utilities |
| **Alignment** | From-scratch | DPO, GRPO, SimPO, ORPO, KTO, SPIN, RLOO, Nash-MD, constitutional AI, debate alignment, 150+ modules |
| **Inference** | Optimized | Speculative decoding (Eagle, Medusa), flash prefill, continuous batching, paged KV cache, structured output, MCTS reasoning, 200+ modules |
| **Security** | Hardened | Gradient inversion defense, GCG adversarial search, prompt injection detector, STRIP backdoor scan, MITRE ATT&CK classifier, YARA-like engine, 24 modules |
| **Agent** | Tool-capable | ReAct loop with timeouts, tool-call parser (XML + JSON), argument validation, budget-bounded termination, multi-step chaining |
| **Serving** | Full-stack | OpenAI-compatible API, WebSocket streaming, React dashboard, session memory, tool calling, 6 personas, safety guardrails |
| **Frontend** | React 19 + Vite | Real-time chat, agent management, memory browser, workflow designer, system health, settings, PWA, dark/light mode |

---

## 🖥️ Screenshots

*The Aurelius dashboard features a modern dark-mode UI with real-time metrics, interactive chat, agent management, and system health monitoring.*

**Key UI Pages:**
- **Dashboard** — Live metrics with line charts, bar charts, and donut charts
- **Chat** — Streaming responses with slash commands, suggestions, and history persistence
- **Agents** — Agent comparison, detail views, and performance analytics
- **Memory** — Layered memory browser with advanced filtering (importance, date range, access count)
- **Workflows** — Visual workflow designer with status tracking
- **System Health** — Health check dashboard with component status and API documentation
- **Settings** — Runtime config, plugin manager, notification preferences, import/restore

---

## 🏁 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
pip install -e ".[serve,dev]"
```

### 2. Start the Backend

```bash
# Full assistant stack (API + web UI)
aurelius serve

# Or directly:
python -m src.serving.aurelius_server --port 7870
```

### 3. Start the Frontend (Development)

```bash
cd frontend
npm install
npm run dev
```

The web UI will be available at `http://localhost:5173` (or the port Vite assigns).

### 4. Build for Production

```bash
cd frontend
npm run build
```

The production build is served statically by the backend from `frontend/dist/`.

---

## 💻 The `aurelius` CLI

```bash
aurelius                              # interactive chat (default)
aurelius chat                         # same as above
aurelius chat -s coding               # use the built-in coding persona
aurelius chat -s security             # use the security expert persona
aurelius chat --model-path <ckpt>     # load trained weights
aurelius serve                        # start API server + browser web UI
aurelius serve --port 8080            # custom API port (default: 7870)
aurelius --version                    # print version
aurelius --help                       # full help
```

### Chat Slash Commands

| Command | Description |
|---|---|
| `/help` | List all commands |
| `/reset` | Clear conversation history |
| `/history` | Show conversation so far |
| `/system <prompt>` | Swap the system prompt on the fly |
| `/save <id>` | Save conversation to `~/.aurelius/conversations/` |
| `/load <id>` | Restore a saved conversation |
| `/list` | List all saved conversations |
| `/model` | Show loaded model info |
| `/clear` | Clear the screen |
| `/quit` | Exit |

### Built-in Personas (`-s` flag)

| Name | Purpose |
|---|---|
| `default` | General helpful assistant |
| `coding` | Software engineer, gives code examples |
| `security` | Cybersecurity expert, authorized testing |
| `researcher` | Analytical, multi-perspective |
| `concise` | Short answers only |
| `creative` | Storyteller, expressive |

---

## 🏗️ Architecture

### Transformer Core

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

- **Positional encoding:** RoPE (θ = 500K) with YaRN context extension
- **Normalization:** RMSNorm
- **Activation:** SwiGLU
- **Attention:** Grouped Query Attention (GQA)
- **Forward signature:** `(loss, logits, present_key_values)` — plain tuple

### Full-Stack Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      React 19 Frontend                       │
│  (Vite + TypeScript + Tailwind + Framer Motion + Zustand)   │
│  Dashboard · Chat · Agents · Memory · Workflows · Settings  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / WebSocket / SSE
┌────────────────────────────▼────────────────────────────────┐
│              Aurelius Server (Python / FastAPI)              │
│  REST API · OpenAI-compatible /v1 · WebSocket chat · SSE    │
│  Auth (opt-in) · Session routing · Semantic memory · Tools  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              Aurelius Transformer (Pure PyTorch)             │
│  1.395B params · GQA · RoPE · YaRN · SwiGLU · RMSNorm      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Usage

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

### WebSocket Chat Streaming

```javascript
const ws = new WebSocket('ws://localhost:7870/ws');
ws.onopen = () => ws.send(JSON.stringify({ type: 'subscribe', channel: 'chat' }));
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.token); // stream tokens in real-time
};
```

---

## 🧪 Running Tests

### Backend (Python)

```bash
# Full suite (32,400+ tests, ~30 min on CPU)
pytest -q

# Focused module tests
pytest -q tests/alignment/test_grpo.py
pytest -q tests/security/test_gcg_attack.py
pytest -q tests/serving/test_api_server.py
```

### Frontend (TypeScript / Vitest)

```bash
cd frontend
npm test          # run once
npm run test:watch # watch mode
npm run test:coverage # with coverage
```

### Security Scan

```bash
bandit -r src/ -f json -o bandit-report.json
```

---

## 📁 Repository Structure

```
Aurelius/
├── src/
│   ├── model/           # Transformer core, attention, SSMs, 150+ modules
│   ├── training/        # Training framework code (open architecture)
│   ├── alignment/       # DPO, GRPO, RLHF, constitutional AI, 150+ modules
│   ├── inference/       # Speculative decoding, batching, caching, 200+ modules
│   ├── eval/            # Benchmarks, scorers, calibration, 100+ modules
│   ├── data/            # Data processing framework code (open architecture)
│   ├── interpretability/# Activation patching, SAEs, probing, 20+ tools
│   ├── security/        # Adversarial defense, backdoor scan, MITRE ATT&CK, 24 modules
│   ├── agent/           # ReAct loop, tool parser, planner, memory writer
│   ├── chat/            # ChatML, Llama-3 templates, conversation management
│   ├── longcontext/     # KV quantization, attention sinks, StreamingLLM
│   ├── retrieval/       # BM25, hybrid search, dense retriever
│   ├── safety/          # Jailbreak detector, output filter, PII scanner
│   ├── serving/         # API server, web UI, streaming, session router
│   ├── cli/             # aurelius terminal command
│   └── ... (20+ surfaces)
├── frontend/            # React 19 + Vite + TypeScript + Tailwind
│   ├── src/
│   │   ├── pages/       # Dashboard, Chat, Agents, Memory, Settings, etc.
│   │   ├── components/  # Charts, Tables, Modals, Toasts, Sidebar
│   │   ├── hooks/       # useApi, useWebSocket, usePushNotifications
│   │   └── stores/      # Zustand state management
│   └── dist/            # Production build (tracked in git)
├── configs/             # ⚠️ Confidential — internal training configs
├── scripts/             # ⚠️ Confidential — internal data & training scripts
├── tests/               # 32,400+ tests across all surfaces
├── deployment/          # Dockerfile, docker-compose, Helm charts
├── docs/                # ⚠️ Confidential — internal documentation
└── CONFIDENTIAL.md      # Confidentiality policy
```

> **Note:** `src/training/` and `src/data/` contain the open **framework code** for training and data processing. The actual training runs, datasets, checkpoints, and proprietary configurations are confidential. See [`CONFIDENTIAL.md`](CONFIDENTIAL.md).

---

## 🎨 Frontend Features

- **Real-time Chat** — WebSocket streaming with typing indicators, slash commands, and quick actions
- **Interactive Dashboard** — Line charts, bar charts, donut charts with live data
- **Agent Management** — Agent comparison, detail views, lifecycle tracking
- **Memory Browser** — Layered memory with importance scoring, date filtering, and search
- **Workflow Designer** — Visual workflow status and task scheduling
- **System Health** — Component health checks, API docs, log viewer
- **Settings Panel** — Runtime config, plugin manager, notification preferences, import/restore
- **PWA Support** — Service worker, offline capability, installable app
- **Dark / Light Theme** — System-aware with manual toggle
- **Keyboard Shortcuts** — `?` for help, `/` for search, `G` + key for navigation
- **Global Search** — Command palette (`Cmd+K`) for quick navigation
- **Toast Notifications** — Animated stack with pause-on-hover
- **Responsive Design** — Mobile-friendly sidebar and layouts

---

## 🔐 Security

- **0 High / 0 Critical** findings (continuously monitored with `bandit`)
- Opt-in API authentication (`require_auth` defaults to `false`)
- API key and session token support
- SSRF-hardened HTTP backend
- Prompt injection detection and sanitization
- Gradient inversion defense and model fingerprinting
- Full CVE ledger: AUR-SEC-2026-0001 through AUR-SEC-2026-0019 (all closed)

---

## 📜 License

Aurelius is released under the **[Aurelius Open License](LICENSE)**.

> **Free to use, modify, and distribute** — for any purpose, personal or commercial. The underlying Aurelius architecture (agent orchestration model, memory hierarchy, design patterns) remains the intellectual property of the authors. You may not claim ownership of the architecture or file patents on it. The Aurelius name and logo are reserved trademarks.

See [`LICENSE`](LICENSE) and [`EULA.md`](EULA.md) for full terms.

### Confidential Materials

Training datasets, proprietary configurations, checkpoint files, experiment logs, and harvest journals are **confidential and internal-only**. They are not licensed for redistribution. See [`CONFIDENTIAL.md`](CONFIDENTIAL.md) for the full confidentiality policy.

---

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on:
- Code style (Black formatter, Ruff linter)
- Test requirements (every module ships with 10–16 tests)
- Security review process
- Cycle-based development workflow

---

## 📬 Repository

**GitHub:** [https://github.com/S3nna13/Aurelius](https://github.com/S3nna13/Aurelius)

**Issues & Discussions:** Open an issue for bugs, a discussion for questions.

---

*Built with ❤️ by the Aurelius team. Pure PyTorch. No compromises.*
