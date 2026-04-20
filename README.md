# Aurelius

Aurelius is a 1.395B-parameter conversational AI assistant and research platform built entirely in pure PyTorch — no HuggingFace Transformers, no einops, no framework wrappers at runtime. Every algorithm is written from scratch. Talk to it like ChatGPT or Claude, extend it like a research codebase.

---

## Quick start — talk to Aurelius right now

```bash
# Install
git clone https://github.com/S3nna13/Aurelius
cd Aurelius
pip install -e .

# Launch
aurelius
```

That opens an interactive terminal chat session. No checkpoint required to try the interface — connect a trained checkpoint with `--model-path` to get real responses.

---

## The `aurelius` command

```
aurelius                              interactive chat (default)
aurelius chat                         same as above
aurelius chat -s coding               use the built-in coding persona
aurelius chat -s security             use the security expert persona
aurelius chat --model-path <ckpt>     load trained weights
aurelius serve                        start API server + browser web UI
aurelius serve --port 8080            custom API port (default: 8080)
aurelius train --config configs/...   launch pretraining
aurelius eval checkpoints/<run>       run evaluation harness
aurelius --version                    print version
aurelius --help                       full help
```

### Chat slash commands

Once inside `aurelius`, type these at the prompt:

| Command | What it does |
|---|---|
| `/help` | list all commands |
| `/reset` | clear conversation history |
| `/history` | show conversation so far |
| `/system <prompt>` | swap the system prompt on the fly |
| `/save <id>` | save conversation to `~/.aurelius/conversations/` |
| `/load <id>` | restore a saved conversation |
| `/list` | list all saved conversations |
| `/model` | show loaded model info |
| `/clear` | clear the screen |
| `/quit` | exit |

### Built-in personas (`-s` flag)

| Name | Purpose |
|---|---|
| `default` | general helpful assistant |
| `coding` | software engineer, gives code examples |
| `security` | cybersecurity expert, authorized testing |
| `researcher` | analytical, multi-perspective |
| `concise` | short answers only |
| `creative` | storyteller, expressive |

---

## Architecture

**1.395B decoder-only transformer**

| Hyperparameter | Value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 24 |
| `n_heads` | 16 |
| `n_kv_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `d_ff` | 5632 (SwiGLU) |
| `vocab_size` | 128 000 |
| `max_seq_len` | 8 192 |

Positional encoding: RoPE (θ = 500K) with YaRN context extension. Normalization: RMSNorm. Activation: SwiGLU. Attention: Grouped Query Attention (GQA).

Model forward signature: `(loss, logits, present_key_values)` — plain tuple.

---

## What is in this repo

| Directory | Contents |
|---|---|
| `src/cli/` | `aurelius` terminal command — interactive REPL, subcommands (chat/serve/train/eval), ANSI colors, slash commands, persona routing |
| `src/model/` | Transformer core, GQA, RoPE/YaRN, MoE (sparse + balanced + upcycling), MoD, dynamic sparse attention, parallel residual blocks, Fourier positional encodings, sliding window attention, linear/sparse/ring attention, SSMs (Mamba, S4, RWKV, Griffin, Jamba, GLA, RetNet, Hyena/H3), diffusion LM head, nGPT, Titans, xLSTM, NSA, TTT-Linear, minGRU, Gated Delta Net, and 150+ architecture modules |
| `src/training/` | Muon + AdamW + ZClip trainer, async RL trainer (double-sided IS + staleness filtering), Shampoo, SOAP, SAM, Lion, NesterovAdan, GaLore, ReLoRA, DoRA, LoRA+, active learning, gradient checkpointing, scheduled sampling, spectral filtering, EWC, curriculum, distillation, RLHF, self-improvement loop, MTP, and 200+ training utilities |
| `src/optimizers/` | Adafactor, LAMB, Lookahead, RAdam, CAME, ADOPT, FAdam, Fira, Signum |
| `src/alignment/` | DPO, GRPO, Dr. GRPO, SimPO, ORPO, KTO, IPO, SPIN, RLOO, Nash-MD, DAPO, WARP, BOND, STILL, SALMON, DITTO, RLHF (PPO), RLCD, PRIME, ODIN, online DPO, double-sided IS loss, constitutional AI (v1–v3), debate alignment, process supervision, reward modeling, red teaming, scheduled sampling, and 150+ alignment modules |
| `src/inference/` | Speculative decoding (standard, tree, Eagle/Eagle-2, Medusa, cascade), Chain of Draft, entropix adaptive sampling, flash/chunked prefill, continuous batching, paged KV cache, KV quantization, prompt compression, lookahead/Jacobi decoding, RAG (FiD, fusion, attributed), structured output, watermarking, MCTS reasoning, test-time compute scaling, arithmetic coding, and 200+ inference modules |
| `src/eval/` | LM harness, BERTScore, LLM-as-judge, causal tracing, ROME weight editing, calibration suite, OOD detection, conformal prediction, probing classifiers, logit lens, tuned lens, membership inference, Vendi score, MT-Bench, faithfulness metrics, model-written evals, and 100+ eval modules |
| `src/data/` | BPE + byte tokenizers, Magpie, FIM, sequence packing, data mixing, curriculum sampling, difficulty scoring, quality filtering, QuRating scorer, synthetic instruction generation, augmentation, deduplication, and 80+ data modules |
| `src/interpretability/` | Activation patching, circuit discovery, LEACE concept erasure, polysemanticity/superposition detector, function vectors, distributed alignment search (DAS), JumpReLU sparse autoencoder, Patchscopes, logit lens, probing, neuron analysis, representation engineering, and 20+ interpretability tools |
| `src/security/` | 24 security modules: gradient inversion, model extraction, STRIP backdoor detector, GCG adversarial suffix search, canary memorization auditor, prompt injection detector, randomized smoothing, Rényi DP privacy accountant, PII/toxicity output scanner, adversarial text augmentation, network intrusion detector, semantic similarity defense, federated aggregator, per-sample gradient clipping, model fingerprinting, robustness evaluator, red-team dataset generator, additive secret sharing, model-stealing defense, threat-intel correlator, MITRE ATT&CK taxonomy classifier (81 techniques, all 12 enterprise tactics, kill-chain ordering), IOC extractor (IPs / domains / URLs / email / hashes / CVE / paths / registry / Bitcoin, with defang/refang), YARA-like rule engine (text + hex + regex strings, AND/OR/NOT, count, filesize), PE file analyzer (DOS + NT + sections + entropy for packer detection), log anomaly detector (volume spike, rare-token, high-entropy URL, off-hours, auth-cluster) |
| `src/agent/` | Tool-call parser (XML Anthropic-style + JSON OpenAI-style, injection-hardened state-machine scan), ReAct agent loop (plan → act → observe → reflect) with per-tool wall-clock timeouts, argument validation via `inspect.signature`, and budget-bounded termination |
| `src/chat/` | ChatML template (injection-hardened encode/decode, role-break rejection), Llama-3 template (`<|begin_of_text|>` / `<|start_header_id|>` format, ipython + tool roles), pluggable tokenizer-agnostic token-id path |
| `src/longcontext/` | INT8 symmetric per-head KV cache quantizer with streaming append, StreamingLLM attention-sinks windowing (sink + rolling buffer with shifted RoPE positions per Xiao 2023) |
| `src/retrieval/` | BM25 (Robertson-Sparck Jones IDF, postings-walk scorer, pure stdlib), hybrid retriever (BM25 + dense cosine with Reciprocal Rank Fusion k=60 or min-max weighted fusion) |
| `src/safety/` | Jailbreak detector (keyword + role-confusion + prompt-injection + repetition-burst, NFKC-normalized, weighted signal fusion), prompt-injection scanner for indirect injection via tool outputs / retrieved docs (HTML/script, homoglyph, zero-width char, embedded-instruction detection) |
| `src/serving/` | `aurelius` CLI REPL, OpenAI-compatible HTTP API server, browser web UI, token-by-token SSE streaming, tool calling + multi-step tool chaining, 6 curated system prompt personas, conversation history (JSON persistence), semantic cross-session memory, session router (consistent hash ring + LRU), response formatter, runtime safety guardrails, ChatML session manager |
| `src/eval/` additions | Needle-in-a-Haystack benchmark (Kamradt protocol, model-agnostic generate_fn), RULER (Hsieh 2024) with multi-key NIAH, multi-value NIAH, variable tracking, common-words extraction, aggregation tasks |
| `configs/` | Training, tokenizer, merge (SLERP/TIES), curriculum, and Ollama configs |
| `scripts/` | Data prep, training, SFT, DPO, model merging, GGUF conversion, local serving |

---

## Installation

```bash
git clone https://github.com/S3nna13/Aurelius
cd Aurelius
pip install -e .
```

For development and the full test suite:

```bash
pip install -e .[dev]
```

**Requirements:** Python 3.12+, PyTorch ≥ 2.4. No runtime dependencies beyond PyTorch.

---

## Running tests

Full suite (15 000+ tests, ~15 min on CPU):

```bash
pytest -q
```

Focused module tests:

```bash
pytest -q tests/alignment/test_grpo.py
pytest -q tests/security/test_gcg_attack.py
pytest -q tests/serving/test_api_server.py
```

---

## Training workflows

### 1. Train a tokenizer

```bash
bash scripts/train_tokenizer.sh
```

### 2. Prepare data

```bash
bash scripts/prepare_data.sh
```

### 3. Pretrain

```bash
aurelius train --config configs/train_1b.yaml
# Resume:
aurelius train --config configs/train_1b.yaml --resume checkpoints/<dir>
```

### 4. Alignment (SFT → DPO → RLHF)

```bash
python -m src.alignment.sft --help
python -m src.alignment.dpo --help
bash scripts/run_sft.sh
bash scripts/run_dpo.sh
```

### 5. Evaluate

```bash
aurelius eval checkpoints/<dir>
```

### 6. Serve

```bash
# Full assistant stack (API + browser UI)
aurelius serve

# Just the API (OpenAI-compatible)
python -m src.serving.api_server --port 8080

# Just the browser UI
python -m src.serving.web_ui --port 7860

# Via Ollama (GGUF)
bash scripts/convert_to_gguf.sh <checkpoint> [output-dir]
bash scripts/serve_local.sh --model-path models/gguf/aurelius-1.3b-q4_k_m.gguf
```

---

## Using the OpenAI-compatible API

Once `aurelius serve` is running, any OpenAI client works:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="aurelius",
    messages=[{"role": "user", "content": "Hello, who are you?"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aurelius","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Package imports

Both styles work:

```python
from src.model.transformer import AureliusTransformer
from aurelius.model.transformer import AureliusTransformer
```

---

## Key design principles

- **Pure PyTorch.** No HuggingFace Transformers, einops, flash-attn, bitsandbytes, peft, trl, accelerate, or deepspeed at runtime.
- **Additive development.** Each cycle adds new modules; existing files are not modified.
- **Full test coverage.** Every module ships with 10–16 tests covering shape/dtype, gradient flow, determinism, edge cases, and numerical stability.
- **Production assistant.** The `aurelius` CLI, API server, web UI, tool calling, streaming, memory, and guardrails make this a deployable conversational AI — not just a research codebase.

---

## Config files

| File | Purpose |
|---|---|
| `configs/train_1b.yaml` | Default pretraining config |
| `configs/curriculum.yaml` | Curriculum learning settings |
| `configs/merge_slerp.yaml` | Model merging (SLERP/TIES) |
| `configs/ollama.Modelfile` | Ollama model definition |
| `configs/tokenizer_config.json` | Tokenizer metadata |

---

## Current status

- **120 implementation cycles** completed
- **19 100+ tests** passing (full suite ~15-28 min on CPU)
- **1 000+ Python source files** across model, training, alignment, inference, eval, data, interpretability, optimizer, security, serving, CLI, and the newer **agent / chat / longcontext / retrieval / safety** surface dirs
- Recent cycles:
  - **cycle-117** — Toolformer-style self-supervised tool-learning data-gen, tau-bench scorer, dynamic NTK-aware context-window extension, grammar-constrained structured-output decoder, step-level Process Reward Model
  - **cycle-118** — hallucination guard (claim-vs-evidence validator), skill scanner (agent-skill supply-chain), canary-token guard (runtime prompt-injection defense), Mythos 6-dim coding rubric + anti-reward-hack system prompt, adversarial code battle (red-vs-blue self-play → DPO pairs), CWE synthetic-bug recipe catalog
  - **cycle-119** — reward-hack detector (12 patterns from Mythos System Card p.65), PyRIT Crescendo multi-turn jailbreak probe, Raptor-style parallel dispatch-task abstraction, 3-state circuit breaker with exponential backoff, Mythos 40-dimension behavioral-audit taxonomy, threat-intel analyst persona with structured CVE/MITRE/actor/IOC output contracts
  - **cycle-120** — g0-style YAML rule engine (12 agentic domains, 6 check types), Mythos 15-dim constitution scorer (Spirit/Level-1/Level-2), SHADE-Arena long-horizon main+side-task eval, RED/BLUE/PURPLE security personas, tool-sandbox denylist (35 rules / 7 categories), s0cli-style vibe code reviewer with skepticism-rule filter
- `aurelius` works as a terminal command after `pip install -e .`
- Frontier-tier surfaces wired in cycles 100–103:
  - **agent** — 10 modules: tool-call parsers, ReAct loop, dispatcher, ToT planner, error recovery, repo-context packer, unified-diff generator, shell planner, code-execution sandbox, task decomposer (DAG + topo sort + parallelizable-group extraction)
  - **chat** — 10 modules: ChatML/Llama-3/Harmony, tool-message formatter, conversation memory, instruction-tuning data, role-aware masks, FIM, multi-turn state, 5-strategy message-truncation policy
  - **longcontext** — 10 modules: INT8/INT4 KV, attention sinks, ring/Infini attention, context compaction, YaRN utilities, chunked prefill, paged KV cache, prefix cache (block-trie + LRU + refcount pinning)
  - **retrieval** — 10 modules: BM25, hybrid RRF, fusion suite, cross-encoder + MMR/Jaccard rerankers, dense embedder, code-aware tokenizer, instruction-prefix embedder, corpus indexer, hard-negative miner (bm25-hard / embedding-hard / in-batch)
  - **safety** — 10 modules: jailbreak, prompt-injection, harm-taxonomy, PII, output filter, prompt-integrity, refusal, constitutional, malicious-code, policy engine (declarative rule pipeline over all sub-filters)
  - **eval** — 9 modules: NIAH, RULER, HumanEval, MBPP, SWE-bench-lite, IFEval, MT-Bench, AlpacaEval, Arena-Hard (Bradley-Terry + bootstrap CIs)
  - **inference** additions — Orca-style continuous-batching scheduler, JSON-mode constrained decoder (structural JSON validity via stack parser)
  - **training** additions — FSDP-lite sharded-DDP wrapper (single-process simulation)
  - **serving** additions — OpenAI Chat Completions request/response validator + API_SHAPE_REGISTRY
  - **optimizers** additions — Schedule-Free AdamW (Defazio 2024, no LR schedule required)
  - **data** additions — instruction-dataset packer with FFD bin-packing + block-diagonal attention mask + response-only loss mask
  - **eval** additions — GPQA (graduate-level MCQ scorer)

---

## License

Aurelius is released under the [MIT License](LICENSE). You may use, modify, and distribute this code freely with attribution.
