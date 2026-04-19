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
| `src/security/` | Gradient inversion attack, model extraction (knockoff nets), STRIP backdoor detector, GCG adversarial suffix search, canary memorization auditor, prompt injection detector, randomized smoothing (certified robustness), Rényi DP privacy accountant, PII/toxicity output scanner, adversarial text augmentation, transformer network intrusion detector, semantic similarity defense, federated aggregator (FedAvg + noise), per-sample gradient clipping, model fingerprinting, robustness evaluator, red-team dataset generator, additive secret sharing — **18 security modules** |
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

- **104 implementation cycles** completed
- **16 900+ tests** passing (full suite ~16-21 min on CPU)
- **1 000+ Python source files** across model, training, alignment, inference, eval, data, interpretability, optimizer, security, serving, CLI, and the newer **agent / chat / longcontext / retrieval / safety** surface dirs
- `aurelius` works as a terminal command after `pip install -e .`
- Frontier-tier surfaces wired in cycles 100–103:
  - **agent** — tool-call parser (XML + JSON), ReAct loop, safe tool-registry dispatcher (rate-limit + timeout + budget), beam-search Tree-of-Thoughts planner, tool-error-recovery strategy (retry/backoff/fallback/escalate)
  - **chat** — ChatML, Llama-3, Harmony (OpenAI gpt-oss format) templates, tool-message formatter, persistent conversation memory (in-memory + atomic-JSON backends with BM25 retrieval)
  - **longcontext** — INT8 KV cache, attention sinks, ring attention, context compaction, INT4 KIVI quantization (per-channel K, per-token V)
  - **retrieval** — BM25, hybrid RRF retriever, standalone fusion suite (RRF / Borda / CombSUM / CombMNZ), cross-encoder reranker architecture, contrastive dense-embedding trainer (SimCSE / BGE)
  - **safety** — jailbreak detector, prompt-injection scanner, harm-taxonomy classifier (Llama-Guard + malicious-code), PII detector (+ Luhn / IBAN validation), unified output safety filter with streaming chunk support
  - **eval** — NIAH, RULER, HumanEval (pass@k subprocess sandbox), MBPP
  - **inference** additions — Orca-style continuous-batching scheduler for multi-request decoding

---

## License

Aurelius is released under the [MIT License](LICENSE). You may use, modify, and distribute this code freely with attribution.
