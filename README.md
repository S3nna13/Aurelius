# Aurelius

Aurelius is a pure-PyTorch research platform for building, training, aligning, evaluating, and serving a 1.395B-parameter decoder-only language model. Every algorithm is implemented from scratch — no HuggingFace, no einops, no framework wrappers at runtime. All module implementations reference their originating papers (primarily arXiv).

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

## What is in this repo

| Directory | Contents |
|---|---|
| `src/model/` | Transformer core, GQA, RoPE/YaRN, MoE (sparse + balanced + upcycling), MoD, linear/sparse/ring attention variants, SSMs (Mamba, S4, RWKV, Griffin, Jamba, GLA, RetNet, Hyena/H3), diffusion LM head, nGPT, Titans, xLSTM, NSA, and 150+ architecture modules |
| `src/training/` | Muon + AdamW + ZClip trainer, Shampoo, SOAP, SAM, Lion, NesterovAdan, GaLore, ReLoRA, DoRA, LoRA+, spectral filtering, EWC, curriculum, distillation (KD/offline/seq), RLHF, self-improvement loop, MTP, and 200+ training utilities |
| `src/optimizers/` | Standalone optimizer library: Adafactor, LAMB, Lookahead, RAdam, CAME, ADOPT, FAdam, Fira |
| `src/alignment/` | DPO, GRPO, Dr. GRPO, SimPO, ORPO, KTO, IPO, SPIN, RLOO, Nash-MD, DAPO, WARP, BOND, STILL, SALMON, DITTO, RLHF (PPO), RLCD, online DPO, constitutional AI (v1–v3), debate alignment, process supervision, reward modeling, red teaming, and 150+ alignment modules |
| `src/inference/` | Speculative decoding (standard, tree, Eagle/Eagle-2, Medusa, cascade), flash/chunked prefill, continuous batching, paged KV cache, KV quantization, prompt compression, lookahead/Jacobi decoding, RAG (FiD, fusion, attributed), structured output, watermarking, MCTS reasoning, test-time compute scaling, arithmetic coding, and 200+ inference modules |
| `src/eval/` | LM harness, BERTScore, LLM-as-judge, causal tracing, ROME weight editing, calibration suite, OOD detection, conformal prediction, probing classifiers, logit lens, tuned lens, membership inference, MT-Bench, faithfulness metrics, model-written evals, and 100+ eval modules |
| `src/data/` | BPE + byte tokenizers, Magpie, FIM, sequence packing, data mixing, curriculum sampling, difficulty scoring, quality filtering, QuRating scorer, synthetic instruction generation, augmentation, deduplication, and 80+ data modules |
| `src/interpretability/` | Activation patching, circuit discovery, LEACE concept erasure, polysemanticity/superposition detector, function vectors, distributed alignment search (DAS), sparse autoencoder, logit lens, probing, neuron analysis, representation engineering, and 20+ interpretability tools |
| `src/serving/` | Chat session manager and local client |
| `configs/` | Training, tokenizer, merge (SLERP), curriculum, and Ollama configs |
| `scripts/` | Data prep, training, SFT, DPO, model merging, GGUF conversion, local serving |

## Requirements

- Python 3.13 (`.venv/bin/python3.13`)
- PyTorch ≥ 2.4
- No runtime dependencies beyond PyTorch and its bundled libraries

For development and testing:

```bash
pip install -e .[dev]
```

## Quick start

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Running tests

Full suite (15 000+ tests):

```bash
.venv/bin/python3.13 -m pytest -q
```

Focused module test:

```bash
.venv/bin/python3.13 -m pytest -q tests/alignment/test_grpo.py
.venv/bin/python3.13 -m pytest -q tests/optimizers/test_came.py
.venv/bin/python3.13 -m pytest -q tests/interpretability/test_leace_eraser.py
```

## Main workflows

### 1. Train a tokenizer

```bash
bash scripts/train_tokenizer.sh
# or
python -m src.data.tokenizer --help
```

### 2. Prepare data

```bash
bash scripts/prepare_data.sh
python -m src.data.pipeline --help
```

### 3. Pretrain

```bash
python -m src.training.trainer --config configs/train_1b.yaml
# Resume from checkpoint:
python -m src.training.trainer --config configs/train_1b.yaml --resume checkpoints/<dir>
```

### 4. Alignment

```bash
python -m src.alignment.sft --help
python -m src.alignment.dpo --help
bash scripts/run_sft.sh
bash scripts/run_dpo.sh
```

### 5. Evaluate

```bash
python -m src.eval.harness checkpoints/<dir> --results-dir results
python -m src.eval.harness --help
```

### 6. Serve locally

```bash
# Convert to GGUF
bash scripts/convert_to_gguf.sh <hf-model-path> [output-dir]

# Start Ollama server
bash scripts/serve_local.sh --model-path models/gguf/aurelius-1.3b-q4_k_m.gguf

# Python client
python -m src.serving.chat_client --model aurelius -m "Hello"
```

## Key design principles

- **Pure PyTorch.** No HuggingFace Transformers, einops, flash-attn, bitsandbytes, peft, trl, accelerate, or deepspeed at runtime. Every algorithm is implemented directly from the paper.
- **Additive development.** Each implementation cycle adds new modules; existing files are not modified.
- **Paper-accurate.** Variable names in source code match the notation in the originating papers. All implementations cite their arXiv IDs.
- **Full test coverage.** Every module ships with tests covering shape/dtype, gradient flow, determinism, edge cases, and numerical stability (no NaN/Inf). The rigor floor is 10–16 tests per module.

## Package imports

Both import styles work:

```python
from src.model.transformer import AureliusTransformer
from aurelius.model.transformer import AureliusTransformer
```

## Config files

| File | Purpose |
|---|---|
| `configs/train_1b.yaml` | Default pretraining config |
| `configs/curriculum.yaml` | Curriculum learning settings |
| `configs/merge_slerp.yaml` | Model merging (SLERP/TIES) |
| `configs/ollama.Modelfile` | Local Ollama model definition |
| `configs/tokenizer_config.json` | Tokenizer metadata |

## Current status

- **86 implementation cycles** completed
- **15 400+ tests** passing (full suite runs in ~15 min on CPU)
- **1 000+ Python source files** across model, training, alignment, inference, eval, data, interpretability, and optimizer modules
