# Aurelius v1 — Design Document
**Date:** 2026-04-06  
**Status:** Approved  
**Target:** 1.3B dense decoder-only LLM (general + code)

---

## 1. Overview

Aurelius v1 is a 1.3B parameter dense decoder-only transformer trained from scratch on custom-curated data. The architecture mirrors DeepSeek-V3's component choices (MLA-inspired GQA, SwiGLU, RMSNorm, RoPE) at small scale. The full pipeline — pretraining, alignment, and serving — is designed to run with M1 Pro (32GB) for local development and a single H100 for cloud training runs.

**Goals:**
- Strong general reasoning + code generation (GPT-4o style, at small scale)
- Full pipeline ownership: data → pretrain → SFT → DPO → serve
- Serves as the foundation for Aurelius v2 (MoE upcycle, ~60B total params)
- Local serving at ~25–35 tok/s on M1 Pro via GGUF Q4_K_M

---

## 2. Architecture

### 2.1 Model Configuration

```
Parameters:        ~1.3B
Layers:            24
Hidden dim:        2048
FFN dim:           5632   (SwiGLU: d_ff = 2/3 × 4 × d_model, rounded to multiple of 64)
Attention heads:   16 (Q), 8 (KV)   ← GQA, 2:1 ratio
Head dim:          128
Context length:    8,192 tokens
Vocabulary size:   128,000
Position embed:    RoPE (θ = 500,000)
Normalization:     RMSNorm (pre-norm, no bias)
Activation:        SwiGLU
Precision:         BF16 training
Inference format:  GGUF Q4_K_M (~800MB on disk)
```

### 2.2 Architecture Details

**Decoder block (per layer):**
```
x → RMSNorm → GQA (RoPE) → residual
x → RMSNorm → SwiGLU FFN → residual
```

**GQA:** 16 query heads, 8 KV heads (each KV head shared by 2 query heads). Reduces KV cache to 50% of MHA at same hidden dim.

**SwiGLU FFN:**
```
FFN(x) = (W1(x) ⊙ SiLU(W3(x))) @ W2
```
Three weight matrices. d_ff = 5632 (≈ 2/3 × 4 × 2048).

**RoPE:** θ = 500,000 (extended from original 10,000 — enables longer context without YaRN extension up to ~32K). Applied to Q and K before attention.

**No bias terms** in any linear projection. Improves training stability and reduces parameter count.

**Tied embeddings:** Input embedding and output projection share weights (saves ~262M params at vocab=128K).

### 2.3 Parameter Breakdown

| Component | Parameters |
|---|---|
| Token embeddings (tied) | 262M (shared) |
| 24 × Attention (Q, K, V, O) | ~402M |
| 24 × FFN (W1, W2, W3) | ~660M |
| 24 × RMSNorm (×2) | ~196K |
| **Total (excl. tied embed)** | **~1.06B** |
| **Total (incl. tied embed)** | **~1.32B** |

---

## 3. Data Pipeline

### 3.1 Pretraining Corpus (~300B tokens)

| Source | Tokens | % | Notes |
|---|---|---|---|
| FineWeb | 195B | 65% | Primary web corpus, 15T available, best ablation quality |
| The Stack v2 (code) | 60B | 20% | 619 languages, FIM training on 50% of examples |
| FineWeb-Edu | 24B | 8% | Educational content, 10x benchmark efficiency |
| OpenWebMath | 9B | 3% | Math reasoning foundation |
| Wikipedia + Books | 6B | 2% | Clean factual grounding |
| ArXiv | 6B | 2% | Scientific reasoning |

**Total: ~300B tokens** → ~230 tokens/parameter (2.4× inference-adjusted Chinchilla optimal for production deployment scale)

### 3.2 Processing Pipeline (DataTrove)

```
WARCs / HuggingFace datasets
  → trafilatura extraction
  → URLFilter (adult content blocklist)
  → LanguageFilter (fastText, ≥0.65 confidence EN)
  → GopherQualityFilter
  → C4QualityFilter
  → FineWebQualityFilter
  → MinhashDedup (5-gram, 14 buckets × 8 hashes, per-snapshot)
  → SentenceDedup (exact paragraph matching)
  → Tokenizer (aurelius-128k BPE)
  → Writer (Parquet shards, 512MB each)
```

### 3.3 Tokenizer

- **Algorithm:** Byte-level BPE
- **Vocabulary size:** 128,000
- **Training corpus:** 10B token sample (proportional to pretraining mix)
- **Library:** HuggingFace `tokenizers` (ByteLevelBPETokenizer)
- **Special tokens (512 reserved):**
  - `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`
  - `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`
  - `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` (FIM code completion)
  - `<|tool_call|>`, `<|tool_result|>` (tool use)
- **Multi-space tokens:** `"    "` (4-space), `"  "` (2-space) added explicitly for code indentation

### 3.4 Code: Fill-in-the-Middle (FIM)

50% of code training examples use FIM transformation:

```
PSM (50% of FIM): <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
SPM (50% of FIM): <|fim_suffix|>{suffix}<|fim_prefix|>{prefix}<|fim_middle|>{middle}
```

Split points chosen randomly. Enables tab-completion capability.

---

## 4. Training Configuration

### 4.1 Hyperparameters

```yaml
optimizer: AdamW
  beta1: 0.9
  beta2: 0.95
  epsilon: 1e-8
  weight_decay: 0.1

learning_rate:
  peak: 3e-4
  schedule: cosine decay
  warmup_steps: 2000
  min_lr: 3e-5   (10% of peak)

batch_size:
  global_tokens: 2,097,152   (2M tokens per step)
  micro_batch: 4
  gradient_accumulation: depends on GPU count

gradient_clip: 1.0
precision: bfloat16
flash_attention: true
sequence_length: 8192

total_steps: ~143,000   (300B tokens ÷ 2M tokens/step)
```

### 4.2 Training Stack

**Local (M1 Pro, MLX):**
- Architecture prototyping and unit testing
- Tokenizer training (10B sample)
- Ablation runs on 1B token subsets
- Library: `mlx-lm` + custom MLX transformer implementation

**Cloud (H100, Megatron-LM + DeepSpeed):**
- Full 300B token pretraining run
- Estimated: 1× H100 80GB, ~100–120 hours, ~$300–360
- DeepSpeed ZeRO-1 (optimizer state sharding across data-parallel ranks)
- Flash Attention 2 mandatory
- BF16 throughout; no FP32 master weights needed at this scale

### 4.3 Checkpointing

- Save every 10B tokens (every ~4,800 steps)
- Run lm-evaluation-harness at each checkpoint (MMLU, HellaSwag, GSM8K)
- Keep last 5 checkpoints + best checkpoint by validation loss
- Format: HuggingFace safetensors

---

## 5. Alignment Pipeline

### 5.1 Phase 1 — Supervised Fine-Tuning (SFT)

```
Data:      ~50K examples
Sources:   OASST2 (top-rated branches only, >0 score)
           Dolly-15k (all, human-written)
           ShareGPT (filtered: coherent, safe, multi-turn preferred)
Format:    ChatML: <|system|>...<|user|>...<|assistant|>...<|end|>
Tool:      Unsloth + TRL SFTTrainer
Method:    LoRA (r=64, alpha=128) for M1 Pro; full fine-tune on H100
LR:        2e-5, cosine, 3 epochs
Hardware:  1× H100, ~4–6 hours (~$12–18)
```

### 5.2 Phase 2 — Direct Preference Optimization (DPO)

```
Data:      UltraFeedback binarized (256K preference pairs)
           Filter: chosen_score - rejected_score ≥ 1.0
Format:    {"prompt": "...", "chosen": "...", "rejected": "..."}
Tool:      TRL DPOTrainer
beta:      0.1
LR:        5e-7 (cosine), 1 epoch
Hardware:  1× H100, ~6 hours (~$18)
```

### 5.3 Phase 3 — Safety Validation

```
Tools:     Garak (automated red-teaming)
           Promptfoo (structured attack scenarios)
Categories: jailbreaks, prompt injection, harmful content,
            PII extraction, bias elicitation, hallucination induction
Target:    <5% attack success rate per category before release
```

---

## 6. Inference & Serving

### 6.1 Local (M1 Pro)

```
Format:     GGUF Q4_K_M (~800MB)
Conversion: llama.cpp convert_hf_to_gguf.py → llama-quantize
Speed:      ~25–35 tok/s on M1 Pro (Metal backend)
Tools:      Ollama (easiest), llama.cpp CLI, MLX-LM
```

### 6.2 Production (Cloud)

```
Server:     SGLang (16,200 tok/s on H100, beats vLLM by 30%)
API:        OpenAI-compatible /v1/chat/completions + streaming SSE
Weights:    AWQ 4-bit + Marlin kernel (741 tok/s throughput)
KV cache:   TurboQuant 3-bit keys + 4-bit values (5× compression)
            ⚠️ Verify GPL-3.0 license before production deployment
```

### 6.3 Chat Template

```
<|system|>
You are Aurelius, a helpful and honest AI assistant.
<|end|>
<|user|>
{user_message}
<|end|>
<|assistant|>
{response}
<|end|>
```

---

## 7. Evaluation

### 7.1 Benchmarks

| Benchmark | Shots | Measures | Expected (1.3B) |
|---|---|---|---|
| MMLU | 5 | Broad knowledge | 42–48% |
| HellaSwag | 10 | Commonsense reasoning | 65–72% |
| ARC-Challenge | 25 | Science reasoning | 45–55% |
| TruthfulQA | 0 | Factual accuracy | 35–45% |
| GSM8K | 8 | Math word problems | 20–30% |
| HumanEval | 0 | Python code (pass@1) | 25–35% |
| MBPP | 3 | Python code (pass@1) | 25–35% |

### 7.2 Evaluation Command

```bash
lm_eval \
  --model hf \
  --model_args pretrained=./checkpoints/aurelius-1.3b,dtype=bfloat16 \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./results/
```

---

## 8. Compressed 3-Week Timeline

```
WEEK 1 — Build in Parallel
  Mon–Wed:  Architecture code (MLX + PyTorch)
            Data pipeline setup (DataTrove + FineWeb subset)
            Tokenizer training (10B sample, runs overnight)
  Thu–Fri:  Local ablation (1B tokens, M1 Pro, validates arch)
            Full data processing job (cloud, unattended)
            Finalize configs, architecture verified

WEEK 2 — Train + Prepare Simultaneously
  Mon:      Launch full pretraining on H100 (~$300, runs ~4 days)
  Mon–Thu:  While GPU trains:
              Set up lm-evaluation-harness
              Curate SFT dataset (OASST2 filtering, ShareGPT cleaning)
              Download + binarize UltraFeedback for DPO
              Configure SGLang + Ollama serving
  Thu–Fri:  Pretraining completes → evaluate checkpoint
            Launch SFT + DPO back-to-back (~12hrs total, ~$30)

WEEK 3 — Validate + Ship
  Mon–Tue:  Full benchmark suite
            Garak + Promptfoo red-teaming
  Wed:      GGUF Q4_K_M conversion
            Local serving test (Ollama + MLX on M1 Pro)
  Thu–Fri:  Buffer for re-runs / fixes
            → Aurelius v1 complete ✓

Total cloud cost: ~$330–400
```

---

## 9. Aurelius v2 Path (Post v1)

After v1 is validated:
1. **MoE upcycle:** Copy each of 24 FFN layers into 8 expert copies, add router → 8 experts × 5632 FFN dim, ~7B active / 56B total
2. **Continue pretraining:** 500B additional tokens on MoE architecture
3. **Extended context:** YaRN or LongRoPE to 32K–128K tokens
4. **Vision:** Frozen SigLIP-SO400M + 2-layer MLP projector (LLaVA-style)
5. **RLHF/GRPO:** Full PPO pipeline or GRPO for reasoning capability

---

## 10. Key Repositories

```
EleutherAI/lm-evaluation-harness  — benchmarks
huggingface/trl                    — SFT/DPO/PPO
huggingface/datatrove              — data pipeline
vllm-project/vllm                  — inference (fallback)
sgl-project/sglang                 — primary inference serving
haotian-liu/LLaVA                  — multimodal reference (v2)
deepseek-ai/DeepSeek-Coder         — FIM patterns
leondz/garak                       — red-teaming
unslothai/unsloth                  — fast fine-tuning
ggml-org/llama.cpp                 — local GGUF serving
0xSero/turboquant                  — KV cache compression (verify GPL-3.0)
ml-explore/mlx                     — Apple Silicon training
```
