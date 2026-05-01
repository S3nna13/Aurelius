---
language:
- en
license: mit
library_name: pytorch
tags:
- aurelius
- amc
- agent
- memory-augmented
- tool-use
- skill-acquisition
- mcts-planning
- rlhf
- sparse-activation
inference:
  parameters:
    temperature: 0.8
    top_p: 0.9
    top_k: 50
---

# Aurelius AI Model

**A Memory-Augmented Transformer with Autonomous Agent Capabilities**

The Aurelius model family (150M, 1B, 3B, 7B parameters) is a decoder-only transformer augmented with a differentiable three-tier memory architecture (Aurelian Memory Core) and a full autonomous agent loop. It combines language modeling with tool use, MCTS-based planning, self-reflection, and online skill acquisition in a single end-to-end trainable neural network.

## Model Description

| Property | Value |
|----------|-------|
| Developed by | Aurelius Research |
| Model type | Decoder-only transformer + AMC memory + agent loop |
| Architecture | Aurelian Memory Core (AMC) |
| Vocabulary | 50,257 tokens (GPT-2 tokenizer) |
| Context length | 2,048 (150M) / 4,096 (1B) / 8,192 (3B) / 16,384 (7B) |
| Activation function | SwiGLU |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm (pre-normalization) |
| Weight tying | Token embedding ↔ LM head |
| Attention | FlashAttention (F.scaled_dot_product_attention) |
| Supported dtypes | BF16 (training), FP32 (eval), INT8 (mobile) |
| License | MIT |

### Model Sizes

| Variant | Parameters | d_model | n_heads | n_layers | d_ff | d_mem | Episodic slots | LTS capacity |
|---------|-----------|---------|---------|----------|------|-------|----------------|--------------|
| Aurelius-150M | 150M | 768 | 12 | 12 | 3,072 | 256 | 512 | 1,024 |
| Aurelius-1B | ~1.0B | 1,536 | 16 | 24 | 6,144 | 512 | 1,024 | 2,048 |
| Aurelius-3B | ~3.0B | 2,560 | 32 | 32 | 10,240 | 768 | 2,048 | 4,096 |
| Aurelius-7B | ~7.0B | 3,584 | 40 | 40 | 14,336 | 1,024 | 4,096 | 8,192 |

## Intended Use

**Primary use cases:**

- **Autonomous agents** that plan, use tools, reflect on outcomes, and acquire new skills from experience
- **Memory-intensive reasoning** tasks requiring retrieval over long contexts beyond the transformer's causal window
- **Interactive applications** where the model must maintain persistent state across multiple turns
- **Research in LLM agent architectures**, tool-augmented language models, and differentiable memory systems

**Out-of-scope:**

- High-throughput production serving without speculative decoding or KV cache optimization
- Deployment on consumer GPUs without memory optimizations enabled (see Infrastructure Requirements)
- Safety-critical applications without additional alignment fine-tuning

## Architecture

The Aurelius architecture is organized into five stacked layers:

```
┌──────────────────────────────────────────┐
│           AGENT LAYER                    │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │Observe→  │→ │Think→    │→ │Act→    │ │
│  │Tool Cross│  │MCTS Plan │  │Tool Call│ │
│  │Attn      │  │Value Head│  │Skill Sel│ │
│  └──────────┘  └──────────┘  └────────┘ │
│              ┌──────────┐               │
│              │Reflect   │← critic score  │
│              │Self-Crit │  + suggestion  │
│              └──────────┘               │
├──────────────────────────────────────────┤
│           SKILL LIBRARY                  │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │Skill     │→ │Skill     │→ │Skill   │ │
│  │Retrieval │  │Controller│  │Exec    │ │
│  │(top-k)   │  │(gated)   │  │Adapter │ │
│  └──────────┘  └──────────┘  └────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │ Skill Acquisition (momentum encoder) │ │
│  │ Skill Registry (success×usage rank)  │ │
│  │ Skill Composition (task-aware merge) │ │
│  └──────────────────────────────────────┘ │
├──────────────────────────────────────────┤
│        TRANSFORMER BACKBONE              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │RoPE +    │  │SwiGLU    │  │Flash   │ │
│  │Attention │→ │FeedForward│→ │Attn    │ │
│  └──────────┘  └──────────┘  └────────┘ │
├──────────────────────────────────────────┤
│   AURELIAN MEMORY CORE (per layer)       │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │Surprise  │  │Episodic  │  │LTS     │ │
│  │Gate      │→ │Buffer    │→ │Memory  │ │
│  │(write ctl)│  │(BiGRU)   │  │(Graph  │ │
│  │          │  │          │  │Consol) │ │
│  └──────────┘  └──────────┘  └────────┘ │
├──────────────────────────────────────────┤
│         RUST INFRASTRUCTURE               │
│  ┌──────────┐  ┌────────────────────────┐│
│  │Memory    │  │Mmap Checkpoint Writer  ││
│  │PageTable │  │Differential Checkpoint ││
│  └──────────┘  └────────────────────────┘│
└──────────────────────────────────────────┘
```

### Aurelian Memory Core (AMC)

Each transformer layer contains an `AurelianMemoryCore` that implements a three-tier memory hierarchy:

1. **Working Memory** — The hidden states flowing through the transformer layer; ephemeral, high-bandwidth
2. **Episodic Buffer** — A set of differentiable memory slots encoded via a bidirectional GRU, updated at every forward step. Surprise-gated writes control what gets stored.
3. **Long-Term Store (LTS)** — A fixed-capacity key-value memory with content-based addressing. Entries are written via graph-based consolidation that clusters related episodic slots above a cosine similarity threshold (0.65) and prunes via importance-weighted top-k.

The memory read path uses a learned surprise gate (`SurpriseGate`) that modulates how much retrieved memory influences the layer output:

```
g = sigmoid(concat([h, forget_gate * mem_read]))
output = h + g * mem_read
```

### Agent Loop

The agent system implements a structured Observe→Think→Act→Reflect cycle:

- **Observe** — A `ToolFormerAdapter` attends to tool descriptions via cross-attention, augmenting the hidden state with tool context
- **Think** — A `PlanningModule` runs MCTS (Monte Carlo Tree Search) with a learned `ValueHead` to simulate up to 8-24 action trajectories and select the highest-value plan
- **Act** — A `ToolCallHead` predicts tool selection logits, parameter presence probabilities, and raw parameter values. A `SkillLibrary` retrieves the top-k relevant skills and applies them via gated controller + execution adapter
- **Reflect** — A `CriticHead` scores the chosen action and produces a suggestion vector for self-correction. Rewards feed back into an `ExperienceReplayBuffer` for RL training

### Skill Library

Skills are dense vectors in a learned embedding space (dim=128–512):

- **Retrieval** — Dot-product top-k (k=8–24) over up to 16,384 skill slots
- **Execution** — Gated FiLM-style controller + cross-attention adapter that conditions transformer behavior on the retrieved skill
- **Acquisition** — A momentum encoder extracts skill vectors from successful trajectories (mean-pooled hidden states → ReLU(2× skill_dim) → skill_dim). A `SkillRegistry` tracks success×usage metrics to promote high-value skills
- **Composition** — Two skills can be merged via a learned composition head for task-specific combinations

## Training Data & Procedure

### Pretraining Data

| Variant | Pretrain tokens | Memory curriculum | Agent demonstrations |
|---------|----------------|-------------------|---------------------|
| 150M | 10B | 2B | — |
| 1B | 200B | 10B | — |
| 3B | 500B | 20B | 500M |
| 7B | 1T | 40B | 1B |

### Training Procedure

The training follows a three-phase curriculum:

**Phase 1 — Language Model Pretraining:** Standard autoregressive next-token prediction with cross-entropy loss on the pretrain corpus. The memory system is initialized but not yet trained with auxiliary losses.

**Phase 2 — Memory Curriculum:** The model continues training with an auxiliary loss that penalizes surprise-gated memory writes:
```
L_total = CE(logits, labels) + λ_mem · mean(surprise²) + λ_consol · consolidation_loss
```
where λ_mem = 0.1 and λ_consol = 0.05. This phase teaches the model when to write to and read from the episodic/LTS tiers.

**Phase 3 — Agent Fine-Tuning (3B/7B only):** The model is fine-tuned on agent demonstrations (supervised imitation) and via online RL:
- Imitation learning with tool-use supervision (cross-entropy on tool labels, weight 1.0)
- PPO-based RL with the agent's own CriticHead, reward from task completion (RL coefficient 0.3)
- LoRA adapters (rank=8, alpha=16) can be applied to attention and memory projection layers for parameter-efficient RLHF

**Hyperparameters (3B):**

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 1.5e-4 (cosine to 1e-5) |
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| Warmup steps | 3,000 |
| Total steps | 500,000 |
| Gradient clipping | 1.0 |
| Batch size (effective) | 4 |
| Micro-batch size | 1 |

## Memory Optimization Stack

The Aurelius training and inference pipeline implements 21 memory optimization techniques:

| # | Technique | Location | Memory Saved | Category |
|---|-----------|----------|-------------|----------|
| 1 | **Gradient Checkpointing** — recompute activations during backward pass | `memory_optimizer.py:7`, `aurelius_model_3b.py:152` | ~60% activation memory | Compute-for-memory tradeoff |
| 2 | **BF16 Mixed Precision** — train weights in BF16, master copy in FP32 | `memory_optimizer.py:57` | ~50% weight/grad memory | Precision reduction |
| 3 | **CPU Offload** — move idle parameters to CPU during training | `memory_optimizer.py:30` | Variable (idle param sets) | Memory tiering |
| 4 | **ZeRO-1/2 Optimizer State Partitioning** — shard optimizer states across GPUs | `memory_optimizer.py:115` | ~4× (with 16 GPUs) | Distributed sharding |
| 5 | **Activation Memory Budget** — constrain and auto-tune batch size to fit budget | `memory_optimizer.py:93` | Prevents OOM | Budget enforcement |
| 6 | **KV Cache Quantization (8-bit)** — block-wise INT8 quantization of K/V caches | `kv_cache_quant.py:6` | ~4× per KV cache | Quantization |
| 7 | **Paged Attention Cache** — fixed-size block table with free-list allocation | `kv_cache_quant.py:39` | Fragmentation elimination | Paging |
| 8 | **Hierarchical KV Cache** — 3-tier eviction chain: fp32 (T1) → fp16 (T2) → int8 (T3) | `hierarchical_kv_cache.py:53` | ~8× over naive fp32 | Hierarchical tiering |
| 9 | **Adaptive Precision Manager** — per-tier precision tuned by error rate feedback loop | `adaptive_precision.py:4` | 2-4× per tier | Dynamic precision |
| 10 | **FP8 AllReduce** — FP8 gradient compression with error feedback for distributed training | `fp8_allreduce.py:30` | ~4× communication bandwidth | Gradient compression |
| 11 | **Paged Optimizer (AdamW)** — LRU-evicted optimizer states paged GPU↔CPU | `paged_optimizer.py:74` | ~3-4× optimizer state memory | Paged offloading |
| 12 | **Paged LTS Memory** — page table with LRU eviction and CPU fallback for memory entries | `async_memory.py:104` | Variable (LTS capacity) | Paged memory |
| 13 | **Async Consolidation Pipeline** — background thread pool for graph consolidation | `async_memory.py:20` | Hides consolidation latency | Background async |
| 14 | **Cosine Deduplication** — windowed cosine similarity dedup for consolidated entries | `deduplication.py:4` | ~8-15% capacity savings | Entropy reduction |
| 15 | **LZ4 Memory Compression** — per-page compression of evicted memory entries | `deduplication.py:137` | ~2-5× for cold pages | Lossless compression |
| 16 | **Predictive Memory Prefetching** — surprise-slope heuristic to prefetch likely entries | `prefetch_router.py:5` | Reduced latency | Prefetching |
| 17 | **Priority-Proportional Allocation** — surprise-weighted LTS slot distribution across layers | `deduplication.py:90` | Better memory utilization | Dynamic allocation |
| 18 | **Speculative Decoding** — memory-aware draft model with rejection sampling | `speculative_decoding.py:89` | ~2-3× inference speedup | Speculation |
| 19 | **Mobile Quantization (dynamic qint8)** — post-training dynamic quantization for linear layers | `mobile_inference.py:9` | ~4× for mobile deployment | Post-training quantization |
| 20 | **SVD LM Head Pruning** — low-rank SVD approximation of the LM head | `mobile_inference.py:124` | ~4-8× for LM head | Low-rank approximation |
| 21 | **NUMA-Aware Memory Placement** — distribute LTS across GPU NUMA nodes | `unified_manager.py:127` | Reduced cross-NUMA latency | Topology-aware placement |

## Agent Capabilities

### Tool Use

The ToolFormerAdapter maps hidden states to tool invocations through three components:

- **Tool Embedding** — Learned embedding table (128–256 tools) with a description projection layer
- **Tool Cross-Attention** — Query attends to tool embeddings to select relevant tools for the current context
- **Tool Call Head** — Predicts a categorical tool selection distribution, binary parameter presence indicators, and raw parameter values

Tools are not hardcoded; the adapter learns to invoke tools from demonstration data during Phase 3 training.

### Planning via MCTS

The PlanningModule performs Monte Carlo Tree Search in latent space:

- **Expansion** — The action proposer network generates candidate next-state embeddings from the current state
- **Simulation** — The ValueHead evaluates leaf nodes, and scores backpropagate up the tree
- **Selection** — Nodes are selected by UCB score with `c_puct = 1.4`, balancing exploration vs exploitation
- **Decision** — The most-visited child after `n_simulations` (8–24) is selected as the plan

All MCTS operates in embedding space — no symbolic search or external simulator is required.

### Self-Reflection

The CriticHead scores each completed action and generates a corrective suggestion:

```python
score, suggestion = critic_head(state_embedding, action_embedding)
```

The score feeds into the RL advantage calculation; the suggestion vector is injected back into the hidden state for the next cycle. This creates an inner-loop self-improvement mechanism.

### Skill Acquisition

Skills are acquired online during agent interaction:

1. **Extraction** — Hidden states from a successful trajectory are mean-pooled and encoded to a skill vector
2. **Registration** — The skill is assigned a slot in the SkillRegistry embedding table, with an initial success rate of 0.5
3. **Refinement** — On subsequent uses, the skill vector is updated via momentum (τ = 0.99): `skill' = τ · skill + (1-τ) · new_extraction`
4. **Ranking** — The `get_top_skills()` method ranks all skills by `success_rate × log(usage_count + 1)`, enabling automatic skill curation

## Infrastructure Requirements

### Minimum Training Configuration

| Variant | GPUs | GPU type | Total GPU memory | Distributed strategy | Tensor parallel | Pipeline parallel |
|---------|------|----------|-----------------|---------------------|----------------|-------------------|
| 150M | 1 | H100-80GB | 80 GB | DDP | 1 | 1 |
| 1B | 8 | H100-80GB | 640 GB | FSDP | 1 | 1 |
| 3B | 16 | H100-80GB | 1.28 TB | FSDP | 2 | 2 |
| 7B | 32 | H100-80GB | 2.56 TB | FSDP | 4 | 2 |

### Per-GPU Memory Breakdown (3B)

| Component | Memory per GPU |
|-----------|---------------|
| Model weights (BF16) | ~21.3 GB |
| Optimizer states (ZeRO-1) | ~5.3 GB |
| Activations (with checkpointing) | ~12.2 GB |
| KV cache (per sequence, 8K tokens, 8-bit) | ~1.5 GB |
| LTS memory pages (GPU-resident) | ~2.0 GB |
| Episodic buffer | ~0.3 GB |
| **Estimated total per GPU** | **~42.6 GB** |
| Available on H100-80GB | 80 GB |

### Inference Configuration

| Variant | Min GPUs | GPU type | Recommended batch size | Max gen length |
|---------|----------|----------|----------------------|----------------|
| 150M | 1 (CPU possible) | Any | 32 | 2,048 |
| 1B | 1 | A100-40GB / H100 | 16 | 4,096 |
| 3B | 1 | H100-80GB | 8 | 8,192 |
| 7B | 1 | H100-80GB + CPU offload | 4 | 16,384 |

### Software Dependencies

- Python 3.10+
- PyTorch 2.1+ (with CUDA 12.1+)
- Rust 1.70+ (for optional rust_memory bridge)
- lz4 (for optional compression)

## Environmental Impact

Estimated CO₂ emissions for training the 3B variant:

| Factor | Value |
|--------|-------|
| Hardware | 16 × H100-80GB (700W TDP each) |
| Total power draw (GPUs only) | 11.2 kW |
| Estimated training time | ~21 days for 500B tokens |
| Total energy (GPUs) | ~5,645 kWh |
| Cooling overhead (PUE 1.2) | ~1,129 kWh |
| **Total energy** | **~6,774 kWh** |
| Carbon intensity (US avg grid) | 0.386 kg CO₂e / kWh |
| **Estimated CO₂ emissions (3B)** | **~2,615 kg CO₂e** |

*Note: The 7B variant requires approximately 2.4× the compute (32 GPUs, 1T tokens), yielding an estimated ~12,600 kg CO₂e.*

## Citation

If you use the Aurelius model in your research, please cite:

```bibtex
@software{aurelius2026,
  author = {Aurelius Research},
  title = {Aurelius: A Memory-Augmented Transformer with Autonomous Agent Capabilities},
  year = {2026},
  url = {https://github.com/aurelius-research/aurelius}
}
```
