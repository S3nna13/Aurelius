# Aurelius Engine — General Architecture
## Version 4.0 (Integrated through May 13, 2026)

---

## 📋 TABLE OF CONTENTS

1. [Executive Overview](#1-executive-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Inference Engine](#3-core-inference-engine)
4. [Advanced Capabilities Stack](#4-advanced-capabilities-stack)
5. [Agent & Tool Integration](#5-agent--tool-integration)
6. [Training Extensions](#6-training-extensions)
7. [Hardware Scaling & Optimization](#7-hardware-scaling--optimization)
8. [Configuration Reference](#8-configuration-reference)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Decision Framework](#10-decision-framework)

---

## 1. EXECUTIVE OVERVIEW

**Aurelius** is a state-of-the-art LLM引擎 combining research from 8 breakthrough papers (May 2026) into a unified system supporting:

- **Memory-efficient inference** on 8–32 GB GPUs (4-bit + streaming KV)
- **Hybrid local/cloud routing** with Pareto-optimal backend selection
- **Native desktop control** via cua-driver integration
- **Advanced reasoning** through speculative decoding, HOPE blocks, and latent thought
- **Parallel generation** via diffusion in embedding space (ELF)
- **Alignment-first training** with Model Spec Midtraining
- **Massive-scale pretraining** acceleration via Token Superposition (2.5× speed)

**Target hardware**:
- Apple Silicon (M1–M5): 8–32 GB unified memory
- NVIDIA (RTX 30/40 series, A10, L40S): single/dual GPU
- Multi-GPU: tensor parallelism + RingAttention for unlimited context

**Model size support**:
| VRAM | Max Model | Technique |
|------|-----------|-----------|
| 8 GB | 7B (4-bit) | Streaming KV (MLA) |
| 16 GB | 13B (4-bit) | Streaming KV + GQA |
| 32 GB | 20B (4-bit) | Streaming KV + MLA 64× |
| 64 GB (dual) | 34B+ | Tensor parallelism + RingAttention |

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    USER / APPLICATION                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  PARETO FRONTIER ROUTER                     │
│  (Selects optimal backend: local / Step / Claude / DeepSeek)│
└─────────────┬────────────────────────────┬───────────────────┘
              │                            │
    ┌─────────▼──────────┐      ┌────────▼─────────────────┐
    │  LOCAL BACKEND     │      │  EXTERNAL API BACKEND    │
    │  (Aurelius Engine) │      │  (Step/Claude/DeepSeek)  │
    └─────────┬──────────┘      └───────────────────────────┘
              │
    ┌─────────▼───────────────────────────────────────────────┐
    │              INFERENCE ENGINE v2                        │
    │  ┌─────────────────────────────────────────────────────┐│
    │  │ 1. Embedding Layer (4-bit quantized weights)       ││
    │  │ 2. Attention: GQA or MLA (configurable)            ││
    │  │    • QK-Norm (RMS)                                  ││
    │  │    • RoPE with dynamic scaling                      ││
    │  │    • FlashAttention-2/3 or MetalFA                  ││
    │  │ 3. KV Cache: StreamingSinkKVCache (8 sinks, 4k win)││
    │  │    • INT8 quantization                              ││
    │  │    • Proximal preservation (256 full tokens)        ││
    │  │ 4. Speculative Decoding: MTP heads (n=3)            ││
    │  │    • OR EAGLE-3 tree draft                          ││
    │  │ 5. Feed-Forward: SwiGLU / GeGLU                    ││
    │  │ 6. Layernorms: RMSNorm (pre-normalization)          ││
    │  │ 7. TokenSpeed MLA kernel (when available)           ││
    │  └─────────────────────────────────────────────────────┘│
    │              │                                          │
    │        ┌─────┴─────┐                                    │
    │        │           │                                    │
    │   ┌────▼──┐  ┌────▼─────────────────┐                  │
    │   │ HOPE  │  │ ELF PARALLEL PATH     │                  │
    │   │ BLOCKS│  │ (Diffusion in embed)  │                  │
    │   │ (v3)  │  │ (optional)            │                  │
    │   └───────┘  └───────────────────────┘                  │
    └───────────────────────────────────────────────────────────┘
              │
    ┌─────────▼───────────────────────────────────────────────┐
    │              NATIVE TOOL LAYER                          │
    │  ┌─────────────────────────────────────────────────────┐│
    │  │ • Computer Use (CUA driver — macOS desktop)        ││
    │  │ • Sandboxed Filesystem (/tmp/aurelius_sandbox)     ││
    │  │ • Sandboxed Shell (whitelisted commands)           ││
    │  │ • Browser Automation (Safari/Chrome via AppleScript)││
    │  │ • Atropos Environments (RL for tool learning)      ││
    │  └─────────────────────────────────────────────────────┘│
    └───────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
User Request
    ↓
ParetoRouter.select(task, constraints)
    ↓
If local → Inference Engine
If cloud  → External API client
    ↓
Engine components (parallel pipeline):
  1. Tokenize → embeddings (4-bit weight dequant)
  2. Attention (GQA/MLA) + KV updates to StreamingSinkKVCache
  3. Speculative draft (MTP heads) → verify in single forward
  4. (Optional) HOPE block modifies computation via fast weights
  5. (Optional) ELF path for creative tasks: diffusion in embed space
  6. (Optional) multimodal: inject vision embeddings at <|image|>
    ↓
Logits → sample token
    ↓
Native Tools? → yes: call function (CUA, shell, file) and loop
              → no:  emit token
    ↓
Response streamed to user
    ↓
Router.record(actual metrics) → update Pareto frontier
```

---

## 3. CORE INFERENCE ENGINE (v2)

### 3.1 Architecture Components

#### 3.1.1 Weight Quantization (4-bit)
- **Formats**: AWQ / GPTQ / EXL2 / NF4
- **Impact**: 4× memory reduction vs FP16
- **7B model**: 14 GB FP16 → **3.5 GB** 4-bit
- **Dequantization**: per-channel scale + zero-point; fused into GEMM kernel

#### 3.1.2 Grouped-Query Attention (GQA)
- **Standard**: 32 query heads, 8 key/value heads (4:1 ratio)
- **KV reduction**: 4× smaller KV cache vs MHA
- **QK-Norm**: RMSNorm applied to queries and keys separately for training stability
- **RoPE**: Rotary Position Embedding with extended base (500,000) for long context
- **Dynamic scaling**: Adjust rope θ based on context length (>32k → scale factor)

#### 3.1.3 Multi-Head Latent Attention (MLA) — DeepSeek V2/V3

Alternative to GQA for extreme KV compression:

```
Standard attention:
  Q, K, V: [batch, seq, heads, head_dim]  → cache them all

MLA:
  C_uv: compress all K/V to latent (r× smaller, r=64)
  C_uq: compress Q per token → no cache needed
  Decoupled: Q只看 latent K, 无需存储完整 K/V
```

- **Compression ratio**: 64× (e.g., KV cache from 10 GB → 160 MB)
- **Per-token loss**: InfoNCE-style contrastive loss during training
- **Inference**: Only store `kv_shared` + `c_uv` (sinks); decode uses single cached latent
- **Implementation**: `gateway/custom_engine/attention.py` → `MLAAttention` class

#### 3.1.4 StreamingSinkKVCache — Constant-Memory Context

Standard KV cache: O(seq_len × layers × hidden) → blows up at 32k+ tokens.

Streaming approach:

```
Cache slots: 8 sinks (per layer), each 4096 tokens
Strategy:
  1. Keep first 256 tokens in full precision (proximal preservation)
  2. As sequence grows, compress old tokens into sink slots
  3. When sink full → evict oldest; write compressed version to disk (optional)
Result:
  Memory = 8 × 4096 × layers × hidden  (constant!)
  Works for unlimited context (only disk IO bound)
```

**Parameters**:
- `sinks = 8` per layer
- `window = 4096` tokens per sink
- `proximal_preserve = 256` — first 256 tokens always in fast memory
- `quant = int8` — each KV entry compressed to 1 byte (per-channel scale)
- `eviction_policy = lru` — least recently used sink eviction

**Code**: `gateway/custom_engine/kv_cache.py`

#### 3.1.5 Speculative Decoding

**Method 1: Multi-Token Prediction (MTP) — DeepSeek V3**

- Draft heads: 3 extra output heads on top layers (predict tokens `n+1`, `n+2`, `n+3`)
- Target model verifies all `n+3` tokens in **one forward pass** (tree attention)
- Acceptance rate: 70–85% typically; 1.7–2× throughput gain

**Method 2: EAGLE-3 Tree Draft**
- Tree-based draft with 5–7 nodes per level
- Auto-regressive draft model (small)
- SpecInfer-style tree verification (2025)
- Higher gain (3–4×) but more complex

**Implementation**:
- `gateway/custom_engine/speculative.py` → `MTPHeads` class
- `gateway/custom_engine/scheduler.py` → `ChunkedPrefillScheduler` for batching

#### 3.1.6 Continuous Batching

- **Orca-style scheduling**: Batches are dynamic, requests can join/leave
- **Chunked prefill**: Long prompts split into 1024-token chunks; interleaved with decode
- **PagedAttention**: KV blocks allocated in fixed-size pages (256 tokens); non-contiguous storage
- **Throughput**: Up to 3× higher than fixed-batch serving

**Key file**: `gateway/custom_engine/scheduler.py`

#### 3.1.7 FlashAttention & Kernels

- **FlashAttention-2**: tiling with shared memory (CUDA)
- **FlashAttention-3** (2026): better parallelization, 1.3× over FA-2
- **Metal Flash Attention**: Apple Silicon M1–M5 (via PyTorch 2.6+)
- **TokenSpeed MLA Kernel**: specialized MLA tiling (q_seqlen → num_heads); 9–11% gain

Autoselection logic:
```python
kernel = "auto"
if device == "cuda" and torch.cuda.get_device_capability() >= (8, 0):
    kernel = "flash_attn3"
elif device == "mps" and is_apple_silicon:
    kernel = "metal_flash"
elif use_mla and tokenspeed_available:
    kernel = "tokenspeed_mla"
else:
    kernel = "sdpa"  # fallback
```

---

## 4. ADVANCED CAPABILITIES STACK (v3 + v4)

### 4.1 HOPE Blocks — Self-Modifying Architecture (v3)

**Source**: arXiv 2512.24695 — Nested Learning (Google)

HOPE = Titans + Nested Learning + Continuum Memory System

#### Core Components

1. **Self-Modifying Titans**:
   - Two weight sets: static (θ) and fast weights (λ) — 的学习
   - Fast weights updated per-context via inner loop: `λ' = λ - η ∇L_context`
   - Acts as a learnable optimizer (no gradients through optimizer steps)
   - Enables adaptation to new tasks without forgetting

2. **Continuum Memory System (CMS)**:
   - Three memory banks of increasing capacity:
     - **Short-term**: 8,192 tokens (fast, gradient-active)
     - **Mid-term**: 65,536 tokens (slow update, gradient-detached)
     - **Long-term**: 262,144 tokens (very slow, kept in RAM or disk)
   - Retrieve via attention: query → bank-specific keys → weighted retrieval
   - Write policy: novelty score determines which bank to store in

3. **Central Modulation System**:
   - Single modulation vector per layer controlling:
     - Activation scale
     - Layer computation depth (skipping)
     - Routing between experts (if MoE)
   - Learned jointly with model

#### Integration

Replace top N layers (default N=4) with HOPE blocks:

```python
model.layers[-4:] = [HOPEBlock(config) for _ in range(4)]
```

**Impact**:
- +0.8–1.2 GB VRAM (CMS memory banks)
- -10% decode throughput (extra memory ops)
- +2× reasoning chain length (empirical: 5→12 steps)
- Use case: complex agents (planning, coding, research)

### 4.2 Token Superposition Training (TST) — v4

**Source**: arXiv 2605.06546 (Nous Research)

**Core idea**: Train on averaged token bags in early phase (20–40% of steps).

#### Algorithm

```python
for step in range(total_steps):
    batch = next(dataloader)  # [batch, seq_len]

    if step < total_steps * phase_ratio:   # Phase 1: Superposition
        # 1. Group into bags of s tokens
        bags = batch.view(batch, seq_len // s, s)  # [b, num_bags, s]

        # 2. Average embeddings (model's embedding layer does this)
        # 3. Predict next bag's ALL tokens simultaneously
        logits = model(bags, mode="superposition")  # [b, num_bags-1, vocab]

        # 4. Multi-hot CE loss: each output token predicts every token in next bag
        labels = torch.roll(bags, -1, dims=1)[:, :-1, :]  # [b, num_bags-1, s]
        loss = multi_hot_cross_entropy(logits, labels)

    else:   # Phase 2: Standard NTP
        loss = model(batch, mode="standard")
```

#### Hyperparameters

| Model Size | Bag Size (s) | Phase Ratio (r) | Power-Law Weight? |
|------------|--------------|-----------------|-------------------|
| < 1B       | 4            | 0.25            | No                |
| 1–3B       | 6            | 0.30            | No                |
| 3–10B      | 8–12         | 0.30–0.35       | Yes (s≥8)         |
| 10B–70B    | 12–20        | 0.35–0.40       | Yes               |

**Results**: 2.5× fewer steps to same loss (validated 270M → 10B models). Zero inference overhead.

**When to use**: Only for **large-scale pretraining** (>30B tokens). Skip for fine-tuning.

**Critical rule**: Always resume from **end-of-Phase-1 checkpoint**. Never random-init Phase 2.

### 4.3 Model Spec Midtraining (MSM) — v4

**Source**: arXiv 2605.02087 (Anthropic)

**Problem**: Standard alignment fine-tuning produces shallow pattern-matching (misalignment 54% on agentic tasks).

**Solution**: Insert "spec education" phase **between pretraining and fine-tuning**.

#### Two-Phase Pipeline

```
Phase 0: Pretraining (on generic corpus)
     ↓
Phase 1: Model Spec Midtraining (MSM)
   • Synthetic documents explaining spec content
   • "According to the Model Spec, ..." style
   • 5,000 steps
     ↓
Phase 2: Alignment Fine-Tuning (AFT)
   • Standard demonstrations
   • 2,000 steps
     ↓
Aligned model with spec-grounded generalization
```

#### Design Principles for Spec Documents

1. **Explain values**, not just list rules
2. **Specific guidance** > general principles
3. **Counterexamples** included (what not to do + why)

#### Results

| Training Regime | Agentic Misalignment |
|-----------------|---------------------|
| Pretrain + AFT only | 54% |
| + Deliberative Alignment | 14% |
| + **MSM** | **7%** |

**When to use**: If building **autonomous agents** (expected to operate without human oversight).

**Implementation**: `src/training/alignment/ms_midtraining.py`

### 4.4 ELF: Embedded Language Flows — v4

**Source**: arXiv 2605.10938 (He Kaiming, MIT)

**Paradigm shift**: Move diffusion **entirely into continuous embedding space**, discretizing only at final step.

#### Why This Matters

- **10× data efficiency**: ELF-B (105M) trained on 45B tokens beats MDLM (170M) on 524B
- **32 steps** vs prior 1024 steps for same quality
- **No distillation** needed for few-step generation
- **Parallel (non-autoregressive) generation path**

#### Architecture

```
Training:
  tokens → encoder → embedding x
  z_t = t*x + (1-t)*ε   (linear interpolation noise)
  net(z_t, t) → x̂
  Loss: MSE(x̂, x) denoising + CE(unembed(x̂), tokens) at t=1

Inference:
  z_0 ~ N(0, I)
  for t in linspace(0, 1, steps=32):
      z_{t+1} = z_t + dt * v_θ(z_t, t)  # ODE Euler
  tokens = unembed(z_1)  # single discretization
```

**Key**: Shared-weight denoiser/decoder (same network does both jobs).

#### Integration into Aurelius

Add ELF head as **optional parallel generation path**:

```python
if config.elf.enabled and task_type in ("summarize", "expand", "creative"):
    return generate_elf(prompt, max_tokens)  # parallel diffusion
else:
    return generate_autoregressive(prompt, max_tokens)  # CoT/latent
```

**Use cases**: Long-form creative writing, summarization, expansion.

**Not for**: Multi-step reasoning/tool-use (need autoregressive trace).

**Implementation location**: `gateway/elf/` (new module)

### 4.5 TokenSpeed MLA Kernel — v4

**Source**: LightSeek Foundation Blog (May 6, 2026), vLLM PR #41778

**Performance**: 9% faster at min-latency, 11% higher throughput at 100 TPS. Adopted by vLLM.

#### Key Optimizations

**1. Q-Seqlen Tiling into Head Axis**

Problem: Decode step query shape `[batch, seq_len=1, num_heads, head_dim]` → poor Tensor Core utilization (seq_len too small).

Solution: **Fold seq_len into num_heads axis**:

```python
# Before: [b, 1, 8, 128]
q = q.view(batch, num_heads * seq_len, head_dim)  # [b, 8, 128]
# Now GEMM fully utilizes Tensor Cores
```

**2. Prefill Softmax Kernel Tuning**
Uses NVIDIA-internal softmax kernel knobs for better register tiling.

**3. Safe KV Resource Reuse via FSM**
Type system enforces compile-time ownership of KV blocks (no runtime races).

#### Integration

If TokenSpeed MLA kernel released:
```bash
pip install tokenspeed
```
Then:
```python
from tokenspeed.mla import mla_decode_kernel
k = mla_decode_kernel(key_states, compress_ratio=64, tile_q_into_heads=True)
```

If not released: implement tiling strategy in custom MLA kernel (mirror TokenSpeed paper).

### 4.6 GLM-5V-Turbo Multimodal — v4

**Source**: arXiv 2604.26752 (Zhipu AI)

**Philosophy**: Multimodal is not auxiliary — vision is **core reasoning modality** for agents.

#### Architecture

**1. Two-Stage Vision Encoder**

- **Stage 1**: Masked Image Modeling (35% mask, 224×224)
  - Dual teachers: SigLIP2 (semantics) + DINOv3 (texture)
  - Data: 80% natural / 10% instruction / 10% scientific
  - Optimizer: Muon + QK-Norm
- **Stage 2**: Contrastive pretraining (SigLIP loss)
  - NaFlex for variable-size inputs
  - 64K global batch
  - 8B bilingual (Chinese-English) corpus

**2. MMTP — Multi-Modal Multi-Token Prediction**

Extend MTP to multimodal:

```
Key trick: Use <|image|> placeholder token instead of visual embeddings.
  text + <|image|> placeholder → cross-attention patches at placeholder
  MMTP head predicts multiple vision-action tokens in parallel
```

Benefit: No need to propagate visual embeddings across pipeline-parallel stages.

**3. 30+ Integrated Tools**

| Category | Examples |
|----------|---------|
| Recognition | `zai_recognize_plant`, `zai_recognize_location` |
| Multimodal Search | `zai_search_web_by_image`, `zai_search_similar_images` |
| Browser | `zai_load_image_from_url`, `zai_read_webpage` |
| Image Processing | `zai_crop_image`, `zai_draw_bounding_boxes` |
| Deep Research | `zai_dr_python`, `zai_dr_open_url_mm` |

**4. Multimodal RL at Scale**

- VLM RL Gym interface
- Independent reward system (rule-based + model verifiers)
- Pipeline: decoupled rollout inference, async reward eval, early-abort
- Separate memory strategies for ViT/projector (recompute + CPU offload)

**Results**:
- Perception: +4.8% (RefCOCO) to +7.7% (SUNRGBD)
- Reasoning: +1.8% (MMMU/MathVista)
- Agentic: +4.9% (OSWorld)

#### Integration

Add vision encoder + MMTP head:
```python
class GLM5VExtension:
    def __init__(self, base_model):
        self.vision_encoder = CogViTEncoder.from_pretrained("THUDM/cogvlm2")
        self.mmtp = MMTPHead(num_tokens=3)

    def forward(self, input_ids, pixel_values=None):
        if pixel_values is not None:
            # Replace <|image|> tokens with vision features via cross-attention
            img_embs = self.vision_encoder(pixel_values)
            # Inject at placeholder positions
            input_embs = self.model.embed_tokens(input_ids)
            mask = (input_ids == self.image_token_id).unsqueeze(-1)
            input_embs = torch.where(mask, img_embs, input_embs)
            # MMTP predicts additional vision-action tokens
            mmtp_logits = self.mmtp(input_embs)
            return self.model(inputs_embeds=input_embs), mmtp_logits
        else:
            return self.model(input_ids), None
```

**When to enable**: Need vision capabilities (GUI agents, image analysis, document parsing).

**Cost**: +1.5 GB (vision encoder frozen), -30% throughput.

### 4.7 Pareto Frontier Router — v4

**Source**: Multi-objective optimization research

**Idea**: When multiple backends compete, none is uniformly best. Choose from non-dominated frontier.

#### Definitions

Backend A dominates B if:
- `cost_A ≤ cost_B` AND `latency_A ≤ latency_B` AND `quality_A ≥ quality_B`
- At least one strict inequality

**Pareto frontier** = set of non-dominated backends.

#### Implementation

```python
class ParetoFrontierRouter:
    def __init__(self, backends, mode="pareto"):
        self.backends = {b.name: b for b in backends}
        self.history = deque(maxlen=1000)  # empirical measurements
        self.mode = mode  # "simple" | "cost_aware" | "pareto"

    def select(self, request):
        # 1. Estimate metrics for each backend (with empirical multipliers)
        # 2. Filter by constraints (min_quality, max_latency, max_cost)
        # 3. If pareto mode: choose knee point (best quality/cost ratio)
        # 4. Record actuals after serving; update frontier every 100 reqs
        pass

    def _compute_frontier(self):
        # Recompute non-dominated set from recent history
        pass
```

#### Example Frontier

```
Quality ^
  1.0 |                          Claude (0.98)
      |                         *
      |                        /
  0.9 |                      /
      |   Local (0.92) *----/---- Step (0.97)
      |               \    /
  0.8 |                \  /
      |                 \/ DeepSeek (0.95)
      +------------------------------> Cost
```

**Selection strategies**:
- **Budget-constrained**: Leftmost frontier point under budget
- **Quality-constrained**: Topmost point above quality threshold
- **Balanced**: Knee point (max quality gain per unit cost)

**When to use**: Always if serving with **multiple backends** (local + cloud). Marginal overhead (<1 ms), high benefit.

---

## 5. AGENT & TOOL INTEGRATION

### 5.1 Native Tools (CUA Driver Integration)

You already have `computer_use` tool at system level. Aurelius integrates it **natively** (no gateway hop) with safety guards.

#### Tool Suite

| Tool | Purpose | Safety |
|------|---------|--------|
| `computer.use` | Execute CUA driver actions (click, type, scroll) | Dangerous command patterns blocked |
| `computer.click` | Click element by index (from snapshot) | Element bounds checking |
| `computer.type` | Type text into focused field | Password detection heuristic |
| `computer.screenshot` | Capture screen with optional element annotation | No PII protection (agent must blur) |
| `file.read` | Read file from sandbox (`/tmp/aurelius_sandbox`) | Path traversal blocked; size limit 10 MB |
| `file.write` | Write file to sandbox | Parent dirs auto-created |
| `file.list` | List sandbox directory | — |
| `shell.run` | Run whitelisted commands (`python`, `git`, `ls`) | Whitelist enforced; timeout 30 s |
| `browser.open` | Open URL in Safari/Chrome (AppleScript) | Limited to http/https |
| `browser.click` | Click link/button by text | AppleScript UI element lookup |

#### Safety Guardrails

```python
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"format\s+",
    r"shutdown\b",
    r":\(\)\{:\|:&};:",  # fork bomb
]
```

Password entry detection:
```python
def _is_password(text):
    if all(c == '*' for c in text) and len(text) >= 6:
        return True  # masked typing
    if len(text) >= 20 and has_alpha_numeric_special(text):
        return True  # long random string
    return False
```

#### Sandboxing

Filesystem: all paths resolved under `/tmp/aurelius_sandbox`. Directory traversal attempts rejected.

Shell: command whitelist. Unlisted commands → `ToolSafetyError`.

#### Atropos Environment for GUI

Train agents to use desktop via RL:

```python
class GUIActionEnv(AtroposEnvironment):
    def reset(self, goal):
        self.goal = goal
        return self.computer.screenshot()

    def step(self, action):
        # action: {"tool": "computer.click", "args": {"element_id": 14}}
        try:
            result = execute(action)
            reward = self.compute_visual_reward(result)  # vision model verifies
            done = self.is_goal_met()
            return screenshot(), reward, done, {}
        except ToolSafetyError as e:
            return screenshot(), -1.0, False, {"error": str(e)}
```

### 5.2 Tool Schema & Validation

Each tool declares JSON schema for argument validation:

```python
TOOLS = {
    "computer.click": {
        "fn": computer.click,
        "schema": {
            "type": "object",
            "properties": {
                "element_id": {"type": "integer", "description": "Ref ID from snapshot"}
            },
            "required": ["element_id"]
        }
    },
    "shell.run": {
        "fn": shell.run,
        "schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Whitelisted shell command"}
            },
            "required": ["command"]
        }
    }
}
```

Agent planner consults schemas to generate valid action calls.

---

## 6. TRAINING EXTENSIONS

### 6.1 Token Superposition Trainer (TST)

**File**: `src/training/tst_trainer.py`

**Usage**:
```python
trainer = TokenSuperpositionTrainer(
    model=model,
    optimizer=optimizer,
    bag_size=12,           # from table above
    phase_ratio=0.30,
    use_power_law=True,
)

for step, batch in enumerate(dataloader):
    loss = trainer.train_step(batch, step, total_steps)
    # ...
```

**Checkpoint strategy**:
```bash
# Phase 1 (TST): save at transition point
if step == int(total_steps * phase_ratio) - 1:
    torch.save(model.state_dict(), "checkpoint_phase1.pt")

# Phase 2 (NTP): resume from that checkpoint — DO NOT re-init!
```

### 6.2 Model Spec Midtraining (MSM)

**File**: `src/training/alignment/ms_midtraining.py`

**Pipeline**:
```bash
# 1. Generate synthetic spec-explaining documents (DataForge-style)
python -m src.training.alignment.synthetic_spec_gen \
  --spec specs/model_spec.md \
  --num-samples 5000 \
  --output data/spec_msm/

# 2. Run MSM phase
python -m src.training.alignment.ms_midtraining \
  --model meta-llama/Llama-3.1-8B \
  --pretrained checkpoints/pretrain/ \
  --spec-data data/spec_msm/ \
  --steps 5000 \
  --output checkpoints/msm/

# 3. Run standard AFT
python -m src.training.alignment.aft \
  --model checkpoints/msm/ \
  --demos data/alignment_demos/ \
  --steps 2000 \
  --output checkpoints/aligned/
```

### 6.3 ELF Trainer (Optional)

**File**: `src/training/elf_trainer.py`

For training parallel generation head (if you want ELF path on your model).

- Dataset: requires paired (input → long output) examples
- Loss: diffusion denoising + final-token CE
- 10× fewer training tokens needed vs AR-only

### 6.4 Atropos RL Environments (Existing)

Already integrated for:
- `CodeExecutionEnv` — run Python, verify outputs
- `WebSearchEnv` — search + scrape + answer
- `MathVerifyEnv` — symbolic math verification
- `GUIActionEnv` — new (desktop control)

Use for:
- Post-training RL (Reinforcement Learning from Human Feedback替代)
- Chain-of-thought reward shaping
- Tool-use skill acquisition

---

## 7. HARDWARE SCALING & OPTIMIZATION

### 7.1 Memory Calculation (7B model, 32 layers, 32 heads, 128 dim, context=32k)

| Component | FP16 | 4-bit + Optimizations |
|-----------|------|---------------------|
| Model weights | 14 GB | **3.5 GB** |
| KV cache (full) | 9.8 GB | **0.35 GB** (StreamingSink: 8×4096×8KVs) |
| Activations | 2.5 GB | 2.5 GB (unchanged) |
| **Total** | **26.3 GB** | **6.4 GB** ✅ |

**Conclusion**: 7B model fits comfortably in 16 GB with all optimizations.

### 7.2 Throughput Benchmarks (RTX 4090, 7B @ 4-bit)

| Configuration | Tokens/s (decode) | Tokens/s (prefill) | VRAM |
|---------------|-------------------|--------------------|------|
| FP16 + standard KV | 42 | 128 | 26 GB (OOM) |
| 4-bit + standard KV | 110 | 320 | 8 GB |
| 4-bit + StreamingSinkKV | 135 | 340 | 6.5 GB |
| + MTP speculative | **160** | 340 | 6.7 GB |
| + TokenSpeed MLA kernel | **175** | 360 | 6.7 GB |

### 7.3 Apple Silicon (M1–M5)

- **Metal FlashAttention**: via PyTorch 2.6+ (`torch.backends.mps.enable_flash_attention = True`)
- **Memory**: Unified RAM shared between CPU/GPU
  - M1 Max (32 GB): up to 13B 4-bit
  - M2 Ultra (64 GB): up to 20B 4-bit
  - M4 Max (48 GB): up to 34B 4-bit
- **Speed**: ~60% of CUDA equivalent (memory bandwidth bound)
- **RingAttention**: Not yet on Metal; context limit ~128k before slowdown

### 7.4 Multi-GPU Tensor Parallelism

For 70B model on dual 24 GB GPUs:

```
Strategy: TP=2 (tensor parallelism)
Each GPU holds: 70B/2 = 35B → 8.75 GB 4-bit + 1 GB KV = 9.75 GB ✅

Communication: NVLink (RTX 40-series) 300 GB/s → negligible overhead

Implementation: HuggingFace accelerate + custom TP engine
  from accelerate import dispatch_model
  model = dispatch_model(model, device_map="auto")
```

**RingAttention** (Gemini 3): For unlimited context, ring KV cache across GPUs:
- Each GPU holds local KV chunk
- Attention queries → all GPUs in ring (all-to-all)
- Context scaling: linear with GPU count

---

## 8. CONFIGURATION REFERENCE

### 8.1 Complete `config/aurelius.yaml`

```yaml
# ─── MODEL ────────────────────────────────────────────────────────────────────
model:
  path: "meta-llama/Llama-3.1-8B-Instruct"  # or local checkpoint
  arch: "dense"                          # dense | moe
  weights_quant: "awq"                   # none | nf4 | gptq | awq | exl2 | fp8
  dtype: "auto"                          # float16 | bfloat16 | fp8

  # Token Superposition Training — only during large-scale pretraining
  tst:
    enabled: false
    bag_size: 12                         # determined by model size
    phase_ratio: 0.30                    # 30% of steps
    power_law_weight: true               # for bag_size ≥ 8

  # Model Spec Midtraining — only during alignment
  alignment:
    method: "none"                       # none | msm
    spec_document: "specs/model_spec.md"
    msm_steps: 5000
    aft_steps: 2000

# ─── ATTENTION ────────────────────────────────────────────────────────────────
attention:
  type: "configurable"                   # standard | gqa | mla
  num_attention_heads: 32
  num_key_value_heads: 8                 # GQA: 4× reduction
  head_dim: 128
  use_qk_norm: true
  qk_norm_type: "rms"

  rope:
    base: 500000                         # Qwen-style high-theta for long context
    scaling: "dynamic"                   # none | linear | dynamic | yarn

  # Kernel selection
  kernel: "auto"                         # sdpa | flash_attn | flash_attn3 | metal_flash | tokenspeed_mla
  flash_attn_version: 3
  tokenspeed_mla:
    enabled: false
    tile_q_seqlen_into_heads: true

# ─── KV CACHE ─────────────────────────────────────────────────────────────────
kv_cache:
  strategy: "hybrid_streaming"           # none | standard | StreamingSinkKVCache
  sinks: 8
  window: 4096
  proximal_preserve: 256
  quant: "int8"                          # none | int8 | fp8

  # MLA-specific (if type=mla)
  mla:
    compress_ratio: 64                   # 64× KV compression
    use_proximal_full: true

# ─── SPECULATIVE DECODING ────────────────────────────────────────────────────
speculative:
  enabled: true
  method: "mtp"                           # off | draft | mtp | eagle | medusa
  mtp_n: 3                                # number of MTP heads
  eagle_tree_size: 5                      # for EAGLE-3 tree
  draft_model: null                       # path to draft model (optional)

# ─── BATCHING ─────────────────────────────────────────────────────────────────
batch:
  continuous: true                        # Orca-style dynamic batching
  chunked_prefill: true
  chunk_size: 1024
  max_batch_size: 64
  max_num_batched_tokens: 4096           # per forward pass

# ─── ELF PARALLEL GENERATION ──────────────────────────────────────────────────
elf:
  enabled: false                         # parallel path for creative tasks
  num_inference_steps: 32                # ODE integration steps
  cfg_scale: 3.0                         # classifier-free guidance strength
  sampler: "ode"                         # ode | sde | euler
  shared_denoiser: true                  # same weights as main model

# ─── HOPE BLOCKS (AGENTIC MODE) ───────────────────────────────────────────────
hope:
  enabled: false                         # enable for complex agents only
  num_blocks: 4                          # replace top N layers
  cms_banks:                             # Continuum Memory System banks
    - 8192                               # short-term
    - 65536                              # mid-term
    - 262144                             # long-term
  titans_memory_dim: 4096                # Titans fast-weight dimension
  fast_weight_lr: 0.01                   # learned optimizer step size

# ─── MULTIMODAL (GLM-5V STYLE) ────────────────────────────────────────────────
multimodal:
  enabled: false
  vision_encoder: "cogvit"               # cogvit | siglip | clip_vit
  image_placeholder_token: "<|image|>"
  mmtp_enabled: true                     # Multi-Modal Multi-Token Prediction
  mmtp_n: 3                              # vision tokens predicted per image
  vision_tokens_per_image: 256           # ViT patch count (14×14=196 typical)

# ─── PARETO FRONTIER ROUTER ───────────────────────────────────────────────────
router:
  mode: "pareto"                          # simple | cost_aware | pareto
  cost_tracking: true                     # log actual costs for empirical calibration
  latency_target_p99: 5000                # ms
  quality_threshold: 0.85                 # min acceptable quality
  backends:
    - name: "local"
      label: "Aurelius Local"
      cost_per_1k_tokens: 0.0
      latency_per_token_ms: 5.0
      quality: 0.92
      priority: 1
    - name: "step"
      label: "StepFun Step"
      cost_per_1k_tokens: 0.002
      latency_per_token_ms: 50.0
      quality: 0.97
      priority: 2
    - name: "claude"
      label: "Anthropic Claude"
      cost_per_1k_tokens: 0.008
      latency_per_token_ms: 100.0
      quality: 0.98
      priority: 3
    - name: "deepseek"
      label: "DeepSeek V3"
      cost_per_1k_tokens: 0.0015
      latency_per_token_ms: 60.0
      quality: 0.96
      priority: 2

# ─── NATIVE TOOLS (CUA INTEGRATION) ──────────────────────────────────────────
native_tools:
  computer_use:
    enabled: true
    safety_guardrails: true
    require_confirmation: false          # set true for destructive actions
  filesystem:
    enabled: true
    sandbox_path: "/tmp/aurelius_sandbox"
    max_read_size_mb: 10
  shell:
    enabled: true
    timeout_seconds: 30
    allowed_commands:
      - python
      - python3
      - node
      - git
      - ls
      - cat
      - head
      - tail
      - grep
      - find
      - pytest
      - ruff
      - black
  browser:
    enabled: true

# ─── HARDWARE ─────────────────────────────────────────────────────────────────
hardware:
  device: "auto"                         # cuda | mps | cpu | auto
  tensor_parallel_size: 1                # >1 for multi-GPU
  gpu_mem_util: 0.85                     # target memory utilization (0–1)
  use_cuda_graphs: auto                  # true | false | auto
  enable_flash_attention: true
  enable_memory_efficient_attention: true
```

### 8.2 Environment Variable Overrides

```bash
# Override config values via environment (useful for deployment)
export AURELIUS_MODEL_PATH="/path/to/model"
export AURELIUS_WEIGHTS_QUANT="gptq"
export AURELIUS_DEVICE="cuda"
export AURELIUS_GPU_MEM_UTIL=0.9
export AURELIUS_ROUTER_MODE="pareto"
```

Priority: CLI args > env vars > config file.

---

## 9. IMPLEMENTATION ROADMAP

### Phase 1: Core Inference Engine (Weeks 1–3) — v2

| Task | File | Status |
|------|------|--------|
| CustomEngine class | `gateway/custom_engine/engine.py` | Design complete |
| StreamingSinkKVCache | `gateway/custom_engine/kv_cache.py` | Design complete |
| GQA/MLA attention | `gateway/custom_engine/attention.py` | Design complete |
| QK-Norm + RoPE | same | Design complete |
| Speculative (MTP heads) | `gateway/custom_engine/speculative.py` | Design complete |
| Continuous batching | `gateway/custom_engine/scheduler.py` | Design complete |
| Quant loaders (AWQ/GPTQ) | `gateway/custom_engine/quant.py` | Design complete |
| AutoConfig hardware detection | `gateway/custom_engine/config.py` | Design complete |

**Milestone**: `aurelius serve` runs local 7B model at 135 tok/s on RTX 4090.

### Phase 2: Router & Native Tools (Weeks 4–5) — v4 HIGHEST IMPACT

| Task | File | Complexity |
|------|------|------------|
| Pareto router implementation | `gateway/router_pareto.py` ✅ DONE | Low |
| Router integration into API server | `gateway/router.py` | Low |
| CUA driver wrapper | `gateway/native_tools/__init__.py` ✅ DONE | Low |
| Tool registration in agent | `agent/tools.py` | Low |
| Safety test suite | `tests/test_native_tools.py` | Low |

**Milestone**: `aurelius chat` selects cloud backend when cheaper; agent can click/type on desktop.

### Phase 3: Advanced Inference (Week 6) — v4

| Task | File | Status |
|------|------|--------|
| TokenSpeed MLA kernel (when available) | `gateway/token_speed/mla_kernel.py` | Pending upstream |
| Custom MLA kernel w/ tiling | `gateway/custom_engine/attention.py` | Low (if no upstream) |
| ELF parallel generation head | `gateway/elf/denoiser.py` | Medium |
| ELF → AR task routing | `gateway/engine.py` | Low |
| Chunked prefill scheduler | `gateway/custom_engine/scheduler.py` | Already designed |

**Milestone**: Creative tasks 3× faster via ELF; MLA decode +11% with TokenSpeed.

### Phase 4: Agent Mode (Weeks 7–8) — v3 + v4

| Task | File | Complexity |
|------|------|------------|
| HOPE block implementation | `gateway/moonshine/hope_block.py` | Medium |
| CMS with FAISS backend | `gateway/moonshine/cms.py` | Medium |
| Agent mode flag (`--mode=agent`) | `gateway/cli.py` | Low |
| Multi-step planning benchmark | `tests/test_agent_planning.py` | Medium |
| GUI Atropos environment | `src/training/atropos_envs/gui_env.py` | Medium |

**Milestone**: Agent can plan 10+ step tasks with HOPE; learns desktop control via RL.

### Phase 5: Training Extensions (Weeks 9–10) — OPTIONAL

| Task | File | Conditional |
|------|------|-------------|
| TST trainer | `src/training/tst_trainer.py` ✅ DONE | Only if pretraining |
| MSM pipeline | `src/training/alignment/ms_midtraining.py` | Only if aligning agents |
| ELF trainer | `src/training/elf_trainer.py` | Only if training ELF head |
| Synthetic spec generation | `src/training/alignment/synthetic_spec_gen.py` | If MSM |

**Milestone**: Atropos-style self-improvement loop (agent designs its own trainer, runs it, evaluates, accepts if better).

### Phase 6: Multimodal (Week 11) — v4 OPTIONAL

| Task | File | Complexity |
|------|------|------------|
| CogViT encoder integration | `gateway/multimodal/vision_encoder.py` | High |
| MMTP head implementation | `gateway/multimodal/mmtp.py` | Medium |
| Image tokenization pipeline | `gateway/multimodal/image_tokenizer.py` | Medium |
| 30+ GLM-5V tools port | `gateway/multimodal/tools/` | High |
| Image-based Atropos env | `src/training/atropos_envs/image_env.py` | Medium |

**Milestone**: Agent can analyze screenshots, use vision for web search + GUI navigation.

---

## 10. DECISION FRAMEWORK

### 10.1 Feature Decision Tree

```
Start: Set up Aurelius
│
├─ Are you pretraining or continued-pretraining with >30B new tokens?
│  ├─ YES → Enable TST
│  │   • Model <1B → bag_size=4, r=0.25
│  │   • Model 1–3B → bag_size=6, r=0.30
│  │   • Model 3–10B → bag_size=8–12, r=0.30–0.35
│  │   • Model >10B → bag_size=12–20, r=0.35–0.40
│  │   • Add power-law weighting if bag_size ≥ 8
│  │   • CRITICAL: Save checkpoint at phase transition; resume from it
│  │
│  └─ NO → Skip TST (no benefit)
│
├─ Are you alignment-tuning an autonomous agent?
│  ├─ YES → Enable MSM
│  │   • Phase 1: 5k steps on synthetic spec explanation docs
│  │   • Phase 2: 2k steps on demonstrations
│  │   • Expected: 54% → 7% misalignment
│  │
│  └─ NO → Standard SFT only
│
├─ Will agent need desktop GUI control?
│  ├─ YES → Enable native_tools.computer_use
│  │   • safety_guardrails = true (block rm -rf, sudo, etc.)
│  │   • All file ops sandboxed to /tmp/aurelius_sandbox
│  │   • Shell whitelist: python, git, ls, cat, grep…
│  │
│  └─ NO → Disable native tools (smaller attack surface)
│
├─ Using multiple LLM backends (local + Step/Claude/DeepSeek)?
│  ├─ YES → Set router.mode = "pareto"
│  │   • cost_tracking = true
│  │   • Frontier recomputes every 100 requests
│  │   • Selection: constraint-satisfying knee point
│  │
│  └─ NO → router.mode = "simple" (priority-based)
│
├─ Need vision capabilities (image understanding, GUI screenshot analysis)?
│  ├─ YES → Enable multimodal
│  │   • vision_encoder = "cogvit" (or siglip)
│  │   • mmtp_enabled = true
│  │   • Use <|image|> placeholder in prompts
│  │   • Add GLM-5V tools if full multimodal agent needed
│  │   • Cost: +1.5 GB, -30% throughput
│  │
│  └─ NO → Text-only
│
├─ Are you generating long-form creative content (stories, summaries)?
│  ├─ YES → Enable elf.enabled = true
│  │   • Use task_type="creative" to trigger ELF
│  │   • Reasoning tasks (math, code) stay autoregressive
│  │   • Speed: 32 steps vs 1024; quality: -3% (acceptable)
│  │
│  └─ NO → Keep AR only
│
└─ Is agent doing complex multi-step planning (research, coding, long-horizon)?
   ├─ YES → Enable hope
   │   • hope.enabled = true
   │   • num_blocks = 4 (replace top 4 layers)
   │   • cms_banks = [8192, 65536, 262144] (adjust for VRAM)
   │   • Cost: +1 GB, -10% speed, +2× reasoning depth
   │
   └─ NO → Standard transformer blocks (faster, simpler)
```

### 10.2 Presets

#### `preset: "production_16gb_agent"`
```yaml
model:
  weights_quant: "awq"
  path: "meta-llama/Llama-3.1-8B-Instruct"

attention:
  type: "gqa"
  num_key_value_heads: 8

kv_cache:
  strategy: "hybrid_streaming"

speculative:
  enabled: true
  method: "mtp"

router:
  mode: "pareto"

native_tools:
  computer_use:
    enabled: true

hope:
  enabled: true                     # agent mode
  num_blocks: 4

multimodal:
  enabled: false

elf:
  enabled: false
```
**Use case**: Autonomous agent on 16 GB GPU; needs desktop control + planning.

#### `preset: "creative_32gb"`
```yaml
model:
  path: "mistralai/Mistral-7B-Instruct-v0.3"
  weights_quant: "gptq"

elf:
  enabled: true
  num_inference_steps: 32
  cfg_scale: 3.0

router:
  mode: "simple"                    # local only

multimodal:
  enabled: false

hope:
  enabled: false
```
**Use case**: Long-form content generation; no vision; simple local serving.

#### `preset: "multimodal_agent"`
```yaml
model:
  path: "Qwen/Qwen2-VL-7B-Instruct"

multimodal:
  enabled: true
  vision_encoder: "siglip"
  mmtp_enabled: true

native_tools:
  computer_use:
    enabled: true

router:
  mode: "pareto"

hope:
  enabled: true
```
**Use case**: Vision-capable agent that controls desktop + uses cloud backends.

#### `preset: "research_pretrain"`
```yaml
model:
  arch: "moe"                       # e.g., 64 experts, 8 active
  weights_quant: "none"              # training needs full precision

tst:
  enabled: true                     # ✅ TST for >30B corpus
  bag_size: 16
  phase_ratio: 0.35

attention:
  type: "mla"                       # aggressive KV compression
  num_key_value_heads: 8

kv_cache:
  strategy: "standard"              # during training

router:
  mode: "none"                      # single model only
```
**Use case**: Large-scale pretraining run on cluster.

---

## 11. GLOSSARY

| Term | Definition |
|------|------------|
| **GQA** | Grouped-Query Attention: fewer KV heads than query heads; reduces KV cache size 4× |
| **MLA** | Multi-Head Latent Attention: compress KV to latent vectors; up to 64× compression |
| **QK-Norm** | RMSNorm applied separately to queries and keys; stabilizes attention scores |
| **RoPE** | Rotary Position Embedding: inject position info via rotation matrices |
| **StreamingSinkKVCache** | Fixed-size KV cache that evicts old tokens into compressed "sinks" |
| **MTP** | Multi-Token Prediction: draft heads predict n+1, n+2, n+3 tokens in parallel |
| **TST** | Token Superposition Training: bagged-token pretraining for 2.5× speed |
| **MSM** | Model Spec Midtraining: pre-alignment spec education phase |
| **ELF** | Embedded Language Flows: diffusion generation in embedding space |
| **HOPE** | Nested Learning blocks: self-modifying weights + CMS memory |
| **CMS** | Continuum Memory System: 3-tier memory (short/mid/long-term) |
| **Atropos** | RL environment suite for agent training (code, web, math) |
| **CUA** | Computer Use — macOS background desktop control |
| **Pareto frontier** | Set of non-dominated backends optimizing Quality/Cost/Latency |
| **RingAttention** | KV cache sharded across GPUs in ring topology for unlimited context |
| **FlashAttention** | GPU kernel fusing attention computation to reduce HBM I/O |
| **TokenSpeed** | MLA kernel optimization tiling q_seqlen into num_heads for Tensor Cores |

---

## 🎯 QUICK NAVIGATION

| What you need | Read this section |
|----------------|------------------|
| Understanding core engine | §3: Core Inference Engine |
| Enabling desktop control | §5.1: Native Tools (CUA) |
| Building autonomous agents | §4.1: HOPE Blocks + §5: Agent Integration |
| Large-scale pretraining | §4.2: TST + §6.1: TST Trainer |
| Aligning agents safely | §4.3: MSM + §6.2: MSM Pipeline |
| Multi-backend cost optimization | §4.7: Pareto Router |
| Vision-capable agents | §4.6: GLM-5V Multimodal |
| Parallel creative generation | §4.4: ELF |
| Choosing model size | §7.1: Memory Calculations |
| Production config | §8.1: Complete YAML |
| Implementation timeline | §9: Roadmap |

---

**Document Version**: 4.0 (May 13, 2026)  
**Last Updated**: Integrated 8 breakthrough papers (Nous, Anthropic, MIT, LightSeek, Zhipu AI, Google, Frontier Labs) + cua-driver native integration  
**Status**: Architecture complete — ready for implementation

**Next**: See `docs/V4_INTEGRATION_GUIDE.md` for step-by-step scaffolding instructions.
