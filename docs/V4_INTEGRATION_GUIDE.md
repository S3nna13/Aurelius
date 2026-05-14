# Aurelius v4.0 — Quick-Start Integration Guide
## (All 8 Breakthrough Papers Integrated — May 13, 2026)

This guide shows **exactly how to enable** each v4 feature in your Aurelius deployment.

---

## 🚀 QUICK START: MINIMAL v4 (5 minutes)

Just add to `config/aurelius.yaml`:

```yaml
router:
  mode: "pareto"                    # ← Switch from "simple" to Pareto frontier
  cost_tracking: true               # ← Enable empirical calibration

native_tools:
  computer_use:
    enabled: true                   # ← Native CUA driver (no gateway)
    safety_guardrails: true

elf:
  enabled: false                    # ← Off by default; enable per-request
```

Restart server. Done. Pareto routing + native computer control active.

---

## 📦 FEATURE-BY-FEATURE ENABLEMENT

### 1. Pareto Frontier Router

**Files**: `gateway/router_pareto.py`, `gateway/router.py` update

**Step 1**: Add import to `router.py`:
```python
from gateway.router_pareto import ParetoFrontierRouter, BackendConfig, default_backends
```

**Step 2**: Replace router initialization:
```python
# Old:
# router = SimpleRouter()

# New:
backend_configs = default_backends()
# Optionally customize:
backend_configs[0].latency_per_token_ms = 4.0  # calibrated local backend
router = ParetoFrontierRouter(backends=backend_configs, mode="pareto")
```

**Step 3**: Update request handler to record outcomes:
```python
# In api_server.py → complete_request()
decision = router.select(request)
response = call_backend(decision.backend, request)

# After response fully sent:
router.record(
    backend_name=decision.backend.name,
    cost=calculate_actual_cost(response),      # tokens × price
    latency_ms=response.latency_ms,
    quality=response.quality_score,             # from evaluator or heuristic
)
```

**Step 4**: Monitor frontier updates:
```bash
# Add endpoint for debugging
curl http://localhost:8080/v1/router/frontier
# Returns: {"frontier": [{"name": "local", "quality": 0.92, ...}, ...]}
```

**What you get**: Dynamic selection across local/Step/Claude/DeepSeek based on real-world Quality/Cost/Latency measurements. Frontier updates every 100 requests.

---

### 2. Native Computer Use (CUA Driver)

**Files**: `gateway/native_tools/__init__.py`, `agent/tools.py` update

**Step 1**: Add to agent's tool registry (`agent/tools.py`):
```python
from gateway.native_tools import get_native_tools, TOOL_ACTION_MAP

native = get_native_tools()

# Merge into existing TOOLS dict
TOOLS.update({
    "computer.use": native["computer"],
    "computer.click": native["computer"],
    "computer.type": native["computer"],
    "computer.scroll": native["computer"],
    "computer.screenshot": native["computer"],
    "file.read": native["filesystem"],
    "file.write": native["filesystem"],
    "file.list": native["filesystem"],
    "shell.run": native["shell"],
    "browser.open": native["browser"],
})
```

**Step 2**: Update agent planner to include these tools:
```python
# agent/planner.py
available_tools = list(TOOLS.keys())
# Now includes: computer.use, file.read, shell.run, browser.open, ...
```

**Step 3**: Safety test:
```python
# tests/test_native_tools.py
def test_safety_block():
    computer = NativeComputerUseTool(safety_guardrails=True)
    with pytest.raises(ToolSafetyError):
        computer.execute("rm -rf /")
```

**Result**: Agent can now control desktop, read/write files (in sandbox), run shell commands, open browser — all with safety guards.

---

### 3. ELF Parallel Generation (Optional)

**Files**: `gateway/elf/` (already created), `gateway/custom_engine/engine.py` update

**Step 1**: Add ELF generation mode flag to config:
```yaml
elf:
  enabled: false        # default off
  num_inference_steps: 32
  cfg_scale: 3.0
```

**Step 2**: Add ELF head to model (if you want parallel path):
```python
# In model loading (engine.py)
if config.elf.enabled:
    from gateway.elf.denoiser import ElfDenoiserDecoder
    model.elf_head = ElfDenoiserDecoder(model.config.hidden_size, model.config.vocab_size)
else:
    model.elf_head = None
```

**Step 3**: Route long-form tasks to ELF:
```python
def generate(self, prompt: str, max_tokens: int, task_type: str = "chat"):
    if self.config.elf.enabled and task_type in ("summarize", "expand", "story"):
        return self._generate_elf(prompt, max_tokens)
    else:
        return self._generate_autoregressive(prompt, max_tokens)

def _generate_elf(self, prompt: str, max_tokens: int):
    # Encode prompt to embedding
    prompt_emb = self.model.embed(prompt)

    # Start from noise
    z = torch.randn(1, max_tokens, self.model.hidden_size)

    # ODE integration
    for t in torch.linspace(0, 1, self.config.elf.num_inference_steps):
        v = self.model.elf_head(z, t)
        z = z + (1 / self.config.elf.num_inference_steps) * v

    # Final discretization
    logits = self.model.elf_head.unembed(z)
    tokens = torch.argmax(logits, dim=-1)
    return tokens
```

**When to enable**: For bulk creative/text- expansion tasks. **Disable** for reasoning/tool-use (CoT/latent better).

---

### 4. Model Spec Midtraining (Alignment)

**Files**: `src/training/alignment/ms_midtraining.py`

**Step 1**: Create your Model Spec document (`specs/model_spec.md`):
```markdown
# Aurelius Model Spec — v1.0

## Core Principles
1. **Helpfulness**: Assist users without causing harm
2. **Honesty**: Do not generate false or misleading information
3. **Safety**: Refuse requests that could cause physical, emotional, or financial harm
4. **Autonomy**: Respect user agency; suggest, don't coerce

## Prohibited Behaviors
- Giving instructions for illegal activities
- Generating hate speech, harassment, or explicit content
- Impersonating individuals without disclosure
- Sharing private or sensitive information

## Conflict Resolution
When principles conflict, prioritize: Safety > Honesty > Helpfulness > Autonomy
```

**Step 2**: Run MSM training (before AFT):
```bash
# Generate synthetic spec-explaining documents
python -m src.training.alignment.synthetic_spec_gen \
  --spec-file specs/model_spec.md \
  --num-samples 5000 \
  --output data/spec_midtraining/

# Train MSM phase
python -m src.training.alignment.ms_midtraining \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --pretrained-checkpoint checkpoints/pretrain/ \
  --spec-dataset data/spec_midtraining/ \
  --msm-steps 5000 \
  --output checkpoints/msm/

# Then standard AFT
python -m src.training.alignment.aft \
  --model checkpoints/msm/ \
  --demonstrations data/alignment_demos/ \
  --steps 2000 \
  --output checkpoints/aligned/
```

**Result**: Final aligned model generalizes from demonstrations according to spec principles, not surface patterns.

**Only if**: You're doing agent alignment (expecting autonomous operation).

---

### 5. Token Superposition Training (Pretraining)

**Files**: `src/training/tst_trainer.py` (already created)

**Step 1**: Choose bag size for your model:
```python
MODEL_BAG_SIZES = {
    "270M": 6,
    "600M": 6,
    "3B": 8,
    "7B": 12,
    "13B": 16,
    "70B": 20,
}
```

**Step 2**: Modify your training script:
```python
from src.training.tst_trainer import TokenSuperpositionTrainer

trainer = TokenSuperpositionTrainer(
    model=model,
    optimizer=optimizer,
    bag_size=MODEL_BAG_SIZES["7B"],   # ← adjust
    phase_ratio=0.30,                  # 30% of steps
    use_power_law=True,
)

for step, batch in enumerate(dataloader):
    loss = trainer.train_step(batch, step, total_steps=125_000)
    # ...

# SAVE CHECKPOINT AT END OF PHASE 1
if step == int(total_steps * 0.30) - 1:
    torch.save(model.state_dict(), "checkpoint_tst_phase1.pt")
```

**Step 3**: Resume from Phase 1 checkpoint for Phase 2:
```bash
# Phase 1 (TST):
python train.py --checkpoint scratch --tst-enabled --steps 37500

# Phase 2 (standard NTP):
python train.py --checkpoint checkpoint_tst_phase1.pt --tst-disabled --steps 87500
```

**WARNING**: Never reinitialize from random weights at Phase 2 start — you **must** continue from Phase 1 checkpoint or gains are lost.

**When to use**: Only during **large-scale pretraining** (>30B tokens). Skip for fine-tuning.

---

### 6. HOPE Blocks (Agent Mode)

**Files**: `gateway/moonshine/` (already from v3), `agent/hope_agent.py`

**Step 1**: Enable in config:
```yaml
hope:
  enabled: true
  num_blocks: 4
  cms_banks: [8192, 65536, 262144]
```

**Step 2**: Swap top layers in model:
```python
# engine.py or model loading
from gateway.moonshine.hope_block import HOPEBlock

def replace_top_layers_with_hope(model, num_hope=4):
    """Replace last N layers with HOPE blocks."""
    original_layers = model.layers[-num_hope:]
    hope_layers = nn.Sequential(*[
        HOPEBlock(model.config.hidden_size)
        for _ in range(num_hope)
    ])
    model.layers[-num_hope:] = hope_layers
    return model
```

**Step 3**: Agent uses HOPE for planning:
```python
class HopeAugmentedAgent:
    def __init__(self, model):
        self.model = replace_top_layers_with_hope(model, num_hope=4)
        self.cms = self.model.layers[-1].cms  # access last block's CMS

    def plan(self, goal: str):
        # HOPE blocks automatically:
        # 1. Retrieve relevant past plans from CMS
        # 2. Modulate computation via fast weights
        # 3. Write new plan summary to CMS
        return self.model.plan(goal)
```

**When to use**: For **complex multi-step agents** (coding, research, long-horizon tasks). Disable for simple chat (overhead not worth it).

---

### 7. GLM-5V Multimodal (Vision)

**Files**: `gateway/multimodal/`

**Step 1**: Enable multimodal in config:
```yaml
multimodal:
  enabled: true
  vision_encoder: "cogvit"          # or "siglip"
  image_placeholder_token: "<|image|>"
  mmtp_enabled: true
  mmtp_n: 3
```

**Step 2**: Load CogViT encoder (download from HuggingFace):
```python
from transformers import AutoModel, AutoProcessor

vision_processor = AutoProcessor.from_pretrained("THUDM/cogvlm2-llama3-chat-19B")
vision_encoder = AutoModel.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", torch_dtype="auto")
# Freeze vision encoder (or LoRA fine-tune later)
for p in vision_encoder.parameters():
    p.requires_grad = False
```

**Step 3**: Handle image tokens in forward pass:
```python
def forward_multimodal(self, input_ids, pixel_values=None):
    text_embeds = self.model.embed_tokens(input_ids)

    if pixel_values is not None:
        # Replace <|image|> tokens with vision embeddings
        image_features = self.vision_encoder(pixel_values)  # [batch, seq_img, hidden]
        # Find indices of <|image|> token in input_ids
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        mask = (input_ids == image_token_id).unsqueeze(-1)

        # Broadcast vision features to token positions
        text_embeds = torch.where(mask, image_features, text_embeds)

    # Continue with standard forward
    return self.model(inputs_embeds=text_embeds)
```

**Tool integration**: GLM-5V ships 30+ tools. You can adopt their tool registry:
```python
# gateway/multimodal/tools/__init__.py
GLM5V_TOOLS = {
    "zai_recognize_plant": recognize_plant,
    "zai_search_web_by_image": search_by_image,
    "zai_crop_image": crop_image,
    # ... 30+ more
}
```

**When to use**: If agent needs **vision** (GUI automation, image analysis, document parsing). Increases model size by ~0.5–1 GB (vision encoder).

---

### 8. TokenSpeed MLA Kernel (When Available)

**Files**: TBD — external to Aurelius (LightSeek open-sources)

**Step 1**: Monitor TokenSpeed GitHub for release:
```bash
git clone https://github.com/lightseekorg/tokenspeed
cd tokenspeed
pip install -e .
```

**Step 2**: Replace your MLA kernel call:
```python
# Old: custom MLA GEMM
k = self.compress_kv(key_states)   # [batch, seq, r, num_heads]

# New: TokenSpeed MLA
from tokenspeed.mla import mla_decode_kernel
k = mla_decode_kernel(key_states, compress_ratio=64, tile_q_into_heads=True)
```

**Step 3**: Benchmark validation:
```bash
# Measure decode latency before/after
python benchmark.py --model llama-3.1-8b --kernel flash_attn
python benchmark.py --model llama-3.1-8b --kernel tokenspeed_mla
# Expect: 9–11% improvement
```

**If not available**: Implement their tiling optimization yourself:
```python
def tokenspeed_mla_tiling(q, k, v):
    """
    Replicate TokenSpeed's q_seqlen → num_heads tiling.
    q shape: [batch, seq_len=1, num_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = q.shape
    # Fold seq_len into num_heads for better GEMM utilization
    q_tiled = q.view(batch, num_heads * seq_len, head_dim)  # [b, h*1, d]
    # Repeat for k, v as needed...
    return torch.matmul(q_tiled, k_tiled.transpose(-2, -1))
```

---

## 🎯 ENABLE ALL v4 FEATURES — FULL CONFIG

```yaml
# config/aurelius.yaml — v4 production

model:
  path: "meta-llama/Llama-3.1-8B-Instruct"
  weights_quant: "awq"

  # Only if doing large-scale pretraining
  tst:
    enabled: false
    bag_size: 12
    phase_ratio: 0.30

  # Only if doing agent alignment
  alignment:
    method: "none"  # "msm" to enable
    spec_document: "specs/model_spec.md"
    msm_steps: 5000

attention:
  type: "mla"
  num_key_value_heads: 8
  use_qk_norm: true
  kernel: "tokenspeed_mla"  # ← Use TokenSpeed if available
  tokenspeed_mla:
    enabled: false           # set true when kernel installed

kv_cache:
  strategy: "mla_streaming"
  sinks: 8
  window: 4096
  quant: "int8"

speculative:
  enabled: true
  method: "mtp"
  mtp_n: 3

# Enable parallel creative generation path (optional)
elf:
  enabled: false
  num_inference_steps: 32
  cfg_scale: 3.0

# Enable agent self-modification (only for complex agents)
hope:
  enabled: false
  num_blocks: 4
  cms_banks: [8192, 65536, 262144]

# Enable vision if needed
multimodal:
  enabled: false
  vision_encoder: "cogvit"
  mmtp_enabled: true

# Pareto frontier routing (always beneficial if multi-backend)
router:
  mode: "pareto"
  cost_tracking: true
  quality_threshold: 0.85

# Native tools including CUA
native_tools:
  computer_use:
    enabled: true
    safety_guardrails: true
  filesystem:
    enabled: true
    sandbox_path: "/tmp/aurelius_sandbox"
  shell:
    enabled: true
    timeout_seconds: 30
```

---

## 🔄 HOW FEATURES INTERACT

```python
# Request flow through v4 stack:

Request → ParetoRouter → select backend
                                      ↓
                      ┌─────────────────┴─────────────────┐
                      │                                   │
                Local backend                     External API (Step/Claude)
                      │                                   │
        ┌─────────────┴─────────────┐                   │
        │                           │                   │
   Engine (v2 core)           HOPE blocks?          Remote inference
        │                           │                   │
   Attention (GQA/MLA)       CMS retrieves         Return result
   KV cache (Streaming)      past memories
   Speculative (MTP)         Fast weights modify
   TokenSpeed MLA kernel      computation per-token
        │
   Generate tokens
        │
   Elf head? → parallel generation → concat with AR
        │
   Output
```

**Interactions**:
- Pareto router runs **before** any inference — selects which stack to use
- HOPE blocks only activate if `hope.enabled=true` (agent mode)
- ELF path is **alternative** to autoregressive — not both simultaneously
- TokenSpeed MLA kernel is **drop-in replacement** for standard MLA GEMMs
- TST is **training-only** — has no inference component
- MSM is **pretraining-only** — affects model weights before fine-tuning
- Multimodal (GLM-5V) extends tokenization pipeline — adds vision embeddings at `<|image|>`
- Native tools are **agent-facing** — they don't change model inference

---

## 🧪 VALIDATION CHECKLIST

Run this after implementation:

```bash
# 1. Pareto router
pytest tests/test_pareto_router.py -v
# Should pass: frontier computation, constraint satisfaction, knee-point selection

# 2. Native tools safety
pytest tests/test_native_tools.py::test_dangerous_commands_blocked -v
pytest tests/test_native_tools.py::test_sandbox_escape_prevented -v

# 3. ELF generation quality (if enabled)
pytest tests/test_elf.py::test_parallel_generation_quality -v
# Compare ELF vs AR perplexity on validation set

# 4. HOPE reasoning depth (if enabled)
pytest tests/test_hope.py::test_reasoning_chain_length -v
# HOPE should enable 2× longer reasoning chains

# 5. TST training speed (if training)
python -m src.training.tst_trainer.benchmark
# Should show 2.5× steps-to-target reduction

# 6. MSM alignment (if doing alignment)
python -m src.training.alignment.ms_midtraining.eval \
  --model checkpoints/aligned/ \
  --benchmark agentic_misalignment
# Should show <10% misalignment (vs 54% baseline)

# 7. End-to-end agent with all v4 features
pytest tests/test_integration/test_agent_v4.py -v
# Agent should: use Pareto router, control desktop via CUA, plan with HOPE,
# and maintain quality above threshold across 100 tasks
```

---

## 📊 PERFORMANCE EXPECTATIONS (7B model, RTX 4090)

| Feature | VRAM Impact | Tokens/s (decode) | Quality Impact | When to Enable |
|---------|-------------|-------------------|----------------|----------------|
| **Pareto router** | +10 MB | negligible | neutral | Always if multi-backend |
| **CUA native tools** | 0 MB | negligible | neutral | If desktop control needed |
| **ELF generation** | +200 MB | 0.6× (but parallel) | -3% (creative OK) | Creative tasks only |
| **TST** | 0 MB | N/A (training) | +0% | Large pretraining only |
| **MSM** | 0 MB | N/A (training) | +5% quality | Agent alignment only |
| **HOPE blocks** | +1.0 GB | 0.85× | +10% reasoning depth | Complex agent mode |
| **TokenSpeed MLA** | 0 MB | 1.09× | neutral | If using MLA + kernel available |
| **GLM-5V multimodal** | +1.5 GB (vision encoder) | 0.7× | +0% vision tasks | Vision required |

**Combined**: Enable Pareto + CUA + (HOPE OR ELF) + TokenSpeed MLA → net gain still positive.

---

## 🚦 TROUBLESHOOTING

### "ImportError: No module named 'computer_use'"
**Fix**: Install cua-driver:
```bash
# If using Hermes agent:
hermes tools install computer_use
# Or standalone:
pip install computer_use  # package name TBD — check Hermes docs
```

### "TST loss doesn't recover in Phase 2"
**Fix**: Checkpoints! Must resume from **end-of-Phase-1 checkpoint**. Never re-randomize.

### "ELF generation quality poor on reasoning"
**Fix**: ELF is for parallel creative generation. Use `task_type="creative"` to trigger ELF; for reasoning tasks it falls back to autoregressive automatically.

### "Pareto router always picks local despite higher-quality cloud"
**Fix**: Your empirical multipliers haven't converged yet. Send more traffic (100+ requests) to build history. Or seed with priors:
```python
router.record("claude", cost=0.008, latency_ms=120, quality=0.98)
router.record("local", cost=0.0, latency_ms=5, quality=0.92)
```

### "HOPE blocks crash with OOM"
**Fix**: CMS memory banks too large. Reduce bank sizes:
```yaml
hope:
  cms_banks: [4096, 32768, 131072]  # smaller
```

---

## 🎓 WHAT TO MEMORIZE

```markdown
Aurelius v4.0 (May 13 2026) — Implementation essentials:

1. Pareto Router:
   - Build empirical frontier from history (cost, latency, quality)
   - Select knee-point or constraint-satisfying point
   - Update every 100 requests
   - File: gateway/router_pareto.py

2. CUA Native Tools:
   - Wrap cua-driver directly (no gateway hop)
   - Safety guards block dangerous commands
   - Sandboxed filesystem/shell at /tmp/aurelius_sandbox
   - File: gateway/native_tools/__init__.py

3. ELF Parallel Generation:
   - Diffusion in embedding space, discretize only at final step
   - 32 steps vs 1024 for prior DLMs; 10× data-efficient
   - Use for creative tasks only; CoT/latent for reasoning
   - File: gateway/elf/

4. Model Spec Midtraining (MSM):
   - Phase 1: synthetic spec document training (5k steps)
   - Phase 2: standard AFT on demonstrations (2k steps)
   - Reduces misalignment 54% → 7%
   - File: src/training/alignment/ms_midtraining.py

5. Token Superposition Training (TST):
   - Bag size s=6–16 depending on model size
   - Phase ratio r=0.3–0.4
   - 2.5× pretraining speedup; zero inference overhead
   - MUST resume from Phase 1 checkpoint — never reinit
   - File: src/training/tst_trainer.py

6. HOPE Blocks (already in v3):
   - Replace top N layers with self-modifying blocks
   - CMS provides long-term memory (3 banks)
   - +0.8 GB mem, -10% speed, +2× reasoning depth
   - Enable only for complex agents

7. TokenSpeed MLA Kernel:
   - Tile q_seqlen into num_heads for better Tensor Core utilization
   - 9–11% decode speedup over baseline MLA
   - Adopted by vLLM PR #41778
   - Watch LightSeek GitHub for release

8. GLM-5V Multimodal:
   - Two-stage CogViT encoder (MIM + contrastive)
   - MMTP: predict multiple vision tokens in parallel
   - 30+ integrated tools (search, GUI, image processing)
   - +5% perception gains with RL fine-tuning
```

---

**You now have the complete v4.0 implementation guide**. All 8 breakthrough papers integrated with:
- Code snippets
- Configuration knobs
- Decision trees (when to enable/disable)
- Integration steps
- Validation tests
- Performance expectations

**Next**: I can scaffold any of these modules. Which would you like to build first?

Priority recommendations:
1. **Pareto router** — Highest impact for multi-backend deployments
2. **CUA native tools** — Core to your computer use need
3. **TST trainer** — Keep in codebase for future pretraining
4. **MSM pipeline** — Keep for alignment work
5. **ELF/HOPE/MMTP** — Optional, advanced capabilities

Tell me your priority, and I'll generate the full implementation with tests.
