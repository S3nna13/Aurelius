# Aurelius Architecture Review

**Author:** Automated Architecture Audit
**Date:** 2026-04-28
**Scope:** 27 Python files + 1 Rust crate (rust_memory)

---

## 1. Layer Boundaries and Interfaces

### Current Layering (top-down)

```
AGENT LAYER      agent_train.py → agent_loop.py → agent_core.py, skills.py
                      ↑                ↑
TRAINING LAYER   train_3b.py, train_optimized.py, train.py → memory_optimizer.py
                      ↑                ↑
MODEL LAYER      aurelius_model_3b.py → aurelius_model_1b.py → aurelius_model.py
                      ↑                ↑
MEMORY LAYER     memory_core.py ← async_memory.py, adaptive_precision.py,
                                    prefetch_router.py, deduplication.py,
                                    ntm_memory.py, moe_memory.py
                      ↑                ↑
RUST LAYER       rust_memory/src/lib.rs (MemoryPageTable, MmapCheckpointWriter)

INFERENCE LAYER  mobile_inference.py, speculative_decoding.py
(unused by any other module)
```

### What's Good

- **Rust crate is isolated** — it exposes a PyO3 module but no Python code imports it yet. This is a clean boundary that will be easy to bridge when needed.
- **`memory_core.py` is genuinely low-level** — only imports torch. Everything else depends on it.
- **`aurelius_model.py` and `aurelius_model_1b.py`** have clean single-dependency: they import only `memory_core.AurelianMemoryCore`.

### What's Broken

- **`aurelius_model_3b.py` violates layering in both directions.** It imports `agent_core` and `skills` inside `AgentSkillBlock.__init__` (line 78-79) and `agent_loop` in `AureliusModel3B.__init__` (line 122). This means the model file is simultaneously:
  - A consumer of the memory layer (correct)
  - A consumer of the agent layer (creates circular potential since `agent_train.py` imports the model)
- **`fused_kernels.py` (line 36) imports `aurelius_model_1b.apply_rotary`** — an inference/optimization file reaching back into a model file. This is a reversed dependency: kernels should be called *by* models, not import them.
- **`unified_manager.py` imports from 5 different memory submodules** (`async_memory`, `adaptive_precision`, `prefetch_router`, `deduplication`), acting as a god object that knows about too many peers.

---

## 2. Dependency Direction Diagram

```
legend: ──→ "depends on"

agent_train.py ──→ aurelius_model_1b ──→ memory_core
                  agent_core
                  skills
                  agent_loop ──→ agent_core
                             ──→ skills

aurelius_model_3b.py ──→ memory_core
                       agent_core          (inline import, __init__)
                       skills              (inline import, __init__)
                       agent_loop          (inline import, __init__)

train_optimized.py ──→ aurelius_model_1b
                     memory_optimizer
                     kv_cache_quant

train_3b.py ──→ aurelius_model_3b
                agent_train
                memory_optimizer

fused_kernels.py ──→ aurelius_model_1b     (BACKWARD edge)

unified_manager.py ──→ async_memory
                       adaptive_precision
                       prefetch_router
                       deduplication

rlhf_lora.py ──→ aurelius_model_3b

memory_core.py ──→ (torch only)            CLEAN
ntm_memory.py   ──→ (torch only)           CLEAN
moe_memory.py   ──→ (torch only)           CLEAN
```

### Violations

1. **`aurelius_model_3b.py` → `agent_loop.py` → `agent_core.py`**: The model should not import the agent loop. The `agent_loop.AgentLoopController` and `agent_loop.AgentMemoryBridge` should be injected or composed at the `AgentAureliusModel` level in `agent_train.py`, not inside the raw model file.

2. **`fused_kernels.py` → `aurelius_model_1b.py`**: The fallback path in `FlashAttention3Wrapper.forward` hot-imports `apply_rotary` from the 1B model file. This should be a shared utility, not a cross-layer import.

3. **`rlhf_lora.py` → `aurelius_model_3b.py`**: RLHF depends on a specific model variant. This is fine for a research project but means RLHF can't be applied to the 150M or 1B models without forking the RLHF code.

---

## 3. Potential Circular Dependencies

**Actual cycle exists:**

```
agent_train.py → aurelius_model_1b → memory_core
agent_train.py → agent_loop → agent_core, skills
agent_train.py → agent_core, skills
```

**No cycle yet**, but `aurelius_model_3b.py` imports `agent_loop` which makes it dangerously close. If `agent_loop.py` ever needs `aurelius_model_3b.AureliusModel3B` (e.g., for an agent that wraps the model), a direct cycle appears.

**Near-miss:** `rlhf_lora.py` imports `aurelius_model_3b`. If `aurelius_model_3b` ever imports `rlhf_lora` (for built-in RL head support), cycle forms.

**Recommendation:** Add `import` cycle detection to CI (`pip install pytest-cycles`).

---

## 4. Suggested Simplifications (Karpathy-style)

### 4.1 Dead Code — Remove These

| File | Dead Code | Reason |
|---|---|---|
| `deduplication.py` | `LZ4MemoryCompressor` | Imports `lz4.frame` at call-time with lazy import. No caller exists. The `unified_manager.py` instantiates it but never calls `compress_page` or `decompress_page`. |
| `deduplication.py` | `MemoryAwareGradientAccumulator` | Instantiated by `unified_manager.py` line 41-42, but `should_accumulate` and `step` are never used to control gradient accumulation anywhere — `unified_manager.on_forward` stores the result but nothing acts on it. |
| `fused_kernels.py` | `KernelRegistry`, `CUDAGraphMemoryWrapper` | No callers. `KernelRegistry.autotune` is a stub that times a single call. |
| `unified_manager.py` | `NUMAAwareMemoryPlacer`, `DistributedMemoryBalancer` | No callers. Defined but never instantiated. |
| `fp8_allreduce.py` | `GradientClippingWithCompression.compress` inside `clip_grad_norm_` | Compresses and decompresses immediately without any communication — it's a no-op waste of cycles. The compression result is never used. |
| `paged_optimizer.py` | `OptimizerStateCompressor` | No callers. `PagedAdamW` manages its own state. |
| `kv_cache_quant.py` | `MemoryBudgetTracker` | Used only for `snapshot`/`report` in `train_optimized.py`. Not connected to any memory management feedback loop. |

### 4.2 Accidental Generalization

- **`GraphConsolidator` in `memory_core.py`**: An entire nn.Module class (35 lines) for `F.normalize → matmul → threshold → adj → degree → normalize → cluster`. This could be a 10-line function. Every `LTSMemory` write does its own threshold logic — the `GraphConsolidator` is never called. The `AsyncConsolidationPipeline._consolidate` duplicates the same logic inline (lines 72-91). Two copies of the same graph consolidation with different thresholds.

- **`NTMController`, `NTMReadHead`, `NTMWriteHead`, `DifferentiableMemoryAugmentedBlock`**: 153 lines of Neural Turing Machine addressing that nothing uses. No model imports `ntm_memory.py`. This is speculative research code.

- **`MoEMemoryRouter`, `MemoryExpert`, `MoELTSMemory`**: 93 lines of MoE memory that nothing imports. Speculative.

- **`SparseLTSRouter`** in `prefetch_router.py`: Uses Python-level nested loops (`for i in range(b): for j in range(t):`). This is a GPU performance anti-pattern. Either use vectorized operations or delete.

- **`PredictiveAttentionRouter`** in `prefetch_router.py`: No callers. Speculative.

### 4.3 Excessive Module Count

The 5 memory-support files (`async_memory.py`, `adaptive_precision.py`, `prefetch_router.py`, `deduplication.py`, `unified_manager.py`) could be consolidated into 2:
- `memory_tiering.py`: `PagedLTSMemory`, `AdaptivePrecisionManager`, `TieredMemoryBank`, `FP8LTSMemory`
- `memory_ops.py`: `AsyncConsolidationPipeline`, `CosineDeduplicator`, `PredictiveMemoryPrefetcher`, `PriorityProportionalAllocator`, `LZ4MemoryCompressor`

This is 5 files → 2 files with no loss of coherence.

### 4.4 Dead Config Parameters

`config.yaml` line 15 has `weight_init: "small_init"` but no code reads this. `aurelius_model_1b.py` has `_init_weights` with hardcoded `std=0.02`.

---

## 5. Integration Points Between Layers

### 5.1 Memory → Model (healthy)

```
memory_core.AurelianMemoryCore ← imported by:
  aurelius_model.py:6
  aurelius_model_1b.py:6
  aurelius_model_3b.py:6
```

Interface: `AurelianMemoryCore(d_model, d_mem, episodic_slots, lts_capacity, ...)` with `.forward(h)` returning `(output, mem_state_dict)`.

This is clean. The only critique: `mem_state` is a raw dict with no schema validation. If keys change, callers silently get `KeyError` at runtime.

### 5.2 Agent → Model (fragile)

`agent_train.AgentAureliusModel` wraps `aurelius_model_1b.AureliusModel1B` by:
1. Running the base model's forward
2. Re-running the base model's token_embedding + blocks manually (lines 41-43 of `agent_train.py`)
3. Reading `mem_states['layer_0']['mem_read']`

This is a **code clone** of the model's forward pass. When `AureliusModel1B.forward` changes, `AgentAureliusModel.forward` will silently diverge. This is a maintenance trap.

**Fix:** Add an `intermediate_hidden` return to `AureliusModel1B.forward` rather than re-running the forward pass manually.

### 5.3 Training → Model (reasonable)

`train.py` → `AureliusModel`, `train_optimized.py` → `AureliusModel1B`, `train_3b.py` → `AureliusModel3B`.

Each training script instantiates its own model variant. There's no shared trainer base class — `MemoryEfficientTrainer` and `AgentTrainer` share zero code despite both doing:
- `optimizer.zero_grad()`
- `model(input_ids)`
- `loss_fn(logits, labels)`
- `loss.backward()`
- `optimizer.step()`

### 5.4 Rust → Python (missing)

`rust_memory` exposes `MemoryPageTable`, `MmapCheckpointWriter`, `DifferentialCheckpointer`, and `estimate_layer_memory` via PyO3. **No Python file imports `aurelius_memory`.** This crate is compiled but disconnected.

### 5.5 Inference → Model (unidirectional, correct)

`mobile_inference.py` and `speculative_decoding.py` do not import any model files. `speculative_decoding.py` takes `target_model: nn.Module` as a generic argument. This is the cleanest integration boundary in the project.

---

## 6. Test Coverage Gaps

There are **zero test files**. Not one `test_*.py` exists. Risk assessment by layer:

| Layer | Risk | Why |
|---|---|---|
| `memory_core.AurelianMemoryCore` | **CRITICAL** | Core architecture. No tests for correct memory read/write, surprise gate behavior, LTS capacity enforcement, graph consolidation thresholding. |
| `aurelius_model_1b.AureliusModel1B` | **CRITICAL** | The main model. No tests for forward shape correctness, gradient flow through memory, rotary embedding position encoding, causal masking. |
| `aurelius_model_3b.AgentSkillBlock` | **CRITICAL** | Most complex block: attention + FFN + memory + tool adapter + skill library. No tests for any sub-component interaction. |
| `agent_core.ToolFormerAdapter` | **HIGH** | Tool call head output shapes, cross-attention correctness, gating behavior. |
| `agent_loop.AgentLoopController` | **HIGH** | Observe→Think→Act→Reflect cycle completeness, MCTS tree expansion, value head consistency. |
| `skills.SkillRegistry` | **HIGH** | Skill retrieval top-k accuracy, embedding update momentum, success rate tracking. |
| `rlhf_lora.PPOTrainer` | **HIGH** | PPO surrogate loss, KL penalty, value function learning, LoRA merge/unmerge. |
| `paged_optimizer.PagedAdamW` | **MEDIUM** | LRU eviction correctness, state offloading/restoring, gradient masking. |
| `fp8_allreduce.FP8DistributedTrainer` | **MEDIUM** | Compression error feedback, hierarchical all-reduce topology, numerical accuracy. |
| `rust_memory.MemoryPageTable` | **MEDIUM** | Eviction scoring, promote/demote accounting, page table full behavior. |
| `deduplication.py` | **LOW** | (speculative/dead code, but `CosineDeduplicator` is the only one actually called) |
| `ntm_memory.py` | **LOW** | (unused) |
| `moe_memory.py` | **LOW** | (unused) |

### Minimum Viable Test Suite

1. `test_memory_core.py`: `AurelianMemoryCore` forward shapes, LTS write triggers after consolidation_freq, surprise gate output range [0,1]
2. `test_aurelius_model.py`: `AureliusModel1B` logits shape `(B, T, vocab_size)`, `generate` produces `(B, T+max_new_tokens)`, deterministic with seed
3. `test_agent_loop.py`: `AgentLoopController.forward` returns all expected keys, `ExperienceReplayBuffer.sample` returns correct shapes
4. `test_skills.py`: `SkillRegistry` forward with skill_ids and without, `learn_skill` updates embeddings, `get_top_skills` returns sorted by success×usage
5. `test_kv_cache.py`: `HierarchicalKVCache` write and read shapes, tier labels correct after eviction chain
6. `test_rust_bridge.py`: (integration) `MemoryPageTable` register/access/promote/demote accounting

---

## Summary of Recommended Actions

| Priority | Action | Files Affected | Effort |
|---|---|---|---|
| P0 | Delete `ntm_memory.py` or add test + integration | `ntm_memory.py` | 5 min |
| P0 | Delete `moe_memory.py` or add test + integration | `moe_memory.py` | 5 min |
| P0 | Fix `aurelius_model_3b.py` inline imports — move `agent_loop`, `agent_core`, `skills` to module-level or inject | `aurelius_model_3b.py:78-79, 122` | 30 min |
| P1 | Extract shared `apply_rotary` to a stand-alone utility so `fused_kernels.py` doesn't import `aurelius_model_1b` | `fused_kernels.py:36`, new `rotary_utils.py` | 15 min |
| P1 | Remove dead `LZ4MemoryCompressor` calls from `unified_manager.py` | `unified_manager.py:45,71-74` | 10 min |
| P1 | Remove `NUMAAwareMemoryPlacer`, `DistributedMemoryBalancer` from `unified_manager.py` | `unified_manager.py:125-187` | 10 min |
| P1 | Consolidate `async_memory.py` + `adaptive_precision.py` + `prefetch_router.py` + `deduplication.py` + `unified_manager.py` → 2 files | 5 files → 2 | 1 hr |
| P2 | Replace `agent_train.AgentAureliusModel` manual forward re-run with proper intermediate hidden return | `agent_train.py:40-43`, `aurelius_model_1b.py` | 30 min |
| P2 | Import `aurelius_memory` in Python (rust_memory crate is compiled but unused) | `unified_manager.py` or new bridge | 1 hr |
| P2 | Write minimum test suite (6 test files) | new files | 2-3 hrs |
| P3 | Convert `GraphConsolidator` to a function | `memory_core.py:63-78` | 10 min |
| P3 | Eliminate deduplicated graph consolidation code in `async_memory._consolidate` | `async_memory.py:71-91` | 30 min |
| P3 | Add `import` cycle detection to CI | `.github/workflows/` or equivalent | 20 min |
