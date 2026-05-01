# Aurelius AI Model — Memory-Augmented Agent with Skills

**Description**: Use when building, training, or extending a memory-augmented transformer model with agent capabilities, differentiable memory systems, and learned skill architectures.

**Trigger phrases**: "aurelius", "memory-augmented", "agent memory", "skill acquisition", "AMC"

---

## Architecture

### 3-Tier Memory
| Tier | Mechanism | Capacity |
|------|-----------|----------|
| Working | Transformer hidden states (context window) | 16K–32K tokens |
| Episodic | FIFO slot buffer via BiGRUSlotEncoder | 4K slots (config-driven) |
| Long-Term | Neural matrix with importance-based write/read (LTSMemory) | 8K+ slots |

Memory flow: `SurpriseGate` scores salience → Episodic slots are consolidated into LTS on fixed schedule → `GraphConsolidator` builds adjacency-based cluster features.

### Agent Loop
1. **Observe** — `ToolFormerAdapter` cross-attends to tool descriptions
2. **Think** — `PlanningModule` runs MCTS with ValueHead for lookahead
3. **Act** — `ToolCallHead` selects tool; `SkillLibrary` retrieves and executes skill
4. **Reflect** — `CriticHead` scores state-action pair and suggests corrections
5. **Learn** — Episodes stored in replay buffer; `SkillAcquisition` extracts new skills via momentum update

### Skill Library
- 8192–16384 learnable skill embeddings (`SkillEmbedding`), config-driven
- Top-16/24 retrieval (`SkillRetriever`) with soft-weighted composition
- Momentum-based acquisition (`SkillAcquisition`, τ = 0.99) from trajectory pooling
- Success-rate tracking per skill for automatic pruning

---

## Key Files

| File | Purpose |
|------|---------|
| `memory_core.py` | SurpriseGate, Episodic slots, LTSMemory, GraphConsolidator, AurelianMemoryCore |
| `agent_core.py` | ToolFormerAdapter, ToolCallHead, PlanningModule (MCTS), ValueHead, CriticHead |
| `skills.py` | SkillEmbedding, SkillRetriever, SkillController, SkillAcquisition, SkillRegistry, SkillLibrary |
| `agent_loop.py` | AgentLoopController, AgentMemoryBridge, AgentContextManager, ExperienceReplayBuffer |
| `hierarchical_kv_cache.py` | 3-tier KV cache with importance-based eviction |
| `moe_memory.py` | Mixture-of-Experts memory routing |
| `ntm_memory.py` | Neural Turing Machine addressing (content+location) |
| `speculative_decoding.py` | Draft-verify loop conditioned on memory context |
| `paged_optimizer.py` | Paged AdamW with CPU offload for large-state training |
| `fp8_allreduce.py` | FP8 gradient compression with all-reduce |
| `rlhf_lora.py` | LoRA-based RLHF with model offloading |
| `mobile_inference.py` | On-device quantized inference pipeline |
| `rust_memory/src/lib.rs` | Rust page table allocator + mmap checkpoint engine |
| `skills_registry.py` | Contract surface for SkillRegistry with RegistryEntry, lookup, and verify_contract |
| `agent_registry.py` | Contract surface for agent components (ToolFormerAdapter, PlanningModule, CriticHead, etc.) |
| `api_registry.py` | Contract surface for training scripts, configs, generation, and model checkpoints |
| `tool_schema_registry.py` | Contract surface for tool modules (memory, cache, optimizer, quantization, etc.) |
| `test_brain_layer.py` | 11 tests for brain_layer.py (RMSNorm, InputEncoder, WorkingMemory, ReasoningCore, etc.) |
| `test_alignment_efficiency.py` | 10 tests for alignment_impl.py and efficiency_impl.py |
| `test_memory_system.py` | 11 tests for async_memory, deduplication, prefetch_router, adaptive_precision |
| `test_remaining.py` | 9 tests for rust_bridge, train_3b, generate_report, agent_train, kv_cache_quant, fused_kernels |
| `tests.py` | 132 tests covering all core modules |

---

## Usage

```python
from memory_core import AurelianMemoryCore
from agent_loop import AgentLoopController, AgentMemoryBridge

mem = AurelianMemoryCore(d_model=768, d_mem=256, episodic_slots=4096,
                         lts_capacity=8192, consolidation_freq=64)

agent = AgentLoopController(d_model=768, n_heads=12, d_mem=256,
                            n_known_tools=64, n_simulations=8)

output = mem(hidden_states)                       # memory-augmented forward
result = agent(hidden_states, full_cycle=True)    # full Observe→Think→Act→Reflect
```

```bash
make test          # pytest tests.py -v (132 core tests)
make test-all      # pytest tests.py test_*.py -v (173 total tests)
make check         # compile-check all .py + Rust
make build_rust    # cargo build release for mmap/page_table
```

---

## Scaling Path

| Size | Config | d_model | Layers | Heads |
|------|--------|---------|--------|-------|
| 150M | `config.yaml` | 768 | 12 | 12 |
| 1.2B | `config_1b.yaml` | 1536 | 24 | 16 |
| 3.3B | `config_3b.yaml` | 2560 | 32 | 32 |
| 7B | `config_7b.yaml` | 3584 | 40 | 40 |
| 14B | — | 5120 | 48 | 48 |
| 32B | — | 7168 | 56 | 56 |
