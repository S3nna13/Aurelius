# Seed Insights: Inference / Serving Papers

---

## 1. FlashAttention (arXiv:2205.14135)

**Core Efficiency Mechanism:** IO-aware exact attention via tiling — instead of materializing the full N×N attention matrix in HBM, tiles of Q, K, V are loaded into fast on-chip SRAM, the softmax is computed incrementally online, and the result is written back. This reduces HBM reads/writes from O(N²) to O(N² / M) where M is SRAM size.

### Implementation Features for Neural Module

1. **Tiled forward kernel** — Loop over blocks of Q, K, V, load into SRAM, compute partial S, apply online softmax with rescaling (track running max and exp sum), accumulate partial O. Avoids full N×N materialization.
2. **Online softmax with rescaling** — For each new tile, update `m_new = max(m_old, rowmax(S))`, compute `l_new = exp(m_old - m_new) * l_old + rowsum(exp(S - m_new))`, and rescale O accordingly.
3. **Backward pass recomputation** — Don't store the full attention matrix. Recompute S from Q, K on-chip during backward using the saved softmax normalizers (m, l), halving memory to O(N).
4. **Block-sparse attention extension** — Accept a block-sparse mask to skip zero tiles entirely, enabling longer context (Path-X 16K, Path-256 64K) without full quadratic compute.
5. **Kernel fusion for multi-head attention** — Fuse the Q·K^T multiply, softmax, and O·V accumulation into a single CUDA kernel per block to eliminate redundant HBM round-trips.

### KV Cache Management

-  Not directly a KV cache paper, but the tiling principle applies: during inference, the KV cache tiles can be streamed from HBM to SRAM block-by-block, computing attention incrementally without loading the full cache.

### Attention Optimization

- IO-complexity-optimal: for SRAM size M, FlashAttention achieves Θ(N²) HBM accesses vs. Θ(N²) for standard attention (N² / M improvement).
- 1.7-2.4× wall-clock speedup vs. PyTorch standard attention; 15% end-to-end BERT-large training speedup.

---

## 2. PagedAttention / vLLM (arXiv:2309.06180)

**Core Efficiency Mechanism:** OS-inspired virtual memory paging for KV cache. The KV cache for each request is divided into fixed-size blocks (pages), mapped through a block table. This eliminates internal/external fragmentation and enables copy-on-write sharing across requests (e.g., parallel sampling, beam search).

### Implementation Features for Neural Module

1. **Block table manager** — Maintain a page table mapping logical KV cache blocks to physical GPU memory blocks. Allocate/free physical blocks on-demand as sequences grow, rather than pre-allocating max-length contiguous buffers.
2. **Paged attention kernel** — Modify the attention kernel to accept non-contiguous KV cache blocks via a block table indirection. Gather K/V heads per block on-the-fly during the attention computation.
3. **Copy-on-write sharing** — For beam search or parallel sampling, share the same prefix KV cache blocks across sequences. Only copy a block when a sequence diverges (COW), dramatically reducing memory for shared prefixes.
4. **Waste-free memory allocation** — Eliminate the ~60-80% KV cache waste in existing systems (FasterTransformer, Orca) caused by internal fragmentation of pre-allocated buffers. Achieve near-zero waste.
5. **Swapping-aware scheduler** — When GPU memory is exhausted, swap least-recently-used KV cache blocks to CPU DRAM. The block-level paging makes swap granularity efficient and fine-grained.

### KV Cache Management

-  Page-level allocation: fixed-size blocks (16-32 tokens per block), mapped via block table. 
-  Copy-on-write for cross-request sharing of prefix KV blocks.
-  Demand-based paging: pages allocated only when tokens are generated, not pre-allocated.
-  2-4× throughput improvement over state-of-the-art (FasterTransformer, Orca).

### Attention Optimization

-  Attention kernel modified to iterate over page table entries, fetching K/V heads from scattered physical blocks.
-  Enables larger batch sizes because KV cache memory is no longer the bottleneck.
-  Longer sequences and larger models see greater improvements.

---

## 3. Speculative Decoding (arXiv:2211.17192)

**Core Efficiency Mechanism:** Use a fast, cheap draft model to generate K candidate tokens in a single forward pass, then verify all K tokens in parallel with the target model using a single forward pass. A rejection sampling scheme ensures the output distribution is *identical* to the target model's distribution (exact sampling, no quality loss).

### Implementation Features for Neural Module

1. **Draft model integration** — Maintain a lightweight draft model (e.g., a smaller transformer or a single-layer head) that runs autoregressively to propose γ candidate tokens. The draft model should share the target's embedding/vocab space.
2. **Parallel verification with modified attention** — Feed the full draft sequence (length γ) through the target model in a single forward pass. Compute logits for all γ positions simultaneously. Cache the KV states from this forward pass to reuse.
3. **Rejection sampling scheme** — For each proposed token, sample from the target distribution. Accept if `random() < min(1, p_target / p_draft)`. Reject at the first failure, resample from `(p_target - p_draft)_+` normalized. This guarantees the output distribution matches the target exactly.
4. **Speculative execution loop** — Orchestrate: draft → K forward passes, target → 1 parallel verify. Accept a prefix of the draft, then discard the tail and repeat. Tune γ dynamically based on acceptance rate.
5. **KV cache reuse across speculation rounds** — On acceptance, append the accepted KV states to the cache. On rejection, recompute KV from the last accepted prefix. Avoid redundant computation by preserving KV cache continuity.

### KV Cache Management

-  KV cache from the verification forward pass is reused: the accepted tokens' KV states are appended to the running cache.
-  On reject, the KV cache rolls back to the last verified prefix position.
-  Draft model's KV cache is managed separately (typically much smaller).

### Attention Optimization

-  No attention algorithm change — the orthogonal gain is from parallelizing the serial decoding bottleneck.
-  2×-3× wall-clock speedup on T5-XXL with no output distribution change.
-  Works with any off-the-shelf model, no retraining needed.
-  Compatible with both PagedAttention and FlashAttention — orthogonal optimization.

---

## Cross-Cutting Themes

| Concern | FlashAttention | PagedAttention | Speculative Decoding |
|---------|---------------|---------------|---------------------|
| Memory | SRAM/HBM tiling | Block-level paging | Draft/target model split |
| Latency | Fewer HBM accesses | Larger batches | Parallel token generation |
| KV Cache | Tile streaming | Page table + COW | KV reuse across rounds |
| Orthogonal | Yes (with both) | Yes (with both) | Yes (with both) |
