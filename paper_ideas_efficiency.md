# Paper Ideas: Training & Inference Efficiency

---

## 1. FlashAttention-2 (arXiv 2307.08691)

**Core Idea:** Tiling attention onto GPU SRAM with improved warp-level work partitioning to reach 50-73% of theoretical max FLOPs/s, making attention nearly as efficient as GEMM.

### Implementable Ideas

**Idea 1: Tiled Online Softmax with 2D Block Partitioning**
- Split Q/K/V into (batch, nhead, seqlen, d) tiles that fit in SRAM
- Compute softmax online: maintain running m_i = max(m_{i-1}, row_max), l_i = exp(m_{i-1} - m_i) * l_{i-1} + exp(row - m_i), accumulate P * V with rescaling
- Assign ceil(seqlen / block_size) thread blocks per head instead of 1 block per head
- Mask out non-exact tile boundaries with a causal mask

```python
def flashattention_2d_tiled(q, k, v, block_size=128):
    batch, nhead, seqlen, d = q.shape
    num_tiles = ceil(seqlen / block_size)
    acc = torch.zeros_like(q)
    lse = torch.full((batch, nhead, seqlen, 1), -float('inf'))
    for j in range(num_tiles):
        k_tile = k[:, :, j*block_size:(j+1)*block_size]
        v_tile = v[:, :, j*block_size:(j+1)*block_size]
        for i in range(num_tiles):
            q_tile = q[:, :, i*block_size:(i+1)*block_size]
            S = q_tile @ k_tile.transpose(-2, -1)
            if i == j:
                S = S.masked_fill(torch.triu(torch.ones_like(S), diagonal=1) == 1, -1e9)
            m_new = torch.maximum(lse[:, :, i*block_size:(i+1)*block_size], S.max(-1, keepdim=True))
            rescale = torch.exp(lse[:, :, i*block_size:(i+1)*block_size] - m_new)
            lse[:, :, i*block_size:(i+1)*block_size] = m_new
            P = torch.exp(S - m_new)
            acc[:, :, i*block_size:(i+1)*block_size] *= rescale
            acc[:, :, i*block_size:(i+1)*block_size] += P @ v_tile
    return acc, lse
```

**Idea 2: Reduce non-matmul FLOPs by fusing rescaling into online softmax**
Fold the online softmax correction factor directly into the matmul accumulation loop instead of separate rescaling steps.

```python
m_new = torch.maximum(m_prev, row_max)
alpha = torch.exp(m_prev - m_new)
l_new = alpha * l_prev + row_sum.exp(m_prev - m_new)
P_hat = alpha.unsqueeze(-1) * P_hat + exp_scores
O = (alpha.unsqueeze(-1) * O + (exp_scores / l_new.unsqueeze(-1)) @ V)
```

**Idea 3: Warp-Level Work Partitioning (split K & Q dims)**
Within each thread block, assign contiguous ranges of the K/V dimension to each warp instead of having all warps redundantly load K/V. Each warp computes its partial S_i = Q @ K_i^T and PV_i = softmax_partial(S_i) @ V_i. Final warp reduces via shared memory. Reduces shared memory reads from 2x per matmul to 1x.

**Idea 4: Backward Pass without Full Attention Matrix Materialization**
Recompute P = exp(S - lse) from stored Q, K, lse on-the-fly during backward. dQ, dK, dV computed via recomputed P. Avoids storing O(N^2) attention matrix; stores only O(N) lse per head.

**Integration:** Replace F.scaled_dot_product_attention with this tiled kernel. No architecture changes needed.

---

## 2. PagedAttention / vLLM (arXiv 2309.06180)

**Core Idea:** Manage KV cache in fixed-size blocks (pages) with virtual-to-physical address translation, eliminating fragmentation and enabling copy-free prefix sharing.

### Implementable Ideas

**Idea 1: Block-Based KV Cache with Page Table**
Divide KV cache into fixed-size blocks (16 tokens). Maintain a block table mapping (request_id, layer, logical_block) -> physical_block. On generation, append KV to current block; when full, allocate new physical block from free list.

```python
class PagedAttentionCache:
    def __init__(self, num_layers, num_heads, head_dim, block_size=16, num_gpu_blocks=2048):
        self.block_size = block_size
        self.k_cache = torch.empty(num_gpu_blocks, num_layers, num_heads, block_size, head_dim)
        self.v_cache = torch.empty(num_gpu_blocks, num_layers, num_heads, block_size, head_dim)
        self.free_blocks = list(range(num_gpu_blocks))
        self.block_table = {}

    def append_kv(self, req_id, layer, key, value):
        logical_blk = self._current_logical_block(req_id, layer)
        if logical_blk not in self.block_table.get((req_id, layer), {}):
            phys = self.free_blocks.pop()
            self.block_table[(req_id, layer, logical_blk)] = phys
        phys = self.block_table[(req_id, layer, logical_blk)]
        offset = self._offset_in_block(req_id, layer)
        self.k_cache[phys, layer, :, offset] = key
        self.v_cache[phys, layer, :, offset] = value
```

**Idea 2: Prefix (Copy-Free) KV Sharing via Block Table**
For requests sharing a common prefix, map initial logical blocks to the same physical blocks. On divergence, clone only the diverging block (COW with refcounting).

**Idea 3: Attention with Block Table Lookup**
During decoding, scatter current token KV into last block, gather preceding blocks by iterating block_table.

```python
def paged_attention(q, logical_block_ids, block_table, k_cache, v_cache, block_size):
    scores = []
    for logical_blk in logical_block_ids:
        phys = block_table[logical_blk]
        k_blk = k_cache[phys]
        s = q @ k_blk.transpose(-2, -1)
        scores.append(s)
    scores = torch.cat(scores, dim=-1)
    attn = torch.softmax(scores / sqrt(d), dim=-1)
    output = 0
    for i, logical_blk in enumerate(logical_block_ids):
        phys = block_table[logical_blk]
        v_blk = v_cache[phys]
        blk_attn = attn[:, i*block_size:(i+1)*block_size]
        output += blk_attn @ v_blk
    return output
```

**Idea 4: Preemption via Block Eviction**
When GPU OOM, free lowest-priority request's block chain. Recompute KV on resume. ~10x memory savings for long-sequence serving.

---

## 3. StreamingLLM / Attention Sinks (arXiv 2309.17453)

**Core Idea:** Initial tokens act as attention sinks absorbing excess attention mass. Keeping first 4 tokens + sliding window enables infinite-length generation without fine-tuning.

### Implementable Ideas

**Idea 1: Attention Sink + Rolling Window Cache**
Maintain a KV cache of fixed size W. Always keep first 4 tokens (sink) + most recent W-4 tokens. Drop middle tokens when sequence exceeds W.

```python
class StreamingLLMCache:
    def __init__(self, window_size=4096, num_sink_tokens=4):
        self.window_size = window_size
        self.num_sink = num_sink_tokens
        self.num_recent = window_size - num_sink_tokens
        self.kv_cache = []
        self.sink_kv = []

    def append(self, k, v):
        if len(self.kv_cache) < self.num_sink:
            self.sink_kv.append((k, v))
            self.kv_cache.append((k, v))
        else:
            self.kv_cache.append((k, v))
            if len(self.kv_cache) > self.window_size:
                evict_start = self.num_sink
                evict_end = len(self.kv_cache) - self.num_recent
                self.kv_cache[evict_start:evict_end] = []

    def get_kv(self):
        return self.sink_kv + self.kv_cache[self.num_sink:]
```

**Idea 2: Dummy Sink Token Injection During Pre-Training**
Add a learnable [SINK] token as first token in every training sequence. This absorbs the attention sink naturally, preventing real tokens from being distorted.

**Idea 3: Positional Encoding Offset for Streaming**
Re-index positions: sink tokens always stay at 0,1,2,3; recent tokens get contiguous positions from num_sink onward.

```python
def stream_rope_positions(sink_len, recent_start_pos, recent_len):
    pos = torch.cat([
        torch.arange(sink_len),
        torch.arange(recent_start_pos, recent_start_pos + recent_len)
    ])
    return pos
```

**Idea 4: On-the-fly KV compression via token merging in the window**
When the window fills, merge the two oldest KV entries via weighted averaging (by attention score) instead of dropping them entirely.

**Integration:** Drop-in replacement for past_key_values in HuggingFace generate(). No training needed.

---

## 4. Megatron-LM (arXiv 1909.08053)

**Core Idea:** Intra-layer tensor model parallelism - split transformer weight matrices column-wise / row-wise across GPUs with minimal allreduce communication.

### Implementable Ideas

**Idea 1: Column-Parallel + Row-Parallel MLP with Fwd/Bwd Allreduce**
Split first projection column-wise (no communication needed). Second projection row-wise (allreduce after to combine).

```python
class MegatronParallelMLP(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, world_size, rank):
        super().__init__()
        assert ffn_dim % world_size == 0
        self.fc1 = nn.Linear(hidden_dim, ffn_dim // world_size, bias=False)
        self.fc2 = nn.Linear(ffn_dim // world_size, hidden_dim, bias=False)

    def forward(self, x):
        intermediate = F.gelu(self.fc1(x))
        output = self.fc2(intermediate)
        torch.distributed.all_reduce(output)
        return output
```

**Idea 2: Fused Attention w/ Column-Parallel QKV Projection**
Each GPU has W_qkv split column-wise: computes local Q_l, K_l, V_l, runs local attention on its heads, then all-gather before output projection.

```python
class MegatronParallelAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, world_size):
        super().__init__()
        assert num_heads % world_size == 0
        self.num_local_heads = num_heads // world_size
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim // world_size, bias=False)
        self.out_proj = nn.Linear(hidden_dim // world_size, hidden_dim, bias=False)

    def forward(self, x):
        qkv_local = self.qkv(x)
        q, k, v = qkv_local.chunk(3, dim=-1)
        attn_out = scaled_dot_product_attention(q, k, v)
        gathered = attn_out.new_zeros(...)
        torch.distributed.all_gather_into_tensor(gathered, attn_out)
        return self.out_proj(gathered)
```

**Idea 3: Gradient Bucketing Overlap**
Bucket gradient all-reduces by layer. Start all-reduce for layer i-1 while layer i backward computes.

**Idea 4: Asynchronous Embedding Gradient Reduction**
Use reduce_scatter on sparse embedding gradients; overlap with first transformer layer forward.

---

## 5. ZeRO: Memory Optimizations (arXiv 1910.02054)

**Core Idea:** Partition optimizer states, gradients, and parameters across data-parallel GPUs so each device stores only 1/Nd of total, reclaiming ~5x memory vs standard DDP.

### Implementable Ideas

**Idea 1: ZeRO Stage 1 - Partition Optimizer States**
Each GPU stores optimizer states for only its assigned parameter partition. Forward/backward run as normal DDP. After backward, all-gather gradients, then update only partition.

```python
class ZeROStage1(nn.Module):
    def __init__(self, module, world_size, rank, lr=1e-4):
        super().__init__()
        self.module = module
        self.optim = torch.optim.AdamW(
            [p for i, p in enumerate(module.parameters()) if i % world_size == rank],
            lr=lr
        )

    def step(self):
        for p in self.module.parameters():
            torch.distributed.all_gather_into_tensor(p.grad, p.grad)
        self.optim.step()
        for p in self.module.parameters():
            torch.distributed.broadcast(p, src=(p._id % self.world_size))
```

**Idea 2: ZeRO Stage 2 - Partition Gradients + Optimizer States**
Extend Stage 1 with reduce-scatter on gradients after backward. Each GPU keeps only 1/Nd of gradients.

**Idea 3: ZeRO Stage 3 - Partition Parameters (FSDP)**
Parameters partitioned across GPUs. Before layer forward, all-gather parameters; after forward, free non-owned shards. Prefetch next layer while current computes.

```python
class ZeROStage3Layer(nn.Module):
    def forward(self, x):
        with all_gather_ctx(self.layer.flat_param):
            out = self.layer(x)
        self.layer.flat_param.free_shard()
        return out
```

**Idea 4: ZeRO-Offload - Offloading States to CPU**
Move fp32 optimizer states + gradients to CPU pinned memory. GPU holds only fp16 params + gradients during fwd/bwd. Optimizer step runs async on CPU while GPU computes next micro-batch.

```python
def zero_offload_step(model, cpu_optim, stream):
    for p in model.parameters():
        cpu_grad = torch.empty_like(p, device='cpu', pin_memory=True)
        with torch.cuda.stream(stream):
            cpu_grad.copy_(p.grad, non_blocking=True)
    cpu_optim.step()
    for p in model.parameters():
        with torch.cuda.stream(stream):
            p.data.copy_(p.fp32_replica.to('cuda', non_blocking=True))
```

**Idea 5: Communication Volume Optimal Scheduling**
Schedule all-gather/reduce-scatter to overlap with compute. Issue all-gather for layer N+1 while computing layer N. Use async CUDA events.

```python
def prefetch_all_gather(fsdp_module):
    for cur, nxt in zip(fsdp_module[:-1], fsdp_module[1:]):
        nxt._prefetch_handles = cur._all_gather_params(async_op=True)
        cur_out = cur.forward(hidden_states)
        hidden_states = cur_out
```

**Integration:** Use DeepSpeed initialize() for drop-in ZeRO or PyTorch FSDP. Pair with gradient checkpointing for max memory.

---

## Cross-Paper Synergies

| Technique | FlashAttn-2 | PagedAttn | StreamingLLM | Megatron | ZeRO |
|-----------|-------------|-----------|--------------|----------|------|
| FlashAttn tiling | - | Block-level | Window kernel | Reduces OOM | Reduces batch memory |
| Paged KV cache | Recompute in pages | - | Sink+slide pages | Shard table | Offload pages |
| Streaming sink | Sink = pinned tile | Sink in block 0 | - | Sink on all GPUs | Sink in shard |
| Megatron split | Split QKV heads | Shard table | Sink across ranks | - | Orthogonal to ZeRO |
| ZeRO partition | Rank tiles subset | 1/Nd pages | Sink replicated | DeepSpeed-Mega | - |

**Recommended stack for training:** ZeRO-2/3 (DeepSpeed) + FlashAttention-2. For 100B+: Megatron tensor parallelism + ZeRO-3 pipeline + ZeRO-Offload.

**Recommended stack for inference:** vLLM (PagedAttention) + StreamingLLM for long context + optional FP8 KV cache quantization.
