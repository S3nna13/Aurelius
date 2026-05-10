# Paper Ideas: Memory & Mixture-of-Experts Architectures

## 1. Sparsely-Gated Mixture of Experts (1701.06538)

**Core Idea:** Replace dense FFN layers with thousands of parallel "expert" sub-networks where a learned sparse gating network activates only a small subset of experts per token, enabling >1000x parameter scaling at constant FLOPs.

### Implementable Ideas

**Idea 1.1 — Noisy Top-k Gating**
```
class NoisyTopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, k=4, noise_std=0.01):
        self.w_gate = nn.Parameter(torch.randn(input_dim, num_experts))
        self.w_noise = nn.Parameter(torch.randn(input_dim, num_experts))
        self.k = k; self.noise_std = noise_std

    def forward(self, x):
        # x: [batch, seq, d_model]
        h = x @ self.w_gate                          # logits per expert
        noise = x @ self.w_noise                      # learned noise
        noise = F.softplus(noise) * self.noise_std    # positive stddev
        noisy_logits = h + torch.randn_like(h) * noise
        top_k_logits, top_k_indices = noisy_logits.topk(self.k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)
        return probs, top_k_indices                    # sparse dispatch
```
- **Loss:** Auxiliary load-balancing loss = CV^2(expert_usage) to prevent collapse
- **Integration:** Drop into any Transformer FFN block; dispatch tokens to selected experts via `torch.gather`, compute expert FFN, combine weighted by probs
- **Architecture:** `TokenDispatch → SparseFFN → Combine` — all-gather expert outputs across devices

**Idea 1.2 — Expert Parallelism with All-to-All Communication**
- Split experts across GPUs; each GPU holds a shard of experts
- Forward: each GPU has N/k tokens per expert; scatter via all-to-all, compute expert FFN, gather via all-to-all
- Integration: wrap in `xs = all_to_all(xs, expert_assignment)` before expert computation
- Key impl detail: capacity factor = `ceil(num_tokens / (num_experts * k)) * C` — tokens exceeding capacity are dropped & pass through residual

**Idea 1.3 — Distillation from MoE to Dense**
- Train large sparse MoE, then distill into dense student via KL(teacher_logits || student_logits) + LM loss
- Integration: freeze MoE teacher, train student on same data; 50-80% of MoE quality at 1/10th params

**Idea 1.4 — Importance & Load Losses**
```python
def load_balance_loss(importance, load):
    # importance = softmax(logits).sum(0)  per expert
    # load = (top_k_mask).sum(0)           per expert
    cv = load.std() / load.mean() + importance.std() / importance.mean()
    return cv * alpha  # alpha ~ 0.01
```

---

## 2. Expert Choice Routing (2202.09368)

**Core Idea:** Invert MoE routing — let each expert select its top-k tokens instead of letting tokens select top-k experts, guaranteeing perfect load balance and letting important tokens route to more experts.

### Implementable Ideas

**Idea 2.1 — Expert-Choice (EC) Routing**
```python
class ExpertChoiceRouter(nn.Module):
    def __init__(self, d_model, num_experts, capacity_per_expert):
        self.router = nn.Linear(d_model, num_experts)  # [d_model, E]
        self.capacity = capacity_per_expert

    def forward(self, x):
        # x: [B*S, d_model] — flatten batch & seq
        scores = self.router(x)  # [B*S, E] — token→expert affinities
        # Each expert picks its top-k tokens
        top_k_scores, top_k_indices = scores.topk(self.capacity, dim=0)
        # top_k_indices: [capacity, E] — which tokens each expert gets
        probs = F.softmax(top_k_scores, dim=0)
        return probs, top_k_indices, scores
```
- **Key property:** Each expert sees exactly `capacity` tokens → perfect load balance (no dropped tokens!)
- **Variable dispatch:** A token can be routed to 0, 1, or many experts (unlike fixed top-k)
- **Integration:** Replace top-k gating; same all-to-all scatter/gather pattern but with fixed expert bucket sizes

**Idea 2.2 — Batched Token Dispatch with 2D Scatter**
- Create dispatch mask: `dispatch_mask[token_idx, expert_idx] = 1` if expert picks token
- Use `einsum` or sparse matrix multiply for efficient dispatch:
```python
def expert_choice_dispatch(x, scores, capacity, num_experts):
    # x: [N, d]; scores: [N, E]
    # For each expert e, take top-capacity tokens
    top_scores, top_tokens = torch.topk(scores.T, capacity, dim=1)  # [E, cap]
    # Build one-hot dispatch: [E, cap, N]
    dispatch_mask = F.one_hot(top_tokens, num_classes=N).float()
    expert_input = dispatch_mask @ x  # [E, cap, d]
    return expert_input, top_scores, dispatch_mask
```

**Idea 2.3 — Adaptive Expert Count per Token**
- EC naturally gives high-scoring tokens more expert coverage: a token might be selected by 1 expert or all E experts
- No explicit auxiliary load-balancing loss needed — the routing is inherently balanced
- Integration with Transformer: replace top-1/top-2 gating, works with any existing MoE block

**Idea 2.4 — 2x Training Speedup Mechanism**
- Source of speedup: no expert collapse/under-training; all experts train at same rate due to fixed capacity
- Implementation: set `capacity = 2 * (N / E)` (2x the uniform allocation) for each expert
- Result: better convergence, 2x fewer steps to reach same perplexity vs Switch/GShard

---

## 3. Switch Transformers (2101.03961)

**Core Idea:** Radical simplification of MoE — route each token to exactly one expert (top-1) with bfloat16 training, reducing communication/computation while scaling to trillions of parameters.

### Implementable Ideas

**Idea 3.1 — Simplified Top-1 Routing**
```python
class SwitchRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        self.w_router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        logits = self.w_router(x)
        probs = F.softmax(logits, dim=-1)
        top1_probs, top1_indices = probs.max(dim=-1)
        # Route each token to exactly 1 expert
        mask = F.one_hot(top1_indices, num_classes=num_experts).float()
        # Aux loss: fraction of tokens routed to each expert ≈ uniform
        aux_loss = (mask.mean(0) * probs.mean(0)).sum() * num_experts
        return top1_probs, top1_indices, aux_loss
```
- **Key insight:** Top-1 routing cuts all-to-all communication in half (only 1 value per token sent)
- **Integration:** Replace standard MoE gating; reduces expert computation by ~2x over top-2

**Idea 3.2 — Capacity Factor & Token Dropping**
```python
def enforce_capacity(dispatch_mask, capacity_factor=1.0):
    tokens_per_expert = dispatch_mask.sum(0)        # [E]
    capacity = int(tokens_per_expert.mean() * capacity_factor)
    # Drop tokens beyond capacity — they pass through via residual connection
    sorted_probs, sorted_indices = dispatch_mask.float().sort(dim=0, descending=True)
    keep = sorted_indices[:capacity]                 # keep top capacity tokens
    return keep
```
- capacity_factor=1.0: no slack, many dropped tokens; capacity_factor=2.0: typical
- Dropped tokens: `output = residual(x)` (no expert computation)

**Idea 3.3 — Expert Choice + Switch Hybrid**
- Use Switch's simplification (single router linear layer) but with Expert Choice's inverted selection
- Results: simple router + balanced buckets + no load loss

**Idea 3.4 — bfloat16 Training for Sparse Models**
- Key trick: router computations in float32 (numerical stability), expert FFN in bfloat16
- Implementation: `with torch.cuda.amp.autocast(dtype=torch.bfloat16)`
- Integration: cast router logits to float32, then back to bf16 for scatter/gather

**Idea 3.5 — Selective Precision per Module**
```python
class SwitchFFN(nn.Module):
    def forward(self, x, expert_weights, expert_indices):
        # Router weights in fp32, expert computation in bf16
        x_bf16 = x.to(torch.bfloat16)
        # dispatch, compute expert(x_bf16), combine
        # final output in fp32
        return output.to(torch.float32)
```

---

## 4. Compressive Transformers (1911.05507)

**Core Idea:** Extend Transformer memory by compressing old hidden states into coarser "compressed memories" using 1D convolutions/attention-pooling, enabling long-range sequence modeling beyond the attention window.

### Implementable Ideas

**Idea 4.1 — Compressive Memory Module**
```python
class CompressiveMemory(nn.Module):
    def __init__(self, d_model, mem_len=512, cmem_len=256, compress_ratio=2):
        self.mem = []                    # fine-grained memory (recent)
        self.cmem = []                   # compressed memory (older)
        self.compressor = nn.Conv1d(d_model, d_model, kernel_size=compress_ratio,
                                    stride=compress_ratio)  # or attention pooling

    def compress(self):
        # Take oldest chunk from mem, compress to cmem
        chunk = self.mem.pop(0)           # [seq_len, d_model]
        compressed = self.compressor(chunk.T).T  # [seq_len//ratio, d_model]
        self.cmem.append(compressed)
        # Evict oldest cmem if over limit
        if len(self.cmem) > self.max_cmem_len:
            self.cmem.pop(0)

    def get_memory(self):
        # Concatenate fine + compressed memory for attention
        return torch.cat([self.cmem, self.mem], dim=0) if self.cmem else self.mem
```
- **Pooling options:** mean-pool, max-pool, 1D-conv, attention-pool (learned query over chunk)
- **Integration:** Attach to each Transformer layer; memory = concatenation of past hidden states + compressed past

**Idea 4.2 — Content-Based Addressing of Compressed Memory**
- Use standard attention keys/values for fine memory
- For compressed memory, store compressed keys and values separately
- Query attends over both `mem` (fine) and `cmem` (compressed) simultaneously:
```python
# In Transformer self-attention:
k = torch.cat([cmem_k, mem_k, cur_k], dim=0)
v = torch.cat([cmem_v, mem_v, cur_v], dim=0)
attn = softmax(q @ k.T / sqrt(d)) @ v
```

**Idea 4.3 — Compressive Loss (Reconstruction Objective)**
- Add auxiliary loss to make compressed representations useful:
```python
def compressive_loss(original, compressed):
    # original: [seq_len, d] — the chunk before compression
    # compressed: [seq_len//ratio, d]
    # Decode compressed back to original resolution
    decoded = F.interpolate(compressed.T, size=original.shape[0], mode='linear').T
    return F.mse_loss(decoded, original)
```
- Forces compressed states to preserve maximal information from original sequence
- Weight: `beta * compressive_loss` where beta ~ 0.1

**Idea 4.4 — Attention-Pooling Compressor**
```python
class AttentionPoolCompressor(nn.Module):
    def __init__(self, d_model, compress_ratio):
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, chunk):  # chunk: [L, d]
        # chunk → [L, d]; compress_ratio = L_out / L
        L_out = chunk.shape[0] // self.compress_ratio
        chunks = chunk.chunk(L_out, dim=0)  # list of [ratio, d]
        outputs = []
        for c in chunks:
            attn = F.softmax(self.query @ c.T / math.sqrt(c.size(-1)), dim=-1)
            out = attn @ c  # [1, d]
            outputs.append(out)
        return torch.stack(outputs).squeeze(1)
```

**Idea 4.5 — PG-19 Style Long-Context Evaluation**
- Benchmark: measure perplexity on sliding windows of length 8192+ tokens
- Integration: implement rolling memory buffer that detaches gradients after each window

---

## 5. Memorizing Transformers (2203.08913)

**Core Idea:** Augment Transformers with a non-differentiable kNN index over recent (key, value) pairs from internal representations, enabling the model to "memorize" new information at inference time without weight updates.

### Implementable Ideas

**Idea 5.1 — kNN-Augmented Attention Layer**
```python
class KNNAttention(nn.Module):
    def __init__(self, d_model, n_heads, memory_size=262144):
        self.attn = MultiheadAttention(d_model, n_heads)
        self.memory_queue = []   # FIFO queue of (key, value) pairs
        self.max_mem = memory_size
        self.k = 32              # number of nearest neighbors

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # Local attention
        local_out = self.attn(q, k, v)
        # kNN retrieval from memory
        mem_k, mem_v = self.get_memory()
        mem_out = knn_lookup(q, mem_k, mem_v, k=self.k)
        # Update memory with current k,v
        self.push_memory(k.detach(), v.detach())  # NO gradients into memory!
        return local_out + mem_out                # combined output
```

**Idea 5.2 — Approximate Nearest Neighbor (ANN) Index**
```python
def knn_lookup(query, memory_keys, memory_values, k=32):
    # query: [B, H, S, d]; memory_keys: [M, d]; memory_values: [M, d]
    # FAISS for GPU-accelerated MIPS
    import faiss
    # Normalize for cosine similarity
    mem_keys_norm = F.normalize(memory_keys, dim=-1)
    query_norm = F.normalize(query, dim=-1)
    # Search
    D, I = faiss.knn_gpu(query_norm.reshape(-1, d).float(),
                         mem_keys_norm.float(), k)
    # Gather values
    top_k_vals = memory_values[I]  # [N, k, d_v]
    weights = F.softmax(D / 0.1, dim=-1)   # temperature-scaled
    return (weights.unsqueeze(-1) * top_k_vals).sum(dim=1)  # weighted sum
```
- **Key insight:** Memory is *non-differentiable* — keys/values stored `.detach()` to prevent vanishing gradients through deep memory

**Idea 5.3 — Cross-Attention to Retrieved Neighbors**
- Instead of weighted sum, use retrieved kNN as additional key-value pairs for cross-attention:
```python
mem_keys = top_k_vals  # use values as keys too [N, k, d]
mem_vals = top_k_vals
# Cross-attend: q attends over [local_k, mem_keys]
k_all = torch.cat([k, mem_keys.transpose(0,1)], dim=1)
v_all = torch.cat([v, mem_vals.transpose(0,1)], dim=1)
out = scaled_dot_product_attention(q, k_all, v_all)
```
- This lets the model learn *how* to use memory rather than hard-coding combination

**Idea 5.4 — Memory Refresh & FIFO Eviction**
```python
def push_memory(self, keys, values):
    # keys: [S, d]; values: [S, d] — flattened over batch
    for k, v in zip(keys, values):
        self.memory_queue.append((k, v))
    while len(self.memory_queue) > self.max_mem:
        self.memory_queue.pop(0)  # FIFO eviction
```
- Memory capacity: up to 262K tokens (performance improves steadily with size)
- Integration: one shared memory per layer or across all layers

**Idea 5.5 — In-Context Memorization for Code & Math**
- The model can "read" new functions/theorems at test time and retrieve them via kNN
- Implementation: during inference, pre-fill memory with kNN vectors from a reference document
- Effect: zero-shot adaptation; model uses retrieved patterns without weight changes
- Integration: plug into any autoregressive decoder; memory is populated with past activations during generation

---

## Cross-Paper Integration Architecture

### Combined MoE + Memory Transformer

```python
class MoEMemoryTransformerLayer(nn.Module):
    def __init__(self, d_model, num_experts, memory_size):
        self.attn = MultiheadAttention(d_model)
        self.memory = KNNAttention(d_model, memory_size)
        self.moe = SparseMoEBlock(d_model, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + kNN memory retrieval
        x = x + self.attn(self.norm1(x))
        x = x + self.memory(self.norm1(x))
        # Sparse MoE FFN (Switch, EC, or Noisy Top-k)
        x = x + self.moe(self.norm2(x))
        return x
```

### Key Integration Points with Existing Codebases

| Module | Hooks Into | Losses Added |
|--------|-----------|-------------|
| NoisyTopKGating | `nn.Module.forward` | LoadBalanceLoss(importance, load) |
| ExpertChoiceRouter | `nn.Module.forward` | None (bal. is implicit) |
| SwitchRouter | `nn.Module.forward` | SwitchAuxLoss(fraction_routed × probs) |
| CompressiveMemory | Per-layer cache | CompressiveLoss(orig, compressed) |
| KNNAttention | Post-attention | None (non-differentiable store) |

### Training Recipe

1. **Stage 1:** Train base Transformer with Sparse MoE (Switch/EC) + load balancing loss
2. **Stage 2:** Add Compressive Memory to extend context window
3. **Stage 3:** Add kNN Memorizing Transformer for fast in-context adaptation
4. **Inference:** kNN memory populated dynamically; compressive memory stores long-range context
