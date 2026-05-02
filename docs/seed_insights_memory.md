# Seed Insights: Memory / Architecture Papers for Neural Module Design

> Extracted from 4 critical papers. Each section: (A) core mechanism, (B) neural module impl, (C) read/write/addressing features, (D) interface with reasoning.

---

## 1. Neural Turing Machines (Graves, Wayne & Danihelka, 2014)
**arXiv:1410.5401**

### A. Core Mechanism
A recurrent controller (LSTM or feedforward) interacts with an external `N x M` memory matrix via differentiable read/write heads. The entire system is end-to-end differentiable via backpropagation-through-time.

### B. Neural Module Implementation
- **Controller**: any differentiable network (LSTM used in paper) that outputs query vectors at each timestep.
- **Memory Matrix**: `N` locations, each an `M`-dim vector. Trainable but the *content* is written/read dynamically.
- **Head Layer**: for each head, emit a *read vector* and *write vector* plus addressing parameters.
- **Loss**: standard supervised cross-entropy + optional memory usage regularizers.

### C. Read/Write/Addressing Features (3-5 concrete module features)
1. **Content-based addressing** — each head emits a *key vector* `k_t`; similarity against all memory rows produces a read/write weighting via softmax: `w_t(i) = softmax( β_t · cos(M_t(i), k_t) )`. This is a learnable attention-over-memory mechanism.
2. **Location-based addressing** — a 1D shift (convolutional) plus interpolation lets the head iterate sequentially through memory. Implemented as: `w_t = g_t · w_t^g + (1-g_t) · w_{t-1}` (interpolation gate), then circular convolution with shift filter `s_t`, then sharpening via `γ_t`.
3. **Sharpening** — after shift, apply `w_t(i) = w_t(i)^γ_t / Σ_j w_t(j)^γ_t` to prevent blurring over multiple shift steps. Critical for precise pointer arithmetic.
4. **Write head dual operation** — each write head produces both an *erase vector* `e_t` and an *add vector* `a_t`. Memory update: `M_t(i) = M_{t-1}(i) · (1 - w_t(i) · e_t) + w_t(i) · a_t`. Erase-before-add enables fine-grained memory modification.
5. **Read head retrieval** — read vector: `r_t = Σ_i w_t(i) · M_t(i)`. Weighted sum across all memory locations (fully differentiable).

### D. Interface with Reasoning
The controller learns *algorithms* (copy, sort, associative recall) by routing read/write operations. Reasoning emerges as learned sequences of memory accesses — the controller learns when to store, when to retrieve, and where to focus via the addressing system. The key insight: **the same memory can store multiple data structures simultaneously** because content-based addressing lets the controller demultiplex them by key.

---

## 2. Differentiable Neural Computers (Graves, Wayne et al., 2016)
**Nature 538, 471–476**

### A. Core Mechanism
Extends NTM with **dynamic memory allocation**, **temporal memory linkage**, and **free/erase gates**. DNC solves the NTM's inability to free memory and track write-order across locations. Introduces a *sparse-link matrix* `L` that records the temporal order of writes.

### B. Neural Module Implementation
- **Controller**: LSTM or feedforward, outputs an *interface vector* that is decomposed into read/write head parameters.
- **Memory**: `N x W` matrix plus three tracking structures: usage vector `u`, precedence vector `p`, link matrix `L`.

### C. Read/Write/Addressing Features (3-5 concrete module features)
1. **Dynamic memory allocation** — a usage vector `u_t` tracks how recently each location was read/written. New writes go to the *least-used* location (not content-addressed). Implemented as: sort `u_t` ascending, assign write weighting to the free locations. Enables memory reuse.
2. **Temporal link matrix** — `L_t[i][j]` stores the probability that location `i` was written to after location `j`. Updated via: `L_t(i,j) = (1 - w_t^w(i) - w_t^w(j)) · L_{t-1}(i,j) + w_t^w(i) · p_{t-1}(j)`. Enables **forward/backward reads** through write order.
3. **Free gates** — each read head produces a free gate `f_t^i ∈ [0,1]` that modulates how much a read operation frees the location: `ψ_t = Π_{i=1}^R (1 - f_t^i · w_{t-1}^{r,i})`. The usage vector is then: `u_t = ψ_t · (u_{t-1} + w_{t-1}^w - u_{t-1} · w_{t-1}^w)`.
4. **Content + location read fusion** — each read head produces both a content-based weighting `w_t^c` and a temporal weighting `w_t^f` (forward) / `w_t^b` (backward). These are gated by an interpolation parameter: `w_t^r = g_t^i · (α_t · w_t^b + β_t · w_t^f + γ_t · w_t^c)`.
5. **Memory-independent computation** — controller weights are entirely separate from memory size. DNCs can be trained with `N=256` and tested with `N=512` without retraining (see Extended Data Figure 2). This separation is critical for scaling.

### D. Interface with Reasoning
DNC solves structured graph tasks (shortest path, transitive inference, block-world reasoning). Reasoning emerges as the controller learns to: (1) allocate fresh memory for each new fact/edge, (2) follow temporal chains via the link matrix to traverse paths, (3) free memory after a subgoal is satisfied. The **temporal link matrix is the key enabler of multi-step reasoning** — it turns memory from a bag of vectors into an ordered structure that supports sequential logic.

---

## 3. End-To-End Memory Networks (Sukhbaatar, Szlam, Weston & Fergus, 2015)
**arXiv:1503.08895**

### A. Core Mechanism
A **stacked attention** architecture where input memories are embedded and repeatedly attended to. Unlike NTM/DNC, there is no explicit write mechanism — the memory is the *input itself* (set of sentences/facts). Multiple *hops* allow iterative reasoning by alternating memory lookup and query update.

### B. Neural Module Implementation
- **Input Memory**: `N` sentences/facts each embedded as vector `m_i` via embedding matrix A.
- **Output Memory**: same sentences in a separate embedding space `c_i` via embedding matrix C.
- **Query (question)**: embedded via embedding B to produce `u`.
- **Multiple Hops**: after each hop, the query is updated: `u^{k+1} = u^k + o^k`, where `o^k` is the memory read output.

### C. Read/Write/Addressing Features (3-5 concrete module features)
1. **Soft attention over input facts** — attention weight: `p_i = softmax(u^T · m_i)`. This is the *only* addressing mechanism. No content-based key, no location shift. Simplicity is the point.
2. **Weighted sum read** — read vector: `o = Σ_i p_i · c_i`. Standard attention (like NTM read) but from a separate *output* embedding space, decoupling "where to look" from "what to extract."
3. **Temporal encoding** — to handle sequence order (missing from bag-of-words memory), each memory slot gets a position-specific bias added: `m_i = A · x_i + T_A(i)`, `c_i = C · x_i + T_C(i)`. The temporal matrix `T_A`/`T_C` is learned per position.
4. **Multi-hop (weight tying)** — 2-3 layers of attention are stacked. Each layer (hop) uses the same embedding matrices (A, C) but the query updates. Weight tying between layers forces the network to learn distinct attention patterns per hop. The number of hops equals the number of reasoning steps the model can perform.
5. **Final answer prediction** — after K hops, the final query `u^K` is fed through a weight matrix W: `a = softmax(u^K · W)`. No recurrent controller — reasoning is purely attention-based.

### D. Interface with Reasoning
Reasoning is *iterative attention*. Each hop refines attention to the relevant supporting facts. For bAbI tasks, a 3-hop MemNN effectively chains: *find entity A → find relation involving A → find target entity B*. The key insight: **no explicit memory writing is needed when the entire "knowledge base" is the memory**. This makes MemNNs simpler than NTM/DNC but limits them to the provided input — they cannot create new memory structures.

---

## 4. Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
**arXiv:2312.00752**

### A. Core Mechanism
A **structured state space model** (SSM) with a *selective scan* mechanism. The hidden state `h_t` evolves as: `h_t = Āh_{t-1} + B̄x_t`, and output `y_t = C̄h_t`. The key innovation: matrices `B̄`, `C̄`, and discretization step `Δt` are *input-dependent* (selective), allowing the model to "forget" or "remember" based on content.

### B. Neural Module Implementation
- **State dimension**: `N` (typically 16-64, much smaller than NTM/DNC's `N x M`).
- **Discretization**: continuous parameters `A, B, Δt` → discrete `Ā, B̄` via zero-order hold: `Ā = exp(Δt · A)`, `B̄ = (Δt · A)^{-1} · (exp(Δt · A) - I) · Δt · B`. Simplified in practice to `Ā = exp(Δt · A)` and `B̄ = Δt · B`.
- **Selective mechanism**: `Δt = softplus(linear(x_t))`, `B = linear(x_t)`, `C = linear(x_t)`. This is the critical difference from prior SSMs — the dynamics depend on the input.

### C. Read/Write/Addressing Features (3-5 concrete module features)
1. **Selective state update** — `Δt` acts as a *gating mechanism* (analogous to an RNN's forget gate). Large `Δt` → fast state update (write). Small `Δt` → state persists (remember). Unlike NTM/DNC, there's no separate memory matrix — the state *is* the memory.
2. **Input-dependent scanning (S6)** — the selective scan kernel computes the recurrence in a hardware-efficient parallel scan (`O(NL)` work, `O(L)` memory). Each element's forget/update is input-dependent, enabling the model to selectively propagate or discard information.
3. **Convolutional mode vs. recurrent mode** — during training, the SSM can be unrolled as a convolution (parallelizable). During inference, it operates as a recurrent network (`O(1)` memory per token). This dual-mode is unique — NTM/DNC must always use sequential attention.
4. **No explicit addressing** — Mamba has *no read/write heads, no attention, no memory matrix*. Memory is implicit in the state vector. This makes it computationally efficient but means it cannot perform the structured data manipulation that NTM/DNC can.
5. **Selective forget = implicit memory allocation** — the input-dependent `Δt` effectively learns *when to overwrite* the state. This is analogous to DNC's dynamic memory allocation, but implicit and integrated into the state dynamics rather than explicit.

### D. Interface with Reasoning
Mamba achieves strong performance on language modeling, DNA modeling, and long-range reasoning *without* explicit memory. The selective mechanism lets the model:
- Store relevant context in the state (`large Δt` for important tokens)
- Ignore irrelevant tokens (`small Δt`)
- Recall information from the state when needed via the output projection `y_t = C̄h_t`

However, Mamba **cannot** perform the explicit data-structure manipulations of NTM/DNC (graph traversal, pointer chains) because its memory is a single dense state vector rather than an addressable matrix. It trades structured reasoning capability for linear-time scalability.

---

## Cross-Paper Synthesis

| Feature | NTM | DNC | MemNN | Mamba |
|---------|-----|-----|-------|-------|
| Memory type | Matrix `N x M` | Matrix + links + usage | Input embeddings | State vector `R^N` |
| Write mechanism | Erase+add | Allocation + erase+add | None (input = memory) | Selective state update |
| Read mechanism | Content + location | Content + temporal links | Soft attention | Output projection |
| Addressing | Content (cosine) + shift | Content + alloc + temporal | Content (dot) | None (implicit) |
| Temporal tracking | Shift (1D conv) | Link matrix | Position embeddings | Recurrence |
| Alloc/Free | None | Usage vector + free gates | N/A | Implicit via Δt |
| Complexity per step | O(NM) | O(N²) for links | O(NM) | O(N) |

### Key Architectural Lessons

1. **Explicit addressing enables structure** — NTM/DNC's content+location addressing lets them implement algorithms. Mamba's implicit memory trades this for speed.
2. **Memory management is critical** — DNC's allocation/free mechanism vs. NTM's memory corruption shows that forgetting is as important as remembering.
3. **The link matrix is expensive but powerful** — DNC's `O(N²)` link update is the cost for temporal chain traversal. MemNN avoids this by baking time into position encodings.
4. **Mamba's selectivity ≈ attention** — the input-dependent `Δt` serves the same role as attention's "what to focus on," but is `O(N)` instead of `O(N²)`.
