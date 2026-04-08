# Heavens Gate Integration — Agent Handoff

**Date:** 2026-04-07  
**Branch:** main  
**Last commit:** `ec22a71` feat: add Magpie self-instruct synthesizer  
**Test suite:** 97 passed, 3 skipped

---

## What This Project Is

Aurelius is a from-scratch PyTorch 1.3B decoder-only transformer (GQA, SwiGLU, RoPE, RMSNorm). All algorithms are implemented natively in PyTorch — no black-box library wrappers, no external API calls (no OpenAI, no Anthropic). Goal: a fully autonomous personal model.

**Run tests:** `.venv/bin/python3.13 -m pytest`  
**Working dir:** `/Users/christienantonio/Desktop/Aurelius`

---

## Completed Tasks (Tasks 1–4)

### Task 1 — KV cache in GroupedQueryAttention ✅
**File:** `src/model/attention.py`  
- `forward(x, freqs_cis, mask=None, past_kv=None) -> tuple[Tensor, tuple[Tensor, Tensor]]`
- Cache stored pre-GQA-expansion (n_kv_heads, not n_heads)
- RoPE applied to new k/v BEFORE concat with cache
- `is_causal = mask is None and past_kv is None`
- Guard: raises `ValueError` if `past_kv is not None and S > 1`
- 13 tests in `tests/model/test_attention.py`

### Task 2 — KV cache + loss + generate() in AureliusTransformer ✅
**File:** `src/model/transformer.py`  
- `forward(input_ids, mask=None, labels=None, past_key_values=None) -> tuple[loss|None, logits, present_key_values]`
- `freqs_cis = self.freqs_cis[past_len : past_len + S]` (position offset — critical)
- Shifted cross-entropy loss: `logits[:, :-1]` vs `labels[:, 1:]`
- `generate(input_ids, max_new_tokens, temperature, top_p, eos_token_id)` with batch-correct top-p nucleus sampling (ascending sort + scatter back to original indices)
- 14 tests in `tests/model/test_transformer.py`

### Task 3 — Memory-mapped tokenized shard loader ✅
**File:** `src/data/tokenized_loader.py`  
- `TokenizedShardDataset(shard_paths, seq_len, stride=None)`
- Uses `np.load(path, mmap_mode='r')` on .npy uint16 shards
- Returns `(input_ids, labels)` int64 tensors of shape `(seq_len,)`
- Labels = input_ids shifted left by 1 (next-token prediction)
- Window count: `(n - (seq_len+1)) // stride + 1` per shard
- O(log n) shard lookup via `np.searchsorted`
- 6 tests in `tests/data/test_tokenized_loader.py`

### Task 4 — Magpie self-instruct synthesizer ✅
**File:** `src/data/magpie.py`  
- `MagpieSynthesizer(model, tokenizer, config)` — uses `AureliusTransformer.generate()` only
- Two passes: (1) pre-query prefix → instruction, (2) full prompt → response
- EOS tokens stripped from both outputs
- Chat template: `<|user|>{instruction}<|end|>\n<|assistant|>`
- 5 tests in `tests/data/test_magpie.py`

---

## Remaining Tasks (Tasks 5–12)

Run each task, then spec review, then code quality review before marking complete.

### Task 5 — Spectrum SNR-based layer selection
**File to create:** `src/alignment/spectrum.py`  
**Test:** `tests/alignment/test_spectrum.py`  
**Paper:** arXiv:2406.06623  
**Purpose:** Automatically selects which transformer layers to apply LoRA to, based on signal-to-noise ratio of their weight matrices.

**Algorithm:**
```python
def compute_snr(weight: Tensor) -> float:
    """SNR = (signal singular values) / (noise singular values), normalized by S[0]."""
    S = torch.linalg.svdvals(weight.float())
    # Robust noise floor: IQR / 1.3489 (Marchenko-Pastur distribution)
    q75, q25 = torch.quantile(S, torch.tensor([0.75, 0.25]))
    sigma = (q75 - q25).item() / 1.3489
    gamma = weight.shape[0] / weight.shape[1]  # aspect ratio
    epsilon = sigma * (1.0 + math.sqrt(gamma))  # Marchenko-Pastur upper bound
    signal = S[S > epsilon].sum().item()
    noise = S[S <= epsilon].sum().item()
    if noise == 0:
        return float('inf')
    snr = (signal / noise) / S[0].item()  # normalize by top singular value
    return snr
```

**Class interface:**
```python
class SpectrumSelector:
    def __init__(self, model: nn.Module, top_k_fraction: float = 0.25): ...
    def select_layers(self) -> list[str]:
        """Returns list of parameter names (e.g. 'layers.0.attn.q_proj.weight')
        sorted by SNR descending, top top_k_fraction selected per module type group."""
```

**Group by module type** (attn.q_proj, attn.k_proj, etc. are separate groups). Select top `top_k_fraction` per group by SNR descending. Higher SNR = more signal = better LoRA target.

**Tests to write:**
- `test_select_layers_returns_list_of_strings`
- `test_select_fraction_respected` — e.g. top_k_fraction=0.5 selects ~half per group
- `test_snr_is_positive` — no negative SNR values
- `test_higher_snr_selected` — manually set weight SNRs, verify correct ones selected

### Task 6 — DoRA weight-decomposed LoRA
**File to create:** `src/alignment/dora.py`  
**Test:** `tests/alignment/test_dora.py`  
**Paper:** arXiv:2402.09353  

**CRITICAL:** `.detach()` on the norm denominator (Section 4.3 of paper). Without this, gradients explode.

**Algorithm:**
```python
class DoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with DoRA adaptation.
    
    Weight decomposition: W = m * (V / ||V||_col)
    where V = W0 + s * B @ A (LoRA update)
          m = learned magnitude vector (initialized to ||W0||_col)
    """
    def __init__(self, weight: Tensor, rank: int, alpha: float = 1.0): ...
    
    def forward(self, x: Tensor) -> Tensor:
        V_prime = self.W + self.scale * (self.B @ self.A)
        # CRITICAL: .detach() on denominator (paper Section 4.3)
        V_prime_norm = V_prime.norm(p=2, dim=1, keepdim=True).detach()
        norm_scale = self.m / V_prime_norm
        base_out = F.linear(x, self.W)
        lora_out = F.linear(x, self.scale * (self.B @ self.A))
        return (norm_scale.squeeze(-1) - 1) * base_out + norm_scale.squeeze(-1) * lora_out
    
    def merge_weights(self) -> Tensor:
        """Return merged weight = (m / ||V'||_col) * V'"""
        V_prime = self.W + self.scale * (self.B @ self.A)
        V_prime_norm = V_prime.norm(p=2, dim=1, keepdim=True)
        return (self.m / V_prime_norm) * V_prime
```

**Tests to write:**
- `test_output_shape` — same shape as nn.Linear
- `test_m_initialized_to_weight_norm` — m values match ||W0||_row at init
- `test_no_gradient_through_norm_denominator` — verify `.detach()` breaks grad flow
- `test_merge_weights_shape` — merged weight same shape as original W
- `test_trainable_params` — only A, B, m are trainable (W is frozen)

**Helper function also needed:**
```python
def apply_dora_to_model(model: nn.Module, target_modules: list[str], rank: int, alpha: float) -> dict[str, DoRALinear]:
    """Replace named linear layers with DoRALinear. Returns dict of replaced modules."""
```

### Task 7 — Regression gate for adapter quality
**File to create:** `src/alignment/regression_gate.py`  
**Test:** `tests/alignment/test_regression_gate.py`  

**Purpose:** Before committing a trained DoRA adapter, compare perplexity on a held-out eval set. If the new model is worse than baseline by more than `threshold_pct`, reject the adapter and archive it.

**Interface:**
```python
@dataclass
class GateResult:
    accepted: bool
    baseline_ppl: float
    new_ppl: float
    regression_pct: float  # (new - baseline) / baseline * 100
    reason: str

class RegressionGate:
    def __init__(self, threshold_pct: float = 5.0): ...
    
    def evaluate(
        self,
        model: AureliusTransformer,
        dataset: Dataset,
        device: str = "cpu",
    ) -> float:
        """Compute perplexity on dataset. Returns perplexity."""
    
    def check(
        self,
        baseline_model: AureliusTransformer,
        new_model: AureliusTransformer,
        dataset: Dataset,
        adapter_path: str | Path | None = None,
    ) -> GateResult:
        """Run gate. If regression > threshold, optionally archive adapter."""
```

**Perplexity computation:** `exp(mean cross-entropy)` over the dataset.

### Task 8 — Lloyd-Max codebook for Beta(2,2)
**File to create:** `src/inference/turboquant/lloyd_max.py`  
**Test:** `tests/inference/turboquant/test_lloyd_max.py`  
**Purpose:** Quantization codebook used in TurboQuant Stage 1 (PolarQuant).

**Key detail:** Beta(2,2) distribution is the distribution of normalized KV cache values after min-max scaling to [0,1]. Lloyd-Max computes optimal quantization centroids for this distribution.

**Closed-form centroid update for Beta(2,2) — NO scipy:**
```python
def _beta22_cdf(t: float) -> float:
    """CDF of Beta(2,2): 3t^2 - 2t^3"""
    return 3.0 * t**2 - 2.0 * t**3

def _beta22_antiderivative_numerator(t: float) -> float:
    """Antiderivative of t * p(t) for Beta(2,2): 2t^3 - (3/2)t^4"""
    return 2.0 * t**3 - 1.5 * t**4

def _beta22_antiderivative_denominator(t: float) -> float:
    """Antiderivative of p(t) for Beta(2,2): 3t^2 - 2t^3"""
    return 3.0 * t**2 - 2.0 * t**3
```

**Interface:**
```python
def compute_lloyd_max_codebook(n_codes: int, n_iter: int = 100) -> Tensor:
    """Compute Lloyd-Max codebook for Beta(2,2) distribution.
    Cache result at module level after first call.
    Returns sorted centroids of shape (n_codes,) in [0, 1]."""

_CODEBOOK_CACHE: dict[int, Tensor] = {}  # module-level cache
```

**Tests:** verify centroids are sorted, in [0,1], n_codes values returned, deterministic across calls.

### Task 9 — PolarQuant (Stage 1 of TurboQuant)
**File to create:** `src/inference/turboquant/polar_quant.py`  
**Test:** `tests/inference/turboquant/test_polar_quant.py`  
**Paper:** arXiv:2504.19874  

**Algorithm:**
1. Random orthogonal rotation: `Q, _ = torch.linalg.qr(torch.randn(d, d))`
2. Rotate input: `x_rot = x @ Q.T`
3. Per-vector min-max normalize to [0, 1]
4. Lloyd-Max quantize (nearest centroid lookup)
5. Store: quantized codes + per-vector min + per-vector max (for dequantization)
6. Compute residual: `residual = x - dequantize(codes, mins, maxs, Q)`

**Interface:**
```python
@dataclass
class PolarQuantState:
    codes: Tensor      # int indices into codebook, shape (B, S, n_kv_heads, head_dim)
    mins: Tensor       # per-vector mins, shape (B, S, n_kv_heads, 1)
    maxs: Tensor       # per-vector maxs
    Q: Tensor          # the rotation matrix

class PolarQuant:
    def __init__(self, dim: int, n_codes: int = 256): ...
    def compress(self, x: Tensor) -> tuple[PolarQuantState, Tensor]:
        """Returns (state, residual)"""
    def decompress(self, state: PolarQuantState) -> Tensor:
        """Reconstruct approximate x from state"""
```

### Task 10 — QJL Gaussian sketch inner product estimator
**File to create:** `src/inference/turboquant/qjl.py`  
**Test:** `tests/inference/turboquant/test_qjl.py`  
**Paper:** arXiv:2406.03482  

**CRITICAL:** S matrix is `N(0,1)` Gaussian, NOT ±1 binary. This is what makes the estimator unbiased.

**Algorithm:**
```python
# Key generation (compress residual):
proj = residual.float() @ S.T        # (B, S, n_kv_heads, sketch_dim)
signs = proj.sign().to(torch.int8)   # binary quantization
norms = residual.norm(dim=-1)        # per-vector norms

# Query projection (full precision):
q_proj = query.float() @ S.T        # (B, 1, n_heads, sketch_dim)

# Asymmetric inner product estimator:
# E[estimate] = <key_residual, query>
estimate = norms * math.sqrt(math.pi / 2) / sketch_dim * (signs.float() * q_proj).sum(dim=-1)
```

**Interface:**
```python
class QJLSketch:
    def __init__(self, dim: int, sketch_dim: int, seed: int = 42): ...
    def compress_keys(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (signs, norms)"""
    def estimate_attention(self, signs: Tensor, norms: Tensor, query: Tensor) -> Tensor:
        """Returns inner product estimates, shape (B, n_heads, S_keys)"""
```

### Task 11 — TurboQuant compressor + KV backend
**Files to create:** `src/inference/turboquant/compressor.py`, `src/inference/turboquant/kv_backend.py`, `src/inference/turboquant/__init__.py`  
**Test:** `tests/inference/turboquant/test_compressor.py`  

**Pipeline:** `x → PolarQuant → (state, residual) → QJL(residual) → (signs, norms)`

**kv_backend.py:** Drop-in replacement for the plain KV cache. Stores compressed keys/values. On attention query: decompress approximately and compute attention.

### Task 12 — Best-of-N with self-certainty scoring
**File to create:** `src/inference/best_of_n.py`  
**Test:** `tests/inference/test_best_of_n.py`  
**Paper:** arXiv:2502.18581  

**Self-certainty score:** Negative KL from uniform distribution.
```python
# For each completion y given prompt x:
# self_certainty = -1/(n_tokens * vocab) * sum(log(vocab * p(t|x, y<t)))
#                = mean(-log(vocab * p))
#                = mean(log(1/vocab) - log(p))  — higher = more confident
def self_certainty_score(logits: Tensor, token_ids: Tensor) -> float:
    """
    logits: (seq_len, vocab_size) — the model's logits for the completion
    token_ids: (seq_len,) — the actual generated tokens
    Returns: self-certainty score (higher = more confident = better)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(len(token_ids)), token_ids]
    vocab_size = logits.shape[-1]
    return (-token_log_probs - math.log(vocab_size)).mean().item()
```

**Interface:**
```python
class BestOfN:
    def __init__(self, model: AureliusTransformer, n: int = 4): ...
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate N completions, score each, return the best."""
```

---

## Key Technical Invariants (Do Not Break)

1. **All algorithms are native PyTorch** — no scipy, no openai, no anthropic, no HuggingFace model wrappers
2. **KV cache stores pre-GQA-expansion keys/values** (n_kv_heads, not n_heads)
3. **DoRA `.detach()` on norm denominator** — non-negotiable, training explodes without it
4. **QJL S matrix is N(0,1) Gaussian** — NOT ±1 binary
5. **Spectrum uses SNR** (higher = better target) — NOT condition number
6. **Top-p sampling uses scatter-back** for batch correctness
7. **Test runner:** `.venv/bin/python3.13 -m pytest`

---

## File Structure So Far

```
src/
  model/
    attention.py       # GQA + KV cache (Task 1)
    transformer.py     # Full transformer + generate() (Task 2)
    config.py
    ffn.py
    rms_norm.py
  data/
    tokenized_loader.py  # Memory-mapped shard loader (Task 3)
    magpie.py            # Self-instruct synthesis (Task 4)
  alignment/
    (empty — Tasks 5, 6, 7 go here)
  inference/
    turboquant/
      (empty — Tasks 8-11 go here)
    (Task 12 goes here)
tests/
  model/          # 27 tests passing
  data/           # 11 tests passing
  alignment/      # (Tasks 5-7)
  inference/      # (Tasks 8-12)
  training/       # 15 tests passing (pre-existing)
```
