# Heavens Gate Integration Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to implement this plan task-by-task.

**Goal:** Integrate seven self-contained modules extracted and re-implemented from the Heavens_Gate codebase to make Aurelius a fully trainable, alignment-capable, inference-optimized personal model — with all algorithms living natively in this codebase (no provider API routes, no black-box library calls where the math lives elsewhere).

**Architecture:** Each module is self-contained with its own tests. The model core fix unblocks Groups 2-4. Groups 2, 3, and 4 are independent of each other and can be implemented in parallel.

**Tech Stack:** PyTorch (all algorithms), numpy (mmap loader), pure Python string analysis (Best-of-N heuristic). No external inference APIs.

---

## Group 1 — Model Core (unblocks everything)

**Problem:** `AureliusTransformer.forward(input_ids, mask)` returns raw logits only. The trainer calls `model(input_ids=input_ids, labels=labels)` and expects a loss. There is no `generate()`. No KV cache. The model cannot train or produce text.

**Files:**
- Modify: `src/model/attention.py` — add `past_kv` in/out to `GroupedQueryAttention.forward()`
- Modify: `src/model/transformer.py` — add `labels` → loss to `forward()`, add KV cache threading, add `generate()`
- Modify: `tests/model/test_transformer.py` — add new test cases

### Changes

**`GroupedQueryAttention.forward()`** gains `past_kv: tuple[Tensor, Tensor] | None = None` and returns `(output, present_kv)`. When `past_kv` is provided, the cached K/V tensors are concatenated along the sequence dimension before computing attention. Returns `present_kv = (k, v)` always.

**`AureliusTransformer.forward()`** signature becomes:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    past_kv: list[tuple[Tensor, Tensor] | None] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, list[tuple[Tensor, Tensor]]]:
```
Returns `(loss, logits, present_kvs)`.

When `labels` is provided:
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, self.config.vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

**`generate()`** implemented from scratch on `AureliusTransformer`:
```python
def generate(
    self,
    input_ids: torch.Tensor,           # (1, prompt_len)
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
) -> torch.Tensor:
```

Top-p (nucleus sampling) implemented from scratch:
1. Sort logits descending
2. Compute cumulative softmax probabilities
3. Zero out tokens where cumsum exceeds `top_p`
4. Always keep at least one token
5. Sample from the filtered distribution via `torch.multinomial`

Top-k implemented from scratch:
1. Find the k-th largest logit value via `torch.topk`
2. Zero out all logits below that threshold
3. Sample

KV cache is a `list[tuple[Tensor, Tensor] | None]` of length `n_layers`, initialized to all `None`. Each layer returns its `present_kv`; the list is passed back into the next forward call.

---

## Group 2 — Data

### Module 2a: Tokenized Loader (`src/data/tokenized_loader.py`)

Memory-mapped numpy shard loader. Solves the gap where the trainer expects DataLoaders but none are constructed anywhere.

```python
class TokenizedDataset(Dataset):
    def __init__(self, shard_dir: Path, seq_len: int):
        # glob *.npy shards in shard_dir, sorted for determinism
        # np.load(path, mmap_mode='r') — OS pages in only accessed regions
        # precompute cumulative sequence counts: cum_seqs[i] = sum of seqs in shards 0..i
        # total_seqs = sum(shard_len // seq_len for each shard)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # binary search cum_seqs to find which shard and offset
        # slice [offset*seq_len : (offset+1)*seq_len] from mmap array
        # input_ids = tensor(slice)
        # labels = input_ids.clone(); labels[-1] = -100  (no target for last token)
        # return {"input_ids": input_ids, "labels": labels}

def build_dataloader(
    shard_dir: Path,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    # persistent_workers=True, pin_memory=True, drop_last=True
    # deterministic seeded sampler when shuffle=True
```

### Module 2b: Magpie Synthetic Generation (`src/data/magpie.py`)

Self-instruct synthetic data using your own local model. No external API.

```python
DOMAIN_PREFIXES = {
    "general":   "<|system|>You are a helpful assistant.<|end|><|user|>",
    "reasoning": "<|system|>You are an expert at step-by-step reasoning.<|end|><|user|>",
    "code":      "<|system|>You are an expert programmer.<|end|><|user|>",
}

def generate_magpie_sample(
    model: AureliusTransformer,
    tokenizer,
    domain: str = "general",
    max_instruction_tokens: int = 128,
    max_response_tokens: int = 512,
    **gen_kwargs,
) -> dict[str, str] | None:
    # encode prefix (no eos appended)
    # call model.generate() to complete the user turn (instruction)
    # stop at <|end|> token
    # re-encode prefix + instruction + "<|end|><|assistant|>"
    # call model.generate() for the response
    # stop at <|end|> token
    # return {"instruction": ..., "response": ..., "domain": domain}
    # return None if parsing fails

def generate_magpie_dataset(
    model, tokenizer,
    n_samples: int,
    output_path: Path,
    domains: list[str] | None = None,
) -> int:
    # round-robin across domains
    # write each valid sample as JSONL line (atomic per-line)
    # return count of valid samples written
```

---

## Group 3 — Alignment

### Module 3a: Spectrum Layer Selection (`src/alignment/spectrum.py`)

SVD-based LoRA target selection. Ranks every weight matrix by condition number (σ_max / σ_min). Low condition number = balanced singular values = most underfit = best LoRA targets.

```python
def compute_condition_number(param: torch.Tensor) -> float:
    mat = param.float().reshape(param.shape[0], -1)
    if mat.shape[0] > 2048 or mat.shape[1] > 2048:
        mat = mat[:2048, :2048]   # subsample large matrices — avoid OOM
    sv = torch.linalg.svdvals(mat)    # descending
    return (sv[0] / (sv[-1] + 1e-8)).item()

def select_spectrum_layers(
    model: nn.Module,
    top_k_pct: float = 0.25,
    param_filter: Callable[[str, Tensor], bool] | None = None,
) -> list[str]:
    # score every named 2D parameter
    # default filter: endswith projection layer suffixes
    # sort ascending by condition number
    # return top_k_pct fraction as list of parameter name strings
    # (these are passed directly to PEFT as target_modules)
```

Runs once before training, ~30 seconds for 1.3B model. No calibration data.

### Module 3b: DoRA (`src/alignment/dora.py`)

Weight-Decomposed Low-Rank Adaptation. Decomposes weight into column magnitude `m` and direction `V`. Low-rank adapter modifies direction; scalar per output column modifies magnitude. Better generalization than LoRA at same parameter count.

```python
class DoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        pretrained_weight: Tensor,
    ):
        self.W = nn.Parameter(pretrained_weight.clone(), requires_grad=False)
        self.m = nn.Parameter(
            pretrained_weight.norm(p=2, dim=1, keepdim=True)
        )  # shape: (out_features, 1)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        V = self.W + self.scale * (self.B @ self.A)
        V_norm = V / (V.norm(p=2, dim=1, keepdim=True) + 1e-8)
        W_eff = self.m * V_norm
        return F.linear(x, W_eff)

    def merge(self) -> nn.Linear:
        # compute W_eff, return as plain nn.Linear (no grad)

def apply_dora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> nn.Module:
    # walk model, replace matching Linear layers with DoRALinear
    # matching = param name ends with any string in target_modules

def merge_dora(model: nn.Module) -> nn.Module:
    # walk model, replace DoRALinear with merged nn.Linear
    # call before saving for serving
```

### Module 3c: Regression Gate (`src/alignment/regression_gate.py`)

Automatic perplexity check before any adapter is considered deployable.

```python
@dataclass
class GateResult:
    baseline_ppl: float
    adapted_ppl: float
    regression_pct: float
    passed: bool
    archive_path: Path | None   # set if rejected

class RegressionGate:
    def __init__(self, eval_data: list[str], threshold_pct: float = 5.0):
        self.eval_data = eval_data      # list of text strings
        self.threshold_pct = threshold_pct

    def _compute_perplexity(self, model, tokenizer) -> float:
        # tokenize each string, forward pass with labels, accumulate loss
        # return exp(mean_cross_entropy_loss)

    def check(
        self,
        base_model: nn.Module,
        adapter_state_dict: dict,
        tokenizer,
        adapter_path: Path,
    ) -> GateResult:
        baseline_ppl = self._compute_perplexity(base_model, tokenizer)
        # apply adapter weights onto a copy of base_model
        adapted_ppl = self._compute_perplexity(adapted_model, tokenizer)
        regression_pct = (adapted_ppl - baseline_ppl) / baseline_ppl * 100
        passed = regression_pct <= self.threshold_pct
        archive_path = None
        if not passed:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            archive_path = adapter_path.parent / f"{adapter_path.name}_rejected_{ts}"
            shutil.copytree(adapter_path, archive_path)
        return GateResult(baseline_ppl, adapted_ppl, regression_pct, passed, archive_path)
```

---

## Group 4 — Inference

### Module 4a: TurboQuant KV Compression (`src/inference/turboquant/`)

Two-stage KV cache compression. Stage 1 (PolarQuant): MSE-optimal quantization after random rotation. Stage 2 (QJL): 1-bit Johnson-Lindenstrauss sketch of the residual. Combines to ~2.5 bits/element with no calibration data.

**`lloyd_max.py`** — Computed once at module import. Never recomputed per-input.
```python
def build_lloyd_max_codebook(bits: int, n_iter: int = 100) -> tuple[Tensor, Tensor]:
    n_bins = 2 ** bits
    grid = torch.linspace(0.0, 1.0, 4096, dtype=torch.float64)
    pdf = 6.0 * grid * (1.0 - grid)   # Beta(2,2): p(x) = 6x(1-x)
    pdf = pdf / pdf.sum()
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # n_bins-1 internal
    for _ in range(n_iter):
        # assignment step: torch.bucketize(grid, boundaries) -> bin indices
        # update step: for each bin, centroid = (pdf * grid * mask).sum() / (pdf * mask).sum()
    return boundaries.float(), centroids.float()

# Module-level constants (computed once on import)
_CODEBOOKS: dict[int, tuple[Tensor, Tensor]] = {}
def get_codebook(bits: int) -> tuple[Tensor, Tensor]:
    if bits not in _CODEBOOKS:
        _CODEBOOKS[bits] = build_lloyd_max_codebook(bits)
    return _CODEBOOKS[bits]
```

**`polar_quant.py`**
```python
class PolarQuant:
    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        G = torch.randn(dim, dim, generator=torch.Generator().manual_seed(seed))
        Q, _ = torch.linalg.qr(G)
        self.R = Q          # fixed orthogonal rotation, not a parameter
        self.boundaries, self.centroids = get_codebook(bits)

    def compress(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_rot = x @ self.R.T                        # rotate to Beta(2,2) space
        x_min = x_rot.min(dim=-1, keepdim=True).values
        x_max = x_rot.max(dim=-1, keepdim=True).values
        scale = (x_max - x_min).clamp(min=1e-8)
        x_norm = (x_rot - x_min) / scale            # [0, 1]
        codes = torch.bucketize(x_norm, self.boundaries)
        x_hat_norm = self.centroids[codes]           # dequantize
        x_hat_rot = x_hat_norm * scale + x_min      # denormalize
        x_mse = x_hat_rot @ self.R                  # rotate back
        residual = x - x_mse
        return x_mse, residual
```

**`qjl.py`**
```python
class QJL:
    def __init__(self, dim: int, sketch_dim: int | None = None, seed: int = 12345):
        sketch_dim = sketch_dim or dim
        S = torch.randint(0, 2, (sketch_dim, dim),
                          generator=torch.Generator().manual_seed(seed)) * 2 - 1
        self.S = S.to(torch.int8)   # fixed sign matrix

    def sketch(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        proj = residual.float() @ self.S.float().T
        signs = proj.sign().to(torch.int8)
        signs[signs == 0] = 1
        norms = residual.norm(dim=-1, keepdim=True)
        return signs, norms

    def estimate_inner_product(self, query: Tensor, signs: Tensor, norms: Tensor) -> Tensor:
        q_proj = (query.float() @ self.S.float().T).sign().to(torch.int8)
        agreement = (q_proj * signs).sum(dim=-1, keepdim=True).float()
        return norms * (math.sqrt(math.pi / 2) / self.S.shape[0]) * agreement
```

**`compressor.py`**
```python
@dataclass
class CompressedKV:
    x_mse: Tensor       # PolarQuant MSE-optimal reconstruction
    signs: Tensor       # QJL 1-bit sketch (int8)
    norms: Tensor       # L2 norms of residuals

class TurboQuantCompressor:
    def __init__(self, dim: int, bits: int = 4,
                 polar_seed: int = 42, qjl_seed: int = 12345):
        self.polar = PolarQuant(dim, bits, seed=polar_seed)
        self.qjl   = QJL(dim, seed=qjl_seed)

    def compress(self, x: Tensor) -> CompressedKV:
        x_mse, residual = self.polar.compress(x)
        signs, norms    = self.qjl.sketch(residual)
        return CompressedKV(x_mse=x_mse, signs=signs, norms=norms)

    def attention_score(self, query: Tensor, ckv: CompressedKV) -> Tensor:
        mse_term      = query @ ckv.x_mse.T
        residual_term = self.qjl.estimate_inner_product(query, ckv.signs, ckv.norms)
        return mse_term + residual_term
```

**`kv_backend.py`**
```python
class KVCacheBackend:
    def __init__(self, n_layers: int, head_dim: int, bits: int = 4):
        self.compressors = [
            TurboQuantCompressor(
                dim=head_dim,
                bits=bits,
                polar_seed=42 + layer_idx,
                qjl_seed=12345 + layer_idx,
            )
            for layer_idx in range(n_layers)
        ]

    def compress(self, layer_idx: int, x: Tensor) -> CompressedKV:
        return self.compressors[layer_idx].compress(x)

    def attention_score(self, layer_idx: int, query: Tensor, ckv: CompressedKV) -> Tensor:
        return self.compressors[layer_idx].attention_score(query, ckv)
```

**Integration:** `GroupedQueryAttention.forward()` accepts optional `kv_backend: KVCacheBackend | None`. When provided, K and V are compressed before being stored; attention scores are computed via `kv_backend.attention_score()` instead of raw matmul.

### Module 4b: Best-of-N Reasoning (`src/inference/best_of_n.py`)

Generate N completions, score with a pure-Python heuristic, return the best.

```python
@dataclass
class ScoredCompletion:
    text: str
    score: float
    token_count: int

class BestOfN:
    def __init__(
        self,
        model: AureliusTransformer,
        tokenizer,
        n: int = 8,
        scorer: Callable[[str], float] | None = None,
    ):
        self.scorer = scorer or heuristic_score

    def generate(self, prompt: str, **gen_kwargs) -> ScoredCompletion:
        # encode prompt
        # run model.generate() N times (temperature > 0 for diversity)
        # decode each
        # score each with self.scorer
        # return ScoredCompletion with highest score

def heuristic_score(text: str) -> float:
    # reasoning_score (0-0.3):
    #   +0.05 per marker in {"therefore", "thus", "because", "step", "first", "finally"}
    #   capped at 0.3
    # step_score (0-0.3):
    #   count numbered steps "1." "2." or paragraph breaks "\n\n"
    #   score = min(step_count / 5, 1.0) * 0.3
    # conciseness_score (0-0.4):
    #   word_count = len(text.split())
    #   if word_count < 20: 0.0 (too short)
    #   if word_count <= 300: (word_count - 20) / 280 * 0.4
    #   if word_count > 300: max(0, 0.4 - (word_count - 300) / 500 * 0.4)
    return reasoning_score + step_score + conciseness_score
```

---

## Testing Strategy

Every module ships with a test file. No test relies on a real trained checkpoint — all tests use random weights or trivially constructable inputs.

| Module | Test file | Key assertions |
|---|---|---|
| Model core | `tests/model/test_transformer.py` | loss is scalar when labels provided, generate() shape, KV cache output == no-cache output |
| Tokenized loader | `tests/data/test_tokenized_loader.py` | shape, mmap not loaded to RAM, labels shifted, no cross-shard leakage |
| Magpie | `tests/data/test_magpie.py` | output has instruction/response keys, domain prefix appears in generation input |
| Spectrum | `tests/alignment/test_spectrum.py` | returns list of strings, all exist in model, subsampling triggers correctly |
| DoRA | `tests/alignment/test_dora.py` | m initialized to column norms, merged weight matches W_eff, grads flow to m and B@A not W |
| Regression gate | `tests/alignment/test_regression_gate.py` | passes on ppl improvement, fails on regression, archive created on failure |
| Lloyd-Max | `tests/inference/test_lloyd_max.py` | codebook boundaries monotone, centroids in (0,1), MSE < uniform quantization |
| PolarQuant | `tests/inference/test_polar_quant.py` | round-trip MSE bounded, residual = x - x_mse exactly |
| QJL | `tests/inference/test_qjl.py` | signs are ±1, inner product estimate is unbiased over 1000 pairs |
| TurboQuant | `tests/inference/test_turboquant.py` | compression ratio matches (bits+1)/16, per-layer seeds differ, attention output RMSE < 0.05 |
| Best-of-N | `tests/inference/test_best_of_n.py` | returns highest-scoring, heuristic scores reasoning higher, n=1 works |

---

## Implementation Order

1. **Group 1 (Model Core)** — must complete first. Unblocks trainer integration, generate(), and KV cache for Groups 2-4.
2. **Groups 2, 3, 4** — fully independent of each other. Can be implemented in any order or in parallel.
3. **Integration test** — after all groups: `model.generate()` → Best-of-N → KV compressed attention path works end to end with random weights.
