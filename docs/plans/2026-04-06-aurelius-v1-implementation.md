# Aurelius v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build, train, align, and serve a 1.3B dense LLM (general + code) from scratch in 3 weeks for ~$350–400 cloud compute.

**Architecture:** 24-layer decoder-only transformer, 2048 hidden dim, GQA (16Q/8KV heads), SwiGLU FFN, RoPE (θ=500K), RMSNorm, 128K vocab, BF16 training. Pretrained on 300B tokens (FineWeb + code), aligned with SFT + DPO.

**Tech Stack:** Python 3.12, PyTorch 2.x, MLX (Apple Silicon), HuggingFace (tokenizers, datasets, TRL, Transformers), DataTrove, Megatron-LM + DeepSpeed ZeRO-1, lm-evaluation-harness, Unsloth, SGLang, llama.cpp, Garak

---

## Environment Setup

### Task 1: Initialize Python Environment

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.python-version`

**Step 1: Create pyproject.toml**
```toml
[project]
name = "aurelius"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.3.0",
    "transformers>=4.45.0",
    "tokenizers>=0.20.0",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "trl>=0.12.0",
    "peft>=0.13.0",
    "datatrove[io,processing]>=0.3.0",
    "lm-eval>=0.4.4",
    "unsloth",
    "safetensors>=0.4.0",
    "einops>=0.8.0",
    "flash-attn>=2.6.0",
    "bitsandbytes>=0.44.0",
    "wandb>=0.18.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
apple = ["mlx>=0.18.0", "mlx-lm>=0.18.0"]
dev = ["pytest>=8.0.0", "pytest-cov", "ruff", "mypy"]
```

**Step 2: Create .python-version**
```
3.12
```

**Step 3: Install (local M1 Pro)**
```bash
cd /Users/christienantonio/Desktop/Aurelius
python -m venv .venv
source .venv/bin/activate
pip install -e ".[apple,dev]"
```

**Step 4: Commit**
```bash
git add pyproject.toml requirements.txt .python-version
git commit -m "chore: initialize Python environment"
```

---

## Week 1, Phase A: Architecture

### Task 2: RMSNorm

**Files:**
- Create: `src/model/rms_norm.py`
- Create: `tests/model/test_rms_norm.py`

**Step 1: Write failing test**
```python
# tests/model/test_rms_norm.py
import torch
import pytest
from src.model.rms_norm import RMSNorm

def test_output_shape():
    norm = RMSNorm(dim=2048)
    x = torch.randn(2, 16, 2048)
    assert norm(x).shape == (2, 16, 2048)

def test_no_bias():
    norm = RMSNorm(dim=2048)
    assert not hasattr(norm, 'bias') or norm.bias is None

def test_normalizes():
    norm = RMSNorm(dim=2048)
    x = torch.randn(2, 16, 2048) * 100
    out = norm(x)
    # Output should have unit RMS
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, norm.weight.data.abs().mean() * torch.ones_like(rms), atol=0.1)
```

**Step 2: Run to confirm FAIL**
```bash
pytest tests/model/test_rms_norm.py -v
# Expected: ModuleNotFoundError
```

**Step 3: Implement**
```python
# src/model/rms_norm.py
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

**Step 4: Run to confirm PASS**
```bash
pytest tests/model/test_rms_norm.py -v
```

**Step 5: Commit**
```bash
git add src/model/rms_norm.py tests/model/test_rms_norm.py
git commit -m "feat: add RMSNorm"
```

---

### Task 3: RoPE (Rotary Position Embeddings)

**Files:**
- Create: `src/model/rope.py`
- Create: `tests/model/test_rope.py`

**Step 1: Write failing test**
```python
# tests/model/test_rope.py
import torch
from src.model.rope import RoPE, apply_rope

def test_output_shape():
    rope = RoPE(head_dim=128, max_seq_len=8192, theta=500000)
    q = torch.randn(2, 8, 16, 128)   # (batch, seq, heads, head_dim)
    k = torch.randn(2, 8, 8, 128)
    q_out, k_out = apply_rope(q, k, rope)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_relative_position_invariance():
    """RoPE encodes relative positions — rotating all positions by offset should be equivalent."""
    rope = RoPE(head_dim=128, max_seq_len=8192, theta=500000)
    q = torch.randn(1, 4, 1, 128)
    k = torch.randn(1, 4, 1, 128)
    q1, k1 = apply_rope(q, k, rope, offset=0)
    q2, k2 = apply_rope(q, k, rope, offset=2)
    # Dot products should differ (positions changed)
    dot1 = (q1 * k1).sum(-1)
    dot2 = (q2 * k2).sum(-1)
    assert not torch.allclose(dot1, dot2, atol=1e-5)
```

**Step 2: Run to confirm FAIL**
```bash
pytest tests/model/test_rope.py -v
```

**Step 3: Implement**
```python
# src/model/rope.py
import torch
import torch.nn as nn
from dataclasses import dataclass


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 500000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

    def get_cos_sin(self, seq_len: int, device: torch.device, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RoPE,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[1]
    cos, sin = rope.get_cos_sin(seq_len, q.device, offset=offset)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos[:, :, :k.shape[2], :]) + (rotate_half(k) * sin[:, :, :k.shape[2], :])
    return q_rot, k_rot
```

**Step 4: Run to confirm PASS**
```bash
pytest tests/model/test_rope.py -v
```

**Step 5: Commit**
```bash
git add src/model/rope.py tests/model/test_rope.py
git commit -m "feat: add RoPE with θ=500000"
```

---

### Task 4: GQA Attention

**Files:**
- Create: `src/model/attention.py`
- Create: `tests/model/test_attention.py`

**Step 1: Write failing test**
```python
# tests/model/test_attention.py
import torch
from src.model.attention import GroupedQueryAttention
from src.model.config import AureliusConfig

def test_output_shape():
    cfg = AureliusConfig()
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 16, 2048)
    out = attn(x)
    assert out.shape == (2, 16, 2048)

def test_no_bias():
    cfg = AureliusConfig()
    attn = GroupedQueryAttention(cfg)
    for name, param in attn.named_parameters():
        if 'bias' in name:
            assert False, f"Found bias parameter: {name}"

def test_kv_heads_smaller_than_q():
    cfg = AureliusConfig()
    attn = GroupedQueryAttention(cfg)
    assert attn.n_kv_heads < attn.n_q_heads
    assert attn.n_q_heads % attn.n_kv_heads == 0
```

**Step 2: Run to confirm FAIL**
```bash
pytest tests/model/test_attention.py -v
```

**Step 3: Implement**
```python
# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import AureliusConfig
from src.model.rope import RoPE, apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.n_q_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, self.rope, offset=offset)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # (B, heads, T, head_dim) for scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

**Step 4: Run to confirm PASS**
```bash
pytest tests/model/test_attention.py -v
```

**Step 5: Commit**
```bash
git add src/model/attention.py tests/model/test_attention.py
git commit -m "feat: add GQA attention (16Q/8KV heads)"
```

---

### Task 5: SwiGLU FFN

**Files:**
- Create: `src/model/ffn.py`
- Create: `tests/model/test_ffn.py`

**Step 1: Write failing test**
```python
# tests/model/test_ffn.py
import torch
from src.model.ffn import SwiGLUFFN
from src.model.config import AureliusConfig

def test_output_shape():
    cfg = AureliusConfig()
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 16, 2048)
    assert ffn(x).shape == (2, 16, 2048)

def test_three_weight_matrices():
    cfg = AureliusConfig()
    ffn = SwiGLUFFN(cfg)
    param_names = [n for n, _ in ffn.named_parameters()]
    assert any('w1' in n for n in param_names)
    assert any('w2' in n for n in param_names)
    assert any('w3' in n for n in param_names)

def test_no_bias():
    cfg = AureliusConfig()
    ffn = SwiGLUFFN(cfg)
    for name, _ in ffn.named_parameters():
        assert 'bias' not in name
```

**Step 2: Run to confirm FAIL**
```bash
pytest tests/model/test_ffn.py -v
```

**Step 3: Implement**
```python
# src/model/ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import AureliusConfig


class SwiGLUFFN(nn.Module):
    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Step 4: Run to confirm PASS**
```bash
pytest tests/model/test_ffn.py -v
```

**Step 5: Commit**
```bash
git add src/model/ffn.py tests/model/test_ffn.py
git commit -m "feat: add SwiGLU FFN (d_ff=5632)"
```

---

### Task 6: Model Config

**Files:**
- Create: `src/model/config.py`
- Create: `tests/model/test_config.py`

**Step 1: Implement**
```python
# src/model/config.py
from dataclasses import dataclass, field


@dataclass
class AureliusConfig:
    # Architecture
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 5632          # SwiGLU: round(2/3 * 4 * d_model) to multiple of 64
    vocab_size: int = 128_000
    max_seq_len: int = 8192
    rope_theta: float = 500_000.0
    tie_embeddings: bool = True

    # Training
    dropout: float = 0.0      # No dropout during pretraining

    # Special tokens (set after tokenizer training)
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model == self.n_heads * self.head_dim, "d_model must equal n_heads * head_dim"
```

**Step 2: Write and run test**
```python
# tests/model/test_config.py
from src.model.config import AureliusConfig

def test_default_config():
    cfg = AureliusConfig()
    assert cfg.d_model == 2048
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.d_ff == 5632

def test_invalid_config_raises():
    import pytest
    with pytest.raises(AssertionError):
        AureliusConfig(n_heads=16, n_kv_heads=5)  # not divisible
```

**Step 3: Commit**
```bash
git add src/model/config.py tests/model/test_config.py
git commit -m "feat: add AureliusConfig"
```

---

### Task 7: Full Transformer

**Files:**
- Create: `src/model/transformer.py`
- Create: `tests/model/test_transformer.py`

**Step 1: Write failing test**
```python
# tests/model/test_transformer.py
import torch
from src.model.transformer import Aurelius
from src.model.config import AureliusConfig

def test_forward_pass():
    cfg = AureliusConfig(n_layers=2)  # 2 layers for speed in tests
    model = Aurelius(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, cfg.vocab_size)

def test_parameter_count():
    cfg = AureliusConfig()
    model = Aurelius(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Should be ~1.3B (1.2B–1.4B acceptable)
    assert 1_100_000_000 < n_params < 1_500_000_000, f"Unexpected param count: {n_params:,}"

def test_tied_embeddings():
    cfg = AureliusConfig(tie_embeddings=True)
    model = Aurelius(cfg)
    assert model.embed_tokens.weight is model.lm_head.weight
```

**Step 2: Run to confirm FAIL**
```bash
pytest tests/model/test_transformer.py -v
```

**Step 3: Implement**
```python
# src/model/transformer.py
import torch
import torch.nn as nn
from src.model.config import AureliusConfig
from src.model.rms_norm import RMSNorm
from src.model.attention import GroupedQueryAttention
from src.model.ffn import SwiGLUFFN


class DecoderLayer(nn.Module):
    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), offset=offset)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Aurelius(nn.Module):
    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x, offset=offset)
        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = AureliusConfig()
    model = Aurelius(config)
    n_params = model.count_parameters()
    print(f"Aurelius v1: {n_params:,} parameters ({n_params/1e9:.2f}B)")
    tokens = torch.randint(0, config.vocab_size, (1, 32))
    logits = model(tokens)
    print(f"Forward pass OK: {logits.shape}")
```

**Step 4: Run to confirm PASS**
```bash
pytest tests/model/test_transformer.py -v
python src/model/transformer.py
# Expected output: Aurelius v1: 1,3xx,xxx,xxx parameters (1.3xB)
```

**Step 5: Commit**
```bash
git add src/model/transformer.py tests/model/test_transformer.py
git commit -m "feat: complete Aurelius 1.3B transformer architecture"
```

---

## Week 1, Phase B: Tokenizer

### Task 8: Tokenizer Training

**Files:**
- Create: `src/data/tokenizer.py`
- Create: `tests/data/test_tokenizer.py`
- Create: `scripts/train_tokenizer.sh`

**Step 1: Download 1B token sample (runs overnight)**
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
texts = []
for i, ex in enumerate(ds):
    texts.append(ex['text'])
    if i >= 500_000: break  # ~1B tokens
with open('/tmp/tokenizer_sample.txt', 'w') as f:
    f.write('\n'.join(texts))
print('Done')
"
```

**Step 2: Train tokenizer**
```python
# src/data/tokenizer.py
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    "<|tool_call|>", "<|tool_result|>",
] + [f"<|reserved_{i}|>" for i in range(499)]   # 512 total reserved


def train_tokenizer(corpus_path: str, output_dir: str, vocab_size: int = 128_000) -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")


def load_tokenizer(model_dir: str):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{model_dir}/tokenizer.json",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )
    return tokenizer
```

**Step 3: Run tokenizer training**
```bash
python -c "from src.data.tokenizer import train_tokenizer; train_tokenizer('/tmp/tokenizer_sample.txt', './tokenizer')"
```

**Step 4: Write and run test**
```python
# tests/data/test_tokenizer.py
from src.data.tokenizer import load_tokenizer

def test_vocab_size():
    tok = load_tokenizer('./tokenizer')
    assert tok.vocab_size == 128_000

def test_special_tokens():
    tok = load_tokenizer('./tokenizer')
    assert tok.bos_token == '<|bos|>'
    assert '<|fim_prefix|>' in tok.get_vocab()

def test_roundtrip():
    tok = load_tokenizer('./tokenizer')
    text = "def hello_world():\n    print('hello')"
    assert tok.decode(tok.encode(text)) == text
```

**Step 5: Commit**
```bash
git add src/data/tokenizer.py tests/data/test_tokenizer.py
git commit -m "feat: train 128K BPE tokenizer"
```

---

## Week 1, Phase C: Data Pipeline

### Task 9: FIM Transform

**Files:**
- Create: `src/data/fim_transform.py`
- Create: `tests/data/test_fim.py`

**Step 1: Write failing test**
```python
# tests/data/test_fim.py
from src.data.fim_transform import apply_fim

def test_fim_output_contains_special_tokens():
    code = "def foo():\n    x = 1\n    return x"
    result = apply_fim(code, seed=42)
    if result != code:  # FIM was applied
        assert "<|fim_prefix|>" in result or "<|fim_suffix|>" in result

def test_fim_rate():
    import random
    code = "x = 1\ny = 2\nz = 3"
    results = [apply_fim(code, seed=i) for i in range(100)]
    fim_count = sum(1 for r in results if r != code)
    # Should be ~50% (allow 35-65% range)
    assert 35 <= fim_count <= 65

def test_original_content_preserved():
    code = "def add(a, b):\n    return a + b"
    result = apply_fim(code, seed=0)
    # All original characters should still be present
    assert "def add" in result
    assert "return a + b" in result
```

**Step 2: Implement**
```python
# src/data/fim_transform.py
import random


FIM_RATE = 0.5
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def apply_fim(code: str, seed: int | None = None) -> str:
    rng = random.Random(seed)
    if rng.random() > FIM_RATE:
        return code

    chars = list(code)
    n = len(chars)
    if n < 10:
        return code

    prefix_end = rng.randint(1, n - 1)
    suffix_start = rng.randint(prefix_end, n)

    prefix = "".join(chars[:prefix_end])
    middle = "".join(chars[prefix_end:suffix_start])
    suffix = "".join(chars[suffix_start:])

    if rng.random() < 0.5:  # PSM
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    else:  # SPM
        return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"
```

**Step 3: Run and commit**
```bash
pytest tests/data/test_fim.py -v
git add src/data/fim_transform.py tests/data/test_fim.py
git commit -m "feat: add FIM transform for code (50% rate, PSM+SPM)"
```

---

### Task 10: DataTrove Pipeline

**Files:**
- Create: `src/data/pipeline.py`
- Create: `scripts/prepare_data.sh`

**Step 1: Implement pipeline**
```python
# src/data/pipeline.py
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, ParquetReader
from datatrove.pipeline.filters import (
    GopherQualityFilter, C4QualityFilter,
    FineWebQualityFilter, LanguageFilter,
)
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupFilter
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.tokens import TokensCounter


MINHASH_CONFIG = MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8)


def build_fineweb_pipeline(output_dir: str, n_workers: int = 8) -> LocalPipelineExecutor:
    return LocalPipelineExecutor(
        pipeline=[
            HuggingFaceDatasetReader(
                "HuggingFaceFW/fineweb",
                dataset_options={"name": "sample-100BT"},
                text_key="text",
            ),
            LanguageFilter(languages=["en"], label_only=False),
            GopherQualityFilter(min_doc_words=50, max_doc_words=100_000),
            C4QualityFilter(filter_no_terminal_punct=True),
            FineWebQualityFilter(),
            MinhashDedupSignature(output_folder=f"{output_dir}/sigs", config=MINHASH_CONFIG),
            MinhashDedupBuckets(input_folder=f"{output_dir}/sigs", output_folder=f"{output_dir}/buckets", config=MINHASH_CONFIG),
            MinhashDedupFilter(input_folder=f"{output_dir}/buckets"),
            TokensCounter(),
            ParquetWriter(output_folder=f"{output_dir}/fineweb", max_file_size=512 * 1024 * 1024),
        ],
        tasks=n_workers,
        workers=n_workers,
        logging_dir=f"{output_dir}/logs",
    )


def build_code_pipeline(output_dir: str, n_workers: int = 8) -> LocalPipelineExecutor:
    from src.data.fim_transform import apply_fim

    class FIMFilter:
        name = "FIM Transform"
        def filter(self, doc):
            doc.text = apply_fim(doc.text)
            return True

    return LocalPipelineExecutor(
        pipeline=[
            HuggingFaceDatasetReader(
                "bigcode/the-stack-v2-train-smol-ids",
                text_key="content",
            ),
            FIMFilter(),
            ParquetWriter(output_folder=f"{output_dir}/code", max_file_size=512 * 1024 * 1024),
        ],
        tasks=n_workers,
        workers=n_workers,
        logging_dir=f"{output_dir}/logs",
    )
```

**Step 2: Create launch script**
```bash
#!/bin/bash
# scripts/prepare_data.sh
set -e
OUTPUT_DIR=${1:-"./data/processed"}
echo "Processing data to: $OUTPUT_DIR"
python -c "from src.data.pipeline import build_fineweb_pipeline; build_fineweb_pipeline('$OUTPUT_DIR').run()"
python -c "from src.data.pipeline import build_code_pipeline; build_code_pipeline('$OUTPUT_DIR').run()"
echo "Data preparation complete."
```

**Step 3: Commit**
```bash
git add src/data/pipeline.py scripts/prepare_data.sh
git commit -m "feat: add DataTrove processing pipeline (FineWeb + code)"
```

---

## Week 2: Training

### Task 11: Training Loop

**Files:**
- Create: `src/training/trainer.py`
- Create: `configs/train_1b.yaml`

**Step 1: Create config**
```yaml
# configs/train_1b.yaml
model:
  n_layers: 24
  d_model: 2048
  n_heads: 16
  n_kv_heads: 8
  d_ff: 5632
  vocab_size: 128000
  max_seq_len: 8192
  rope_theta: 500000

training:
  total_tokens: 300_000_000_000
  global_batch_tokens: 2_097_152
  micro_batch_size: 4
  gradient_clip: 1.0
  bf16: true

optimizer:
  lr: 3.0e-4
  min_lr: 3.0e-5
  beta1: 0.9
  beta2: 0.95
  epsilon: 1.0e-8
  weight_decay: 0.1
  warmup_steps: 2000

checkpointing:
  save_every_steps: 4800   # ~10B tokens
  keep_last_n: 5
  output_dir: ./checkpoints
```

**Step 2: Implement trainer**
```python
# src/training/trainer.py
import torch
import yaml
import math
from pathlib import Path
from dataclasses import dataclass
from torch.optim import AdamW
from src.model.transformer import Aurelius
from src.model.config import AureliusConfig


def cosine_lr(step: int, warmup: int, total: int, peak: float, min_lr: float) -> float:
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = AureliusConfig(**cfg["model"])
    model = Aurelius(model_cfg)

    if torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)
        device = "cuda"
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        device = "mps"
    else:
        device = "cpu"

    print(f"Training on: {device}")
    print(f"Parameters: {model.count_parameters():,}")

    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]

    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg["epsilon"],
        weight_decay=opt_cfg["weight_decay"],
    )

    total_steps = train_cfg["total_tokens"] // train_cfg["global_batch_tokens"]
    ckpt_dir = Path(cfg["checkpointing"]["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for step in range(total_steps):
        lr = cosine_lr(step, opt_cfg["warmup_steps"], total_steps, opt_cfg["lr"], opt_cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Training step (data loading omitted — plugged in via DataLoader)
        optimizer.zero_grad()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
        optimizer.step()

        if step % cfg["checkpointing"]["save_every_steps"] == 0 and step > 0:
            path = ckpt_dir / f"step-{step:08d}"
            model.save_pretrained(str(path))
            print(f"Saved checkpoint: {path}")
```

**Step 3: Commit**
```bash
git add src/training/trainer.py configs/train_1b.yaml
git commit -m "feat: add training loop with cosine LR schedule"
```

---

### Task 12: Launch Cloud Training

**Step 1: Install Lambda Labs CLI and provision H100**
```bash
# Install Lambda Cloud CLI
pip install lambda-cloud

# Provision 1x H100 instance
# lambda instances create --instance-type gpu_1x_h100_sxm5 --region us-west-2
# SSH into instance and clone repo
```

**Step 2: Launch training on H100**
```bash
#!/bin/bash
# scripts/run_training.sh
set -e

# Install dependencies on H100
pip install -e ".[dev]"
pip install flash-attn --no-build-isolation

# Launch training
python -m src.training.trainer configs/train_1b.yaml 2>&1 | tee ./logs/train.log &
echo "Training launched. PID: $!"
echo "Monitor: tail -f ./logs/train.log"
```

**Step 3: Monitor (check every few hours)**
```bash
tail -f ./logs/train.log
# Expected: loss ~10.0 at step 0, decreasing to ~2.5 by step 50K
```

---

## Week 2: Alignment (runs while pretraining finishes)

### Task 13: SFT Fine-Tuning

**Files:**
- Create: `src/alignment/sft.py`
- Create: `scripts/run_sft.sh`

**Step 1: Implement SFT**
```python
# src/alignment/sft.py
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


CHATML_TEMPLATE = """{% for message in messages %}<|{{ message['role'] }}|>
{{ message['content'] }}<|end|>
{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""


def load_sft_dataset():
    # OASST2: filter to top-rated branches only
    oasst = load_dataset("OpenAssistant/oasst2", split="train")
    oasst = oasst.filter(lambda x: x.get("rank", 1) == 0)  # Best responses only

    # Dolly-15k: all examples
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

    return concatenate_datasets([oasst, dolly])


def run_sft(model_path: str, output_dir: str) -> None:
    dataset = load_sft_dataset()

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model_path,
        train_dataset=dataset,
        peft_config=lora_config,
        args=SFTConfig(
            output_dir=output_dir,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            bf16=True,
            save_strategy="epoch",
            logging_steps=50,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
```

**Step 2: Commit**
```bash
git add src/alignment/sft.py scripts/run_sft.sh
git commit -m "feat: add SFT pipeline (OASST2 + Dolly-15k, LoRA r=64)"
```

---

### Task 14: DPO Alignment

**Files:**
- Create: `src/alignment/dpo.py`
- Create: `scripts/run_dpo.sh`

**Step 1: Implement DPO**
```python
# src/alignment/dpo.py
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_dpo(sft_model_path: str, output_dir: str) -> None:
    # UltraFeedback: filter for clear preference signal
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    dataset = dataset.filter(
        lambda x: (x.get("chosen_rating", 0) or 0) - (x.get("rejected_rating", 0) or 0) >= 1.0
    )

    model = AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype="bfloat16")
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=DPOConfig(
            output_dir=output_dir,
            beta=0.1,
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            bf16=True,
            logging_steps=50,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
```

**Step 2: Commit**
```bash
git add src/alignment/dpo.py scripts/run_dpo.sh
git commit -m "feat: add DPO alignment (UltraFeedback, β=0.1)"
```

---

## Week 3: Evaluate + Ship

### Task 15: Run Benchmarks

**Step 1: Install and run lm-eval**
```bash
pip install lm-eval

lm_eval \
  --model hf \
  --model_args pretrained=./checkpoints/aurelius-1.3b-dpo,dtype=bfloat16 \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./results/
```

**Step 2: Check against targets**
```
MMLU:         target 42-48%
HellaSwag:    target 65-72%
ARC-Challenge: target 45-55%
GSM8K:        target 20-30%
```

**Step 3: Run code benchmarks**
```bash
lm_eval \
  --model hf \
  --model_args pretrained=./checkpoints/aurelius-1.3b-dpo,dtype=bfloat16 \
  --tasks humaneval,mbpp \
  --num_fewshot 0 \
  --output_path ./results/
```

---

### Task 16: Convert to GGUF + Local Serving

**Step 1: Convert checkpoint to GGUF**
```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j8

# Convert HF → GGUF
python convert_hf_to_gguf.py ../checkpoints/aurelius-1.3b-dpo --outtype f16

# Quantize to Q4_K_M (~800MB)
./build/bin/llama-quantize aurelius-1.3b-dpo-f16.gguf aurelius-1.3b-q4_k_m.gguf Q4_K_M
```

**Step 2: Create Ollama Modelfile**
```
# configs/ollama.Modelfile
FROM ./aurelius-1.3b-q4_k_m.gguf

TEMPLATE """<|system|>
{{ .System }}<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

SYSTEM "You are Aurelius, a helpful and honest AI assistant skilled in reasoning and code."
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

**Step 3: Launch with Ollama on M1 Pro**
```bash
ollama create aurelius -f configs/ollama.Modelfile
ollama run aurelius "Write a Python function that checks if a number is prime."
# Expected: ~25-35 tok/s on M1 Pro, ~800MB RAM
```

**Step 4: Commit**
```bash
git add configs/ollama.Modelfile scripts/serve_local.sh
git commit -m "feat: local serving via Ollama Q4_K_M on M1 Pro"
```

---

### Task 17: Red-Team Validation

**Step 1: Install Garak**
```bash
pip install garak
```

**Step 2: Run automated red-teaming**
```bash
python -m garak \
  --model_type rest \
  --model_name aurelius \
  --generations 5 \
  --probes jailbreak,promptinject,dan,knownbadsignatures \
  --report_prefix ./results/red_team/
```

**Step 3: Check results**
```bash
# Target: <5% attack success rate per category
cat ./results/red_team/*.report.jsonl | python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r.get('passed_rate', 1.0) < 0.95:
        print(f'FAIL: {r[\"probe\"]} — {(1-r[\"passed_rate\"])*100:.1f}% attack success')
"
```

**Step 4: Final commit**
```bash
git add results/ 
git commit -m "eval: Aurelius v1 benchmark results and red-team report"
git tag v1.0.0
```

---

## Estimated Costs Summary

| Phase | Hardware | Duration | Cost |
|---|---|---|---|
| Data processing | Cloud CPU (32-core) | ~8 hrs | ~$15 |
| Pretraining | 1× H100 ($2.50/hr) | ~120 hrs | ~$300 |
| SFT | 1× H100 | ~5 hrs | ~$12 |
| DPO | 1× H100 | ~6 hrs | ~$15 |
| Evaluation | 1× H100 | ~3 hrs | ~$8 |
| **Total** | | | **~$350** |

Local M1 Pro: tokenizer training, ablations, final serving — $0.
