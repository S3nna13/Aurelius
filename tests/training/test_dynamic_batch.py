"""Tests for dynamic_batch.py — dynamic batch size finder with gradient accumulation."""
import pytest
import torch
import torch.nn as nn

from src.training.dynamic_batch import (
    DynamicBatchConfig,
    BatchScaleResult,
    try_batch_size,
    find_max_batch_size,
    compute_grad_accum_steps,
    scale_batch,
)


# ---------------------------------------------------------------------------
# Small test model — mirrors the spec: n_layers=2, d_model=64, etc.
# We use a plain nn.Embedding + nn.Linear to stay fast on CPU and avoid
# pulling in the full AureliusTransformer (which requires head_dim consistency).
# ---------------------------------------------------------------------------

class _SmallModel(nn.Module):
    """Tiny language-model stub: embedding → linear → logits."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, V)
        return self.head(self.embed(x))


def _make_model() -> _SmallModel:
    return _SmallModel(vocab_size=256, d_model=64)


def _small_cfg(**kwargs) -> DynamicBatchConfig:
    defaults = dict(
        target_global_batch_tokens=1024,
        seq_len=8,
        min_batch_size=1,
        max_batch_size=32,
        safety_factor=0.9,
        device="cpu",
    )
    defaults.update(kwargs)
    return DynamicBatchConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. try_batch_size — small succeeds
# ---------------------------------------------------------------------------

def test_try_batch_size_small_succeeds():
    model = _make_model()
    result = try_batch_size(model, batch_size=1, seq_len=8, device="cpu")
    assert result is True


# ---------------------------------------------------------------------------
# 2. try_batch_size — huge fails (OOM even on CPU)
# ---------------------------------------------------------------------------

def test_try_batch_size_huge_fails(monkeypatch):
    """Verify that try_batch_size returns False on RuntimeError (OOM simulation).

    We use monkeypatch to inject a RuntimeError during the forward pass,
    which is exactly what PyTorch raises on CPU OOM. This avoids relying on
    the host machine having insufficient memory for batch_size=99999.
    """
    import src.training.dynamic_batch as dyn

    _orig_randint = torch.randint

    call_count = [0]

    def _failing_randint(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] > 0:
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 99999 bytes")
        return _orig_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", _failing_randint)

    model = _make_model()
    result = try_batch_size(model, batch_size=99999, seq_len=32, device="cpu")
    assert result is False


# ---------------------------------------------------------------------------
# 3. compute_grad_accum_steps — basic calculation
# ---------------------------------------------------------------------------

def test_compute_grad_accum_basic():
    # target=1024, batch=4, seq=16 → tokens_per_step=64 → ceil(1024/64)=16
    steps = compute_grad_accum_steps(safe_batch_size=4, seq_len=16, target_global_batch_tokens=1024)
    assert steps == 16


# ---------------------------------------------------------------------------
# 4. compute_grad_accum_steps — minimum is 1 even when batch > target
# ---------------------------------------------------------------------------

def test_compute_grad_accum_minimum_one():
    # batch*seq already exceeds target
    steps = compute_grad_accum_steps(
        safe_batch_size=1000,
        seq_len=2048,
        target_global_batch_tokens=1024,
    )
    assert steps >= 1


# ---------------------------------------------------------------------------
# 5. find_max_batch_size — returns positive integer
# ---------------------------------------------------------------------------

def test_find_max_batch_size_returns_positive():
    model = _make_model()
    cfg = _small_cfg()
    result = find_max_batch_size(model, cfg)
    assert result >= 1


# ---------------------------------------------------------------------------
# 6. find_max_batch_size — result within configured range
# ---------------------------------------------------------------------------

def test_find_max_batch_size_within_range():
    model = _make_model()
    cfg = _small_cfg(max_batch_size=16)
    result = find_max_batch_size(model, cfg)
    assert result <= cfg.max_batch_size


# ---------------------------------------------------------------------------
# 7. scale_batch — returns a BatchScaleResult
# ---------------------------------------------------------------------------

def test_scale_batch_returns_result():
    model = _make_model()
    cfg = _small_cfg()
    result = scale_batch(model, cfg)
    assert isinstance(result, BatchScaleResult)


# ---------------------------------------------------------------------------
# 8. scale_batch — safe_batch_size <= max_batch_size
# ---------------------------------------------------------------------------

def test_scale_batch_safe_less_than_max():
    model = _make_model()
    cfg = _small_cfg()
    result = scale_batch(model, cfg)
    assert result.safe_batch_size <= result.max_batch_size


# ---------------------------------------------------------------------------
# 9. scale_batch — utilization in (0, 2.0]
# ---------------------------------------------------------------------------

def test_scale_batch_utilization_range():
    model = _make_model()
    cfg = _small_cfg()
    result = scale_batch(model, cfg)
    assert 0 < result.utilization <= 2.0


# ---------------------------------------------------------------------------
# 10. BatchScaleResult — grad_accum_steps >= 1
# ---------------------------------------------------------------------------

def test_batch_scale_result_accum_positive():
    model = _make_model()
    cfg = _small_cfg()
    result = scale_batch(model, cfg)
    assert result.grad_accum_steps >= 1
