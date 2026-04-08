import pytest
import torch
from src.model.attn_backend import (
    AttentionBackend, BackendInfo, AttentionBenchmarkResult,
    detect_available_backends, get_sdpa_backend_context,
    run_attention, benchmark_attention
)

def test_detect_backends_returns_dict():
    backends = detect_available_backends()
    assert isinstance(backends, dict)
    assert AttentionBackend.MATH in backends

def test_math_backend_always_available():
    backends = detect_available_backends()
    assert backends[AttentionBackend.MATH].available is True

def test_flash_requires_cuda():
    backends = detect_available_backends()
    flash_info = backends[AttentionBackend.FLASH]
    if not torch.cuda.is_available():
        assert flash_info.available is False

def test_run_attention_shape():
    B, H, S, D = 2, 4, 16, 32
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = run_attention(q, k, v, is_causal=True)
    assert out.shape == (B, H, S, D)

def test_run_attention_with_mask():
    B, H, S, D = 1, 2, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    mask = torch.zeros(S, S)
    out = run_attention(q, k, v, mask=mask)
    assert out.shape == (B, H, S, D)

def test_benchmark_returns_results():
    results = benchmark_attention(batch_size=1, n_heads=2, seq_len=16, head_dim=16, n_trials=2)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r, AttentionBenchmarkResult)
        assert r.time_ms > 0

def test_benchmark_output_shape():
    results = benchmark_attention(batch_size=1, n_heads=2, seq_len=8, head_dim=8, n_trials=1)
    for r in results:
        assert r.output_shape == (1, 2, 8, 8)

def test_sdpa_context_no_crash():
    """get_sdpa_backend_context should not raise even on CPU."""
    import contextlib
    ctx = get_sdpa_backend_context(AttentionBackend.MATH)
    with ctx:
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        run_attention(q, k, v)
