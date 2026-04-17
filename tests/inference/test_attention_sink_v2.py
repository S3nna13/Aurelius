"""Tests for attention_sink_v2 — StreamingLLM attention sink implementation.

Tiny configs throughout:
    n_sinks=2, window_size=4, d_model=16, n_heads=2, seq_len=8, batch=2
"""

import torch
import torch.nn as nn
import pytest

from src.inference.attention_sink_v2 import (
    AttentionSinkDetector,
    StreamingKVCache,
    SinkAwareAttention,
    SinkTokenInitializer,
    StreamingLLMDecoder,
    SinkAnalyzer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
B = 2
H = 2
T = 8
D = 16
D_HEAD = D // H  # 8
N_SINKS = 2
WIN = 4


def _rand_attn(b=B, h=H, t=T) -> torch.Tensor:
    """Return a valid (B, H, T, T) softmax attention weight matrix."""
    raw = torch.rand(b, h, t, t)
    return torch.softmax(raw, dim=-1)


def _rand_kv(b=B, h=H, t=T, d=D_HEAD) -> tuple:
    """Return (keys, values) tensors of shape (B, H, T, D_head)."""
    return torch.randn(b, h, t, d), torch.randn(b, h, t, d)


# ===========================================================================
# 1 — AttentionSinkDetector.detect: sink_mask shape (B, T) bool
# ===========================================================================
def test_detect_sink_mask_shape():
    detector = AttentionSinkDetector(n_sink_candidates=N_SINKS)
    attn = _rand_attn()
    sink_mask, _ = detector.detect(attn)
    assert sink_mask.shape == (B, T)
    assert sink_mask.dtype == torch.bool


# ===========================================================================
# 2 — AttentionSinkDetector.detect: sink_scores shape (B, T)
# ===========================================================================
def test_detect_sink_scores_shape():
    detector = AttentionSinkDetector(n_sink_candidates=N_SINKS)
    attn = _rand_attn()
    _, sink_scores = detector.detect(attn)
    assert sink_scores.shape == (B, T)
    assert sink_scores.dtype in (torch.float32, torch.float64)


# ===========================================================================
# 3 — AttentionSinkDetector: exactly n_sink_candidates positions flagged per
#     batch item
# ===========================================================================
def test_detect_top_k_flagged():
    detector = AttentionSinkDetector(n_sink_candidates=N_SINKS)
    attn = _rand_attn()
    sink_mask, _ = detector.detect(attn)
    # Each row must have exactly N_SINKS True entries
    counts = sink_mask.sum(dim=-1)  # (B,)
    assert (counts == N_SINKS).all(), f"Expected {N_SINKS} sinks per item, got {counts}"


# ===========================================================================
# 4 — StreamingKVCache.update: retained size ≤ n_sinks + window_size
# ===========================================================================
def test_kv_cache_max_size():
    cache = StreamingKVCache(n_sinks=N_SINKS, window_size=WIN)
    k, v = _rand_kv(t=T)  # 8 tokens, more than n_sinks + window_size = 6
    cache.update(k, v)
    assert cache.size() <= N_SINKS + WIN


# ===========================================================================
# 5 — StreamingKVCache: first n_sinks tokens always retained
# ===========================================================================
def test_kv_cache_sinks_retained():
    cache = StreamingKVCache(n_sinks=N_SINKS, window_size=WIN)
    k, v = _rand_kv(t=T)
    kept_k, _ = cache.update(k, v)
    # Sink portion of kept cache must match original first n_sinks positions
    assert torch.allclose(kept_k[:, :, :N_SINKS, :], k[:, :, :N_SINKS, :])


# ===========================================================================
# 6 — StreamingKVCache.reset: size becomes 0
# ===========================================================================
def test_kv_cache_reset():
    cache = StreamingKVCache(n_sinks=N_SINKS, window_size=WIN)
    k, v = _rand_kv()
    cache.update(k, v)
    cache.reset()
    assert cache.size() == 0


# ===========================================================================
# 7 — SinkAwareAttention: output shape (B, T, D)
# ===========================================================================
def test_sink_attention_output_shape():
    attn = SinkAwareAttention(d_model=D, n_heads=H, n_sinks=N_SINKS, window_size=WIN)
    x = torch.randn(B, T, D)
    out = attn(x)
    assert out.shape == (B, T, D)


# ===========================================================================
# 8 — SinkAwareAttention: gradients flow back through output
# ===========================================================================
def test_sink_attention_grad_flows():
    attn = SinkAwareAttention(d_model=D, n_heads=H, n_sinks=N_SINKS, window_size=WIN)
    x = torch.randn(B, T, D, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


# ===========================================================================
# 9 — SinkAwareAttention: window_size truncation maintains output shape
# ===========================================================================
def test_sink_attention_long_sequence_shape():
    """Feed more tokens than n_sinks + window_size; output shape must still
    match the input sequence length."""
    attn = SinkAwareAttention(d_model=D, n_heads=H, n_sinks=N_SINKS, window_size=WIN)
    long_T = N_SINKS + WIN + 4  # exceeds cache capacity
    x = torch.randn(B, long_T, D)
    out = attn(x)
    assert out.shape == (B, long_T, D)


# ===========================================================================
# 10 — SinkTokenInitializer.prepend: output shape (B, n_sinks + T, D)
# ===========================================================================
def test_sink_initializer_prepend_shape():
    init = SinkTokenInitializer(n_sinks=N_SINKS, d_model=D)
    x = torch.randn(B, T, D)
    out = init.prepend(x)
    assert out.shape == (B, N_SINKS + T, D)


# ===========================================================================
# 11 — SinkTokenInitializer.sink_loss: scalar, ≤ 0
# ===========================================================================
def test_sink_initializer_loss_scalar_nonpositive():
    init = SinkTokenInitializer(n_sinks=N_SINKS, d_model=D)
    attn = _rand_attn()  # (B, H, T, T)
    loss = init.sink_loss(attn)
    assert loss.dim() == 0, "sink_loss must be a scalar"
    assert float(loss.item()) <= 0.0, f"sink_loss must be ≤ 0, got {loss.item()}"


# ===========================================================================
# 12 — StreamingLLMDecoder.generate: output shape (B, T + max_new_tokens),
#      valid vocab ids
# ===========================================================================
def test_decoder_generate_shape_and_vocab():
    vocab_size = 32
    max_new = 3

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, D)
            self.proj  = nn.Linear(D, vocab_size)

        def forward(self, ids):
            return self.proj(self.embed(ids))

    model = TinyLM()
    decoder = StreamingLLMDecoder(model, n_sinks=N_SINKS, window_size=WIN)
    input_ids = torch.randint(0, vocab_size, (B, T))
    out = decoder.generate(input_ids, max_new_tokens=max_new)
    assert out.shape == (B, T + max_new)
    assert (out >= 0).all() and (out < vocab_size).all()


# ===========================================================================
# 13 — SinkAnalyzer.sink_ratio: in [0, 1]
# ===========================================================================
def test_analyzer_sink_ratio_range():
    analyzer = SinkAnalyzer()
    attn = _rand_attn()
    ratio = analyzer.sink_ratio(attn, n_sinks=N_SINKS)
    assert 0.0 <= ratio <= 1.0, f"sink_ratio out of range: {ratio}"


# ===========================================================================
# 14 — SinkAnalyzer.sink_stability: ≥ 0
# ===========================================================================
def test_analyzer_sink_stability_nonneg():
    analyzer = SinkAnalyzer()
    layers = [_rand_attn() for _ in range(4)]
    stab = analyzer.sink_stability(layers)
    assert stab >= 0.0, f"sink_stability must be ≥ 0, got {stab}"


# ===========================================================================
# 15a — SinkAnalyzer.window_coverage: in [0, 1]
# 15b — = 1 when window covers everything
# ===========================================================================
def test_analyzer_window_coverage():
    analyzer = SinkAnalyzer()

    # General case: partial coverage
    cov = analyzer.window_coverage(total_tokens=20, n_sinks=2, window_size=4)
    assert 0.0 <= cov <= 1.0, f"window_coverage out of range: {cov}"

    # Edge case: window exactly covers all tokens → coverage = 1
    cov_full = analyzer.window_coverage(total_tokens=6, n_sinks=2, window_size=4)
    assert cov_full == 1.0, f"Expected 1.0 for full coverage, got {cov_full}"

    # When window > total, still clamped to 1.0
    cov_over = analyzer.window_coverage(total_tokens=3, n_sinks=2, window_size=4)
    assert cov_over == 1.0


# ===========================================================================
# 16 — StreamingKVCache with many updates (> window_size): size stays bounded
# ===========================================================================
def test_kv_cache_bounded_after_many_updates():
    cache = StreamingKVCache(n_sinks=N_SINKS, window_size=WIN)
    for _ in range(10):  # 10 × single-token updates
        k, v = _rand_kv(t=1)
        cache.update(k, v)
    assert cache.size() <= N_SINKS + WIN, (
        f"Cache size {cache.size()} exceeds bound {N_SINKS + WIN}"
    )


# ===========================================================================
# 17 — Sink tokens always kept through 5 consecutive updates
# ===========================================================================
def test_kv_cache_sinks_persist_across_updates():
    cache = StreamingKVCache(n_sinks=N_SINKS, window_size=WIN)
    # First update plants the sinks
    k0, v0 = _rand_kv(t=N_SINKS + 1)
    kept_k0, _ = cache.update(k0, v0)
    sink_snapshot = kept_k0[:, :, :N_SINKS, :].clone()

    # Five more updates (each adds 1 new token)
    for _ in range(5):
        k_new, v_new = _rand_kv(t=1)
        kept_k, _ = cache.update(k_new, v_new)

    # Sink portion must remain identical
    assert torch.allclose(kept_k[:, :, :N_SINKS, :], sink_snapshot), (
        "Sink KV pairs changed across updates!"
    )
