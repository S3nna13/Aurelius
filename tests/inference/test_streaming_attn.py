"""Tests for src/inference/streaming_attn.py — StreamingLLM attention sink + sliding window cache."""  # noqa: E501

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.streaming_attn import (
    SinkAttention,
    SinkTokenCache,
    StreamingConfig,
    compute_cache_efficiency,
    compute_position_ids_for_cache,
)

# ---------------------------------------------------------------------------
# Shared test parameters  (n_heads=4, head_dim=16, d_model=64, n_sink=2, window=8)
# ---------------------------------------------------------------------------

N_HEADS = 4
HEAD_DIM = 16
D_MODEL = 64  # must equal N_HEADS * HEAD_DIM
N_SINK = 2
WINDOW = 8

SMALL_CFG = StreamingConfig(
    n_sink_tokens=N_SINK,
    window_size=WINDOW,
    max_cache_size=N_SINK + WINDOW,  # 10
    eviction_strategy="sliding",
)


# ===========================================================================
# 1. StreamingConfig defaults
# ===========================================================================


def test_streaming_config_defaults():
    cfg = StreamingConfig()
    assert cfg.n_sink_tokens == 4
    assert cfg.window_size == 256
    assert cfg.max_cache_size == 260
    assert cfg.eviction_strategy == "sliding"


# ===========================================================================
# 2. SinkTokenCache.update returns correct shapes
# ===========================================================================


def test_sink_token_cache_update_shape():
    cache = SinkTokenCache(SMALL_CFG, n_layers=2, n_heads=N_HEADS, head_dim=HEAD_DIM)
    B, T_new = 2, 3
    k = torch.randn(B, N_HEADS, T_new, HEAD_DIM)
    v = torch.randn(B, N_HEADS, T_new, HEAD_DIM)
    ck, cv = cache.update(k, v, layer_idx=0)
    assert ck.shape == (B, N_HEADS, T_new, HEAD_DIM)
    assert cv.shape == (B, N_HEADS, T_new, HEAD_DIM)


# ===========================================================================
# 3. SinkTokenCache with small input (< window_size) keeps all tokens
# ===========================================================================


def test_sink_token_cache_small_input_keeps_all():
    cache = SinkTokenCache(SMALL_CFG, n_layers=1, n_heads=N_HEADS, head_dim=HEAD_DIM)
    # Feed 5 tokens — below window_size=8 and max_cache_size=10, should keep all
    B, T_new = 1, 5
    k = torch.randn(B, N_HEADS, T_new, HEAD_DIM)
    v = torch.randn(B, N_HEADS, T_new, HEAD_DIM)
    ck, cv = cache.update(k, v, layer_idx=0)
    assert ck.size(2) == T_new
    assert cv.size(2) == T_new


# ===========================================================================
# 4. SinkTokenCache with large input evicts middle but keeps sinks
# ===========================================================================


def test_sink_token_cache_large_input_keeps_sinks():
    cache = SinkTokenCache(SMALL_CFG, n_layers=1, n_heads=N_HEADS, head_dim=HEAD_DIM)
    B = 1
    # Feed tokens in two batches so that total > max_cache_size (10)
    # First batch: max_cache_size tokens (fills exactly)
    T_first = SMALL_CFG.max_cache_size  # 10
    k1 = torch.randn(B, N_HEADS, T_first, HEAD_DIM)
    v1 = torch.randn(B, N_HEADS, T_first, HEAD_DIM)
    # Mark the first N_SINK tokens with a distinguishable value
    k1[:, :, :N_SINK, :] = 999.0
    cache.update(k1, v1, layer_idx=0)

    # Second batch: push 3 more tokens to force eviction
    T_extra = 3
    k2 = torch.randn(B, N_HEADS, T_extra, HEAD_DIM)
    v2 = torch.randn(B, N_HEADS, T_extra, HEAD_DIM)
    ck, cv = cache.update(k2, v2, layer_idx=0)

    # Cache size must not exceed max_cache_size
    assert ck.size(2) <= SMALL_CFG.max_cache_size

    # Sink tokens (first N_SINK) must still be present (value=999)
    assert (ck[:, :, :N_SINK, :] == 999.0).all(), "Sink tokens were evicted"


# ===========================================================================
# 5. SinkTokenCache.clear resets size to 0
# ===========================================================================


def test_sink_token_cache_clear():
    cache = SinkTokenCache(SMALL_CFG, n_layers=1, n_heads=N_HEADS, head_dim=HEAD_DIM)
    k = torch.randn(1, N_HEADS, 4, HEAD_DIM)
    v = torch.randn(1, N_HEADS, 4, HEAD_DIM)
    cache.update(k, v, layer_idx=0)
    assert cache.get_cache_size() == 4
    cache.clear()
    assert cache.get_cache_size() == 0


# ===========================================================================
# 6. SinkTokenCache size never exceeds max_cache_size
# ===========================================================================


def test_sink_token_cache_size_never_exceeds_max():
    cache = SinkTokenCache(SMALL_CFG, n_layers=1, n_heads=N_HEADS, head_dim=HEAD_DIM)
    B = 1
    # Feed many small batches
    for _ in range(20):
        k = torch.randn(B, N_HEADS, 3, HEAD_DIM)
        v = torch.randn(B, N_HEADS, 3, HEAD_DIM)
        ck, cv = cache.update(k, v, layer_idx=0)
        assert ck.size(2) <= SMALL_CFG.max_cache_size


# ===========================================================================
# 7. compute_position_ids_for_cache output shape is (cache_size,)
# ===========================================================================


def test_compute_position_ids_shape():
    cache_size = 10
    pos = compute_position_ids_for_cache(seq_len=20, cache_size=cache_size, n_sink=N_SINK)
    assert pos.shape == (cache_size,)
    assert pos.dtype == torch.long


# ===========================================================================
# 8. compute_position_ids_for_cache sink positions are 0..n_sink-1
# ===========================================================================


def test_compute_position_ids_sink_values():
    cache_size = 10
    n_sink = N_SINK
    pos = compute_position_ids_for_cache(seq_len=20, cache_size=cache_size, n_sink=n_sink)
    expected_sink = torch.arange(n_sink, dtype=torch.long)
    assert torch.equal(pos[:n_sink], expected_sink)


# ===========================================================================
# 9. SinkAttention output shape (B, T, d_model)
# ===========================================================================


def test_sink_attention_output_shape():
    torch.manual_seed(0)
    attn = SinkAttention(D_MODEL, N_HEADS, HEAD_DIM, SMALL_CFG)
    B, T = 2, 5
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL)


# ===========================================================================
# 10. SinkAttention cache grows with sequential calls
# ===========================================================================


def test_sink_attention_cache_grows():
    torch.manual_seed(1)
    attn = SinkAttention(D_MODEL, N_HEADS, HEAD_DIM, SMALL_CFG)
    B = 1

    # First call: 3 tokens
    x1 = torch.randn(B, 3, D_MODEL)
    attn(x1)
    size_after_first = attn.cache.get_cache_size()
    assert size_after_first == 3

    # Second call: 3 more tokens
    x2 = torch.randn(B, 3, D_MODEL)
    attn(x2)
    size_after_second = attn.cache.get_cache_size()
    assert size_after_second == 6  # 3 + 3, still within window


# ===========================================================================
# 11. SinkAttention result differs from standard attention after long context
# ===========================================================================


def test_sink_attention_differs_from_standard_after_long_context():
    """After filling the cache, the sink module compresses context; outputs differ."""
    torch.manual_seed(42)
    cfg_large = StreamingConfig(
        n_sink_tokens=N_SINK,
        window_size=WINDOW,
        max_cache_size=N_SINK + WINDOW,
    )
    sink_attn = SinkAttention(D_MODEL, N_HEADS, HEAD_DIM, cfg_large)

    # Build a standard (full) attention for comparison using same weights
    class FullAttn(nn.Module):
        def __init__(self, ref: SinkAttention) -> None:
            super().__init__()
            self.q_proj = ref.q_proj
            self.k_proj = ref.k_proj
            self.v_proj = ref.v_proj
            self.o_proj = ref.o_proj
            self.n_heads = ref.n_heads
            self.head_dim = ref.head_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            import math

            B, T, _ = x.shape

            def sh(t):
                return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            q, k, v = sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x))
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            # causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(out)

    full_attn = FullAttn(sink_attn)

    # Feed a long sequence that exceeds max_cache_size so sink effect kicks in
    # We do it token by token through sink_attn to simulate streaming
    B = 1
    T_total = cfg_large.max_cache_size + 5  # exceeds cache
    x_seq = torch.randn(B, T_total, D_MODEL)

    # Run sink attention token by token
    sink_out_parts = []
    for t in range(T_total):
        xt = x_seq[:, t : t + 1, :]
        out_t = sink_attn(xt)
        sink_out_parts.append(out_t)
    sink_out = torch.cat(sink_out_parts, dim=1)  # (B, T_total, D_MODEL)

    # Run full attention on the whole sequence at once
    full_out = full_attn(x_seq)

    # They should differ (sink discards middle context)
    assert not torch.allclose(sink_out, full_out, atol=1e-4), (
        "Sink attention output should differ from full attention after long context"
    )


# ===========================================================================
# 12. compute_cache_efficiency compression_ratio < 1 for long sequences
# ===========================================================================


def test_cache_efficiency_compression_ratio():
    cfg = SMALL_CFG  # max_cache_size = 10
    seq_len = 100  # much longer than max_cache_size
    result = compute_cache_efficiency(cfg, seq_len)
    assert result["compression_ratio"] < 1.0
    assert result["cache_size"] == float(cfg.max_cache_size)
    assert result["full_cache_size"] == float(seq_len)


# ===========================================================================
# 13. compute_cache_efficiency sink_fraction = n_sink / max_cache_size
# ===========================================================================


def test_cache_efficiency_sink_fraction():
    cfg = SMALL_CFG
    result = compute_cache_efficiency(cfg, seq_len=50)
    expected = cfg.n_sink_tokens / cfg.max_cache_size
    assert abs(result["sink_fraction"] - expected) < 1e-9
