"""Tests for memory-efficient chunked attention implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.memory_efficient_attn import (
    MemoryEfficientAttention,
    chunked_attention,
    compute_attention_memory,
    online_softmax_update,
)

# ---------------------------------------------------------------------------
# Test dimensions (kept small for speed)
# ---------------------------------------------------------------------------
B = 2  # batch size
T = 16  # sequence length
H = 4  # number of heads
D = 32  # head dimension
CHUNK = 4  # chunk size for tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reference_attention(
    q: torch.Tensor,  # (B, H, T_q, D)
    k: torch.Tensor,  # (B, H, T_k, D)
    v: torch.Tensor,  # (B, H, T_k, D)
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard full-matrix attention for reference."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = scale * q @ k.transpose(-2, -1)  # (B, H, T_q, T_k)
    if causal:
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(T_q, T_k, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def make_small_config(n_heads: int = H, n_kv_heads: int = H) -> AureliusConfig:
    """Create a small AureliusConfig for testing."""
    d_model = n_heads * D
    return AureliusConfig(
        n_layers=2,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=D,
        d_ff=d_model * 2,
        vocab_size=1000,
        max_seq_len=128,
    )


# ---------------------------------------------------------------------------
# 1. online_softmax_update matches standard softmax
# ---------------------------------------------------------------------------


def test_online_softmax_matches_standard():
    """Chunked online softmax should match a single-pass full softmax."""
    torch.manual_seed(42)
    T_k = 8
    scores_all = torch.randn(B, H, T, T_k)
    v_all = torch.randn(B, H, T_k, D)

    # Reference: full softmax in one shot
    ref = F.softmax(scores_all, dim=-1) @ v_all  # (B, H, T, D)

    # Online accumulation in two chunks
    half = T_k // 2
    running_max = torch.full((B, H, T), float("-inf"))
    running_sum = torch.zeros(B, H, T)
    running_out = torch.zeros(B, H, T, D)

    for start in range(0, T_k, half):
        s_chunk = scores_all[:, :, :, start : start + half]
        v_chunk = v_all[:, :, start : start + half, :]
        running_max, running_sum, running_out = online_softmax_update(
            running_max, running_sum, running_out, s_chunk, v_chunk
        )

    result = running_out / running_sum.unsqueeze(-1)
    assert torch.allclose(result, ref, atol=1e-5), f"Max diff: {(result - ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# 2. chunked_attention output shape
# ---------------------------------------------------------------------------


def test_chunked_attention_output_shape():
    """chunked_attention should return (B, H, T_q, D) for (B, H, T_q, D) inputs."""
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = chunked_attention(q, k, v, chunk_size=CHUNK, causal=False)
    assert out.shape == (B, H, T, D), f"Expected ({B}, {H}, {T}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. chunked_attention matches standard attention (no causal mask)
# ---------------------------------------------------------------------------


def test_chunked_attention_matches_standard():
    """Without causal masking, chunked attention must equal full softmax attention."""
    torch.manual_seed(1)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    ref = reference_attention(q, k, v, causal=False)
    out = chunked_attention(q, k, v, chunk_size=CHUNK, causal=False)

    assert torch.allclose(out.float(), ref.float(), atol=1e-4), (
        f"Max diff: {(out.float() - ref.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 4. chunked_attention with causal=True matches causal reference
# ---------------------------------------------------------------------------


def test_chunked_attention_causal_matches_standard():
    """With causal=True, chunked attention must match causal masked attention."""
    torch.manual_seed(2)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    ref = reference_attention(q, k, v, causal=True)
    out = chunked_attention(q, k, v, chunk_size=CHUNK, causal=True)

    assert torch.allclose(out.float(), ref.float(), atol=1e-4), (
        f"Max diff: {(out.float() - ref.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 5. chunk_size=1 still gives correct result
# ---------------------------------------------------------------------------


def test_chunked_attention_chunk_size_1():
    """chunk_size=1 (most extreme chunking) should still be numerically correct."""
    torch.manual_seed(3)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    ref = reference_attention(q, k, v, causal=True)
    out = chunked_attention(q, k, v, chunk_size=1, causal=True)

    assert torch.allclose(out.float(), ref.float(), atol=1e-4), (
        f"Max diff: {(out.float() - ref.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 6. MemoryEfficientAttention output shape
# ---------------------------------------------------------------------------


def test_memory_efficient_attn_output_shape():
    """MemoryEfficientAttention should return (B, T, D) from (B, T, D) input."""
    torch.manual_seed(4)
    cfg = make_small_config()
    model = MemoryEfficientAttention(cfg, chunk_size=CHUNK)
    model.eval()
    x = torch.randn(B, T, cfg.d_model)
    out = model(x)
    assert out.shape == (B, T, cfg.d_model), f"Expected ({B}, {T}, {cfg.d_model}), got {out.shape}"


# ---------------------------------------------------------------------------
# 7. MemoryEfficientAttention matches manual reference
# ---------------------------------------------------------------------------


def test_memory_efficient_attn_matches_reference():
    """MemoryEfficientAttention output should match a hand-rolled causal attention."""
    torch.manual_seed(5)
    cfg = make_small_config()
    model = MemoryEfficientAttention(cfg, chunk_size=CHUNK)
    model.eval()

    x = torch.randn(B, T, cfg.d_model)

    with torch.no_grad():
        # Reference: manually compute attention using the same weights
        q = model.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = model.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = model.v_proj(x).view(B, T, H, D).transpose(1, 2)

        ref_attn = reference_attention(q, k, v, causal=True)  # (B, H, T, D)
        ref_out = ref_attn.transpose(1, 2).contiguous().view(B, T, cfg.d_model)
        ref_out = model.o_proj(ref_out)

        model_out = model(x)

    assert torch.allclose(model_out, ref_out, atol=1e-4), (
        f"Max diff: {(model_out - ref_out).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 8. compute_attention_memory returns all required keys
# ---------------------------------------------------------------------------


def test_compute_attention_memory_keys():
    """compute_attention_memory must return a dict with all 4 required keys."""
    result = compute_attention_memory(seq_len=T, n_heads=H, head_dim=D, chunk_size=CHUNK)
    required_keys = {"standard_attn_mb", "chunked_attn_mb", "qkv_mb", "reduction_factor"}
    assert set(result.keys()) == required_keys, (
        f"Missing keys: {required_keys - set(result.keys())}"
    )
    for key, val in result.items():
        assert isinstance(val, float), f"Key '{key}' value should be float, got {type(val)}"


# ---------------------------------------------------------------------------
# 9. reduction_factor > 1 when chunk_size < seq_len
# ---------------------------------------------------------------------------


def test_memory_reduction_factor_greater_1():
    """With chunk_size < seq_len, reduction_factor should be > 1."""
    assert CHUNK < T, "Test requires chunk_size < seq_len"
    result = compute_attention_memory(seq_len=T, n_heads=H, head_dim=D, chunk_size=CHUNK)
    assert result["reduction_factor"] > 1.0, (
        f"Expected reduction_factor > 1, got {result['reduction_factor']}"
    )


# ---------------------------------------------------------------------------
# 10. use_cache=True returns tuple
# ---------------------------------------------------------------------------


def test_efficient_attn_use_cache():
    """With use_cache=True, forward should return (output, present_kv) tuple."""
    torch.manual_seed(6)
    cfg = make_small_config()
    model = MemoryEfficientAttention(cfg, chunk_size=CHUNK)
    model.eval()
    x = torch.randn(B, T, cfg.d_model)

    result = model(x, use_cache=True)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-element tuple, got {len(result)}"

    out, present_kv = result
    assert out.shape == (B, T, cfg.d_model), f"Output shape mismatch: {out.shape}"
    assert isinstance(present_kv, tuple), f"present_kv should be tuple, got {type(present_kv)}"
    assert len(present_kv) == 2, f"present_kv should have 2 elements, got {len(present_kv)}"
