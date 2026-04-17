"""Tests for src/model/rope_extensions_v2.py.

15 tests covering RotaryEmbedding, NTKRoPE, YaRNRoPE, DynamicNTKRoPE,
and RoPEAnalyzer with tiny configs.

Config: head_dim=8, base=10000.0, seq_len=8, batch=2, n_heads=2.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from aurelius.model.rope_extensions_v2 import (
    DynamicNTKRoPE,
    NTKRoPE,
    RoPEAnalyzer,
    RotaryEmbedding,
    YaRNRoPE,
)

# ---------------------------------------------------------------------------
# Shared tiny config constants
# ---------------------------------------------------------------------------

HEAD_DIM = 8
BASE = 10000.0
SEQ_LEN = 8
BATCH = 2
N_HEADS = 2


def _qk(seq_len: int = SEQ_LEN, head_dim: int = HEAD_DIM) -> tuple[Tensor, Tensor]:
    torch.manual_seed(7)
    q = torch.randn(BATCH, N_HEADS, seq_len, head_dim)
    k = torch.randn(BATCH, N_HEADS, seq_len, head_dim)
    return q, k


# ===========================================================================
# 1. RotaryEmbedding: output shapes match input (B, H, T, D)
# ===========================================================================

def test_rotary_embedding_output_shapes():
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    q, k = _qk()
    q_rot, k_rot = rope.forward(q, k)
    assert q_rot.shape == q.shape, f"q_rot shape mismatch: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"k_rot shape mismatch: {k_rot.shape} != {k.shape}"


# ===========================================================================
# 2. RotaryEmbedding: position 0 → identity rotation (no change to vectors)
# ===========================================================================

def test_rotary_embedding_position_zero_identity():
    """At position 0 the rotation angle is 0, so output == input."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    # Use seq_len=1 so only position 0 exists
    q, k = _qk(seq_len=1)
    q_rot, k_rot = rope.forward(q, k)
    assert torch.allclose(q_rot, q, atol=1e-5), "Position 0 should be identity rotation"
    assert torch.allclose(k_rot, k, atol=1e-5), "Position 0 should be identity rotation"


# ===========================================================================
# 3. RotaryEmbedding: different positions → different rotations
# ===========================================================================

def test_rotary_embedding_different_positions_differ():
    """Positions 1, 2, 3, … must produce distinct rotations."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    q, k = _qk(seq_len=4)
    q_rot, _ = rope.forward(q, k)
    # Position 1 and position 2 outputs should differ even with same input values
    uniform = torch.ones(BATCH, N_HEADS, 4, HEAD_DIM)
    k_uniform = torch.ones(BATCH, N_HEADS, 4, HEAD_DIM)
    q_rot_u, _ = rope.forward(uniform, k_uniform)
    assert not torch.allclose(q_rot_u[:, :, 1, :], q_rot_u[:, :, 2, :], atol=1e-6), (
        "Different positions should produce different rotations"
    )


# ===========================================================================
# 4. RotaryEmbedding: norm preserved ||q_rot|| == ||q||
# ===========================================================================

def test_rotary_embedding_norm_preserved():
    """RoPE is a rotation — must preserve L2 norm exactly."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    q, k = _qk()
    q_rot, k_rot = rope.forward(q, k)
    norm_q_in = q.norm(dim=-1)
    norm_q_out = q_rot.norm(dim=-1)
    norm_k_in = k.norm(dim=-1)
    norm_k_out = k_rot.norm(dim=-1)
    assert torch.allclose(norm_q_in, norm_q_out, atol=1e-5), "q norm not preserved"
    assert torch.allclose(norm_k_in, norm_k_out, atol=1e-5), "k norm not preserved"


# ===========================================================================
# 5. NTKRoPE: base_new > original base (scaled up)
# ===========================================================================

def test_ntk_rope_base_scaled_up():
    """NTK-modified base must be strictly larger than original base."""
    scale_factor = 8.0
    rope = NTKRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=scale_factor)
    # NTK base = base * scale_factor^(d/(d-2))
    expected_base_new = BASE * (scale_factor ** (HEAD_DIM / (HEAD_DIM - 2)))
    assert rope.base > BASE, f"NTKRoPE base {rope.base} should be > original {BASE}"
    assert abs(rope.base - expected_base_new) < 1.0, (
        f"NTKRoPE base {rope.base} ≠ expected {expected_base_new}"
    )


# ===========================================================================
# 6. NTKRoPE.context_window_extension: equals scale_factor
# ===========================================================================

def test_ntk_rope_context_window_extension():
    scale_factor = 8.0
    rope = NTKRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=scale_factor)
    ext = rope.context_window_extension()
    assert ext == scale_factor, f"context_window_extension() = {ext}, expected {scale_factor}"


# ===========================================================================
# 7. NTKRoPE: output shapes correct, norm preserved
# ===========================================================================

def test_ntk_rope_shapes_and_norm():
    rope = NTKRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=4.0)
    q, k = _qk()
    q_rot, k_rot = rope.forward(q, k)
    assert q_rot.shape == q.shape, f"q_rot shape {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"k_rot shape {k_rot.shape} != {k.shape}"
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5), "NTKRoPE norm not preserved"
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5), "NTKRoPE norm not preserved"


# ===========================================================================
# 8. YaRNRoPE: output shapes correct (B, H, T, D)
# ===========================================================================

def test_yarn_rope_output_shapes():
    rope = YaRNRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=4.0)
    q, k = _qk()
    q_rot, k_rot = rope.forward(q, k)
    assert q_rot.shape == q.shape, f"YaRN q_rot shape {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"YaRN k_rot shape {k_rot.shape} != {k.shape}"


# ===========================================================================
# 9. YaRNRoPE: output differs from standard RoPE (scale_factor != 1)
# ===========================================================================

def test_yarn_rope_differs_from_standard():
    """YaRN with scale_factor > 1 should produce different output than standard RoPE."""
    standard = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    yarn = YaRNRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=8.0)
    q, k = _qk()
    q_std, _ = standard.forward(q, k)
    q_yarn, _ = yarn.forward(q, k)
    assert not torch.allclose(q_std, q_yarn, atol=1e-4), (
        "YaRN output should differ from standard RoPE when scale_factor > 1"
    )


# ===========================================================================
# 10. YaRNRoPE: norm approximately preserved (within 5% due to mscale correction)
# ===========================================================================

def test_yarn_rope_norm_approximately_preserved():
    """YaRN applies a magnitude correction (mscale), norm should be within 5%."""
    rope = YaRNRoPE(head_dim=HEAD_DIM, base=BASE, scale_factor=4.0, mscale=0.1)
    q, k = _qk()
    q_rot, k_rot = rope.forward(q, k)
    norm_in = q.norm(dim=-1)
    norm_out = q_rot.norm(dim=-1)
    ratio = (norm_out / (norm_in + 1e-8))
    assert (ratio - 1.0).abs().max().item() < 0.05, (
        f"YaRN norm ratio out of 5% window: max deviation {(ratio - 1.0).abs().max().item():.4f}"
    )


# ===========================================================================
# 11. DynamicNTKRoPE: seq_len <= max_pos_emb → same as standard RoPE
# ===========================================================================

def test_dynamic_ntk_short_seq_matches_standard():
    """Within training range, DynamicNTKRoPE should match standard RoPE exactly."""
    max_pos = SEQ_LEN * 2  # 16, use seq_len=8 which is <= 16
    standard = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    dynamic = DynamicNTKRoPE(head_dim=HEAD_DIM, base=BASE, max_position_embeddings=max_pos)
    q, k = _qk()
    q_std, k_std = standard.forward(q, k)
    q_dyn, k_dyn = dynamic.forward(q, k)
    assert torch.allclose(q_std, q_dyn, atol=1e-5), "DynamicNTK should match standard RoPE for short seqs"
    assert torch.allclose(k_std, k_dyn, atol=1e-5), "DynamicNTK should match standard RoPE for short seqs"


# ===========================================================================
# 12. DynamicNTKRoPE: seq_len > max_pos_emb → different from standard (adaptive scaling)
# ===========================================================================

def test_dynamic_ntk_long_seq_differs_from_standard():
    """Beyond training range, DynamicNTKRoPE scales the base — output should differ."""
    max_pos = 4  # small so SEQ_LEN=8 exceeds it
    standard = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    dynamic = DynamicNTKRoPE(head_dim=HEAD_DIM, base=BASE, max_position_embeddings=max_pos)
    q, k = _qk()  # seq_len=8 > max_pos=4
    q_std, _ = standard.forward(q, k)
    q_dyn, _ = dynamic.forward(q, k)
    assert not torch.allclose(q_std, q_dyn, atol=1e-4), (
        "DynamicNTKRoPE should use a scaled base and differ from standard RoPE for long seqs"
    )


# ===========================================================================
# 13. RoPEAnalyzer.frequency_distribution: all keys present, min_freq < max_freq
# ===========================================================================

def test_rope_analyzer_frequency_distribution():
    analyzer = RoPEAnalyzer()
    dist = analyzer.frequency_distribution(head_dim=HEAD_DIM, base=BASE)
    required_keys = {"min_freq", "max_freq", "n_high_freq", "n_low_freq"}
    assert required_keys.issubset(set(dist.keys())), (
        f"Missing keys: {required_keys - set(dist.keys())}"
    )
    assert dist["min_freq"] < dist["max_freq"], (
        f"min_freq {dist['min_freq']} should be < max_freq {dist['max_freq']}"
    )
    assert isinstance(dist["n_high_freq"], int)
    assert isinstance(dist["n_low_freq"], int)
    assert dist["n_high_freq"] + dist["n_low_freq"] == HEAD_DIM // 2


# ===========================================================================
# 14. RoPEAnalyzer.rotation_matrix: shape (D, D), near-orthogonal (|det| ≈ 1)
# ===========================================================================

def test_rope_analyzer_rotation_matrix():
    analyzer = RoPEAnalyzer()
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)
    R = analyzer.rotation_matrix(rope, seq_pos=3)
    assert R.shape == (HEAD_DIM, HEAD_DIM), f"Expected ({HEAD_DIM}, {HEAD_DIM}), got {R.shape}"
    det = torch.det(R.double()).item()
    assert abs(abs(det) - 1.0) < 0.01, f"|det(R)| = {abs(det):.6f}, expected ≈ 1"


# ===========================================================================
# 15. Causal attention with RotaryEmbedding: full forward/backward pass succeeds
# ===========================================================================

def test_causal_attention_with_rotary_embedding_forward_backward():
    """Full forward + backward through a minimal causal self-attention block."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM, base=BASE)

    # Minimal causal attention parameters
    d_model = N_HEADS * HEAD_DIM  # 16
    W_q = nn.Linear(d_model, d_model, bias=False)
    W_k = nn.Linear(d_model, d_model, bias=False)
    W_v = nn.Linear(d_model, d_model, bias=False)
    W_o = nn.Linear(d_model, d_model, bias=False)

    torch.manual_seed(42)
    x = torch.randn(BATCH, SEQ_LEN, d_model, requires_grad=False)

    # Project
    q = W_q(x).view(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 2)  # (B, H, T, D)
    k = W_k(x).view(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = W_v(x).view(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 2)

    # Apply RoPE
    q_rot, k_rot = rope.forward(q, k)

    # Scaled dot-product with causal mask
    scale = HEAD_DIM ** -0.5
    attn_scores = (q_rot @ k_rot.transpose(-2, -1)) * scale  # (B, H, T, T)
    causal_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Aggregate values
    out = attn_weights @ v  # (B, H, T, D)
    out = out.transpose(1, 2).reshape(BATCH, SEQ_LEN, d_model)
    out = W_o(out)

    # Backward pass
    loss = out.mean()
    loss.backward()

    # Verify gradients exist on projection weights
    assert W_q.weight.grad is not None, "W_q should have gradients"
    assert W_k.weight.grad is not None, "W_k should have gradients"
    assert W_v.weight.grad is not None, "W_v should have gradients"
    assert W_o.weight.grad is not None, "W_o should have gradients"
    assert torch.isfinite(out).all(), "Attention output contains non-finite values"
