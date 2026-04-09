"""Tests for hierarchical_compression module."""

import pytest
import torch

from src.model.hierarchical_compression import (
    AutoCompressorBlock,
    CompressedContextAttention,
    CompressionConfig,
    HierarchicalCompressor,
    SegmentCompressor,
    compress_and_attend,
)

# Default small config for fast tests
D_MODEL = 64
N_HEADS = 2
D_FF = 128
N_SUMMARY_TOKENS = 4
COMPRESSION_RATIO = 4
N_COMPRESS_LAYERS = 2
B = 2
T = 32


def make_config(**kwargs):
    defaults = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_summary_tokens=N_SUMMARY_TOKENS,
        compression_ratio=COMPRESSION_RATIO,
        n_compress_layers=N_COMPRESS_LAYERS,
    )
    defaults.update(kwargs)
    return CompressionConfig(**defaults)


def make_input(b=B, t=T, d=D_MODEL):
    torch.manual_seed(0)
    return torch.randn(b, t, d)


# ---------------------------------------------------------------------------
# 1. CompressionConfig defaults
# ---------------------------------------------------------------------------
def test_compression_config_defaults():
    cfg = CompressionConfig()
    assert cfg.compression_ratio == 4
    assert cfg.n_summary_tokens == 8
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.d_ff == 2048
    assert cfg.n_compress_layers == 2
    assert cfg.dropout == 0.1


# ---------------------------------------------------------------------------
# 2. SegmentCompressor output shape
# ---------------------------------------------------------------------------
def test_segment_compressor_output_shape():
    torch.manual_seed(0)
    cfg = make_config()
    model = SegmentCompressor(cfg)
    seg_size = N_SUMMARY_TOKENS * COMPRESSION_RATIO  # 16
    x = make_input(t=seg_size)
    out = model(x)
    assert out.shape == (B, N_SUMMARY_TOKENS, D_MODEL)


# ---------------------------------------------------------------------------
# 3. SegmentCompressor gradient flow
# ---------------------------------------------------------------------------
def test_segment_compressor_gradient_flow():
    torch.manual_seed(0)
    cfg = make_config()
    model = SegmentCompressor(cfg)
    seg_size = N_SUMMARY_TOKENS * COMPRESSION_RATIO
    x = make_input(t=seg_size)
    x.requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 4. HierarchicalCompressor output shapes
# ---------------------------------------------------------------------------
def test_hierarchical_compressor_output_shapes():
    torch.manual_seed(0)
    cfg = make_config()
    model = HierarchicalCompressor(cfg)
    x = make_input()  # (2, 32, 64)
    compressed, original = model(x)
    # segment_size = 4 * 4 = 16; n_seg = 32/16 = 2; compressed_len = 2 * 4 = 8
    assert original.shape == (B, T, D_MODEL)
    assert compressed.shape[0] == B
    assert compressed.shape[2] == D_MODEL
    assert compressed.ndim == 3


# ---------------------------------------------------------------------------
# 5. HierarchicalCompressor short sequence (shorter than segment_size)
# ---------------------------------------------------------------------------
def test_hierarchical_compressor_short_sequence():
    torch.manual_seed(0)
    cfg = make_config()
    model = HierarchicalCompressor(cfg)
    seg_size = cfg.n_summary_tokens * cfg.compression_ratio  # 16
    short_t = seg_size - 4  # 12
    x = make_input(t=short_t)
    compressed, original = model(x)
    # Should pad to one full segment → 1 seg → n_summary_tokens summaries
    assert original.shape == (B, short_t, D_MODEL)
    assert compressed.shape == (B, N_SUMMARY_TOKENS, D_MODEL)


# ---------------------------------------------------------------------------
# 6. HierarchicalCompressor long sequence (> 2 segments)
# ---------------------------------------------------------------------------
def test_hierarchical_compressor_long_sequence():
    torch.manual_seed(0)
    cfg = make_config()
    model = HierarchicalCompressor(cfg)
    seg_size = cfg.n_summary_tokens * cfg.compression_ratio  # 16
    long_t = seg_size * 3  # 48 → exactly 3 segments
    x = make_input(t=long_t)
    compressed, original = model(x)
    expected_comp_len = 3 * N_SUMMARY_TOKENS  # 12
    assert original.shape == (B, long_t, D_MODEL)
    assert compressed.shape == (B, expected_comp_len, D_MODEL)


# ---------------------------------------------------------------------------
# 7. Compression rate > 1 for long sequences
# ---------------------------------------------------------------------------
def test_compression_rate():
    torch.manual_seed(0)
    cfg = make_config()
    model = HierarchicalCompressor(cfg)
    x = make_input()  # T=32, segment_size=16, n_seg=2, comp=8
    compressed, original = model(x)
    rate = model.compute_compression_rate(original.size(1), compressed.size(1))
    assert rate > 1.0


# ---------------------------------------------------------------------------
# 8. CompressedContextAttention output shape
# ---------------------------------------------------------------------------
def test_compressed_context_attention_shape():
    torch.manual_seed(0)
    cfg = make_config()
    model = CompressedContextAttention(cfg)
    current = make_input()  # (2, 32, 64)
    compressed_past = make_input(t=8)  # (2, 8, 64)
    out = model(current, compressed_past)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 9. CompressedContextAttention gradient flow
# ---------------------------------------------------------------------------
def test_compressed_context_attention_gradient_flow():
    torch.manual_seed(0)
    cfg = make_config()
    model = CompressedContextAttention(cfg)
    current = make_input()
    current.requires_grad_(True)
    compressed_past = make_input(t=8)
    out = model(current, compressed_past)
    out.sum().backward()
    assert current.grad is not None
    assert current.grad.shape == current.shape


# ---------------------------------------------------------------------------
# 10. compress_and_attend pipeline
# ---------------------------------------------------------------------------
def test_compress_and_attend_pipeline():
    torch.manual_seed(0)
    cfg = make_config()
    compressor = HierarchicalCompressor(cfg)
    attn = CompressedContextAttention(cfg)
    x = make_input()
    output, rate = compress_and_attend(x, compressor, attn)
    assert output.shape == (B, T, D_MODEL)
    assert isinstance(rate, float)
    assert rate > 0.0


# ---------------------------------------------------------------------------
# 11. AutoCompressorBlock output shape
# ---------------------------------------------------------------------------
def test_autocompressor_block_output_shape():
    torch.manual_seed(0)
    cfg = make_config()
    block = AutoCompressorBlock(cfg)
    x = make_input()
    output, rate = block(x)
    assert output.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 12. AutoCompressorBlock compression rate positive
# ---------------------------------------------------------------------------
def test_autocompressor_block_rate_positive():
    torch.manual_seed(0)
    cfg = make_config()
    block = AutoCompressorBlock(cfg)
    x = make_input()
    _, rate = block(x)
    assert rate > 0.0
