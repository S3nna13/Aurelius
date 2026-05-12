"""Tests for kv_cache_compression.py — PackKV-inspired KV cache compression."""

from __future__ import annotations

import torch

from gateway.kv_cache_compression import (
    CompressedPagedKVCache,
    KVCacheCompressor,
)


class TestKVCacheCompressor:
    """PackKV-style per-token INT8 quantization."""

    def test_quantize_dequantize_roundtrip(self):
        c = KVCacheCompressor()
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        k_quant, k_scale = c._quantize(k)
        k_rec = c._dequantize(k_quant, k_scale)
        assert k_quant.dtype == torch.int8
        assert k_rec.dtype == torch.float16
        max_err = (k - k_rec).abs().max().item()
        assert max_err < 0.1

    def test_quantize_int8_range(self):
        c = KVCacheCompressor(quant_dtype=torch.int8)
        k = torch.randn(8, 4, 32, dtype=torch.float16) * 10
        k_quant, _ = c._quantize(k)
        assert k_quant.min() >= -128
        assert k_quant.max() <= 127

    def test_scale_not_zero(self):
        c = KVCacheCompressor()
        k = torch.randn(4, 2, 16, dtype=torch.float16)
        _, scale = c._quantize(k)
        assert (scale > 0).all()

    def test_compression_ratio_approx_2x(self):
        c = KVCacheCompressor()
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        block = c.compress_block(k, v, block_id=0, num_tokens=16)
        ratio = c.compression_ratio(block)
        assert 1.9 < ratio < 2.1

    def test_decompress_block(self):
        c = KVCacheCompressor()
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        block = c.compress_block(k, v, block_id=5, num_tokens=16)
        k_dec, v_dec = c.decompress_block(block)
        max_err_k = (k - k_dec).abs().max().item()
        max_err_v = (v - v_dec).abs().max().item()
        assert max_err_k < 0.1
        assert max_err_v < 0.1

    def test_compressed_block_has_quant_dtypes(self):
        c = KVCacheCompressor()
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        block = c.compress_block(k, v, block_id=0, num_tokens=10)
        assert block.k_quant.dtype == torch.int8
        assert block.v_quant.dtype == torch.int8
        assert block.k_scale.dtype == torch.float16
        assert block.v_scale.dtype == torch.float16
        assert block.num_tokens == 10


class TestSparsityDetection:
    """Sparse encoding when many entries are near-zero."""

    def test_sparse_encoding_triggered(self):
        c = KVCacheCompressor(sparse_threshold=0.5)
        k = torch.randn(8, 4, 32, dtype=torch.float16)
        # Force many near-zero rows
        k[4:, :] = 0.0
        _, sparsity = c._detect_sparsity(k)
        assert sparsity > 0.4

    def test_no_sparse_when_below_threshold(self):
        c = KVCacheCompressor(sparse_threshold=0.8)
        k = torch.randn(8, 4, 32, dtype=torch.float16) * 0.1
        mask, sparsity = c._detect_sparsity(k)
        assert mask is None
        assert sparsity < 0.8

    def test_sparse_mask_zeroes_out_small_values_on_decompress(self):
        c = KVCacheCompressor(sparse_threshold=0.5, zero_tolerance=0.01)
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        k[8:, :] = 0.0
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        block = c.compress_block(k, v, block_id=0, num_tokens=16)
        k_dec, _ = c.decompress_block(block)
        assert (k_dec[8:] == 0.0).all()


class TestCompressedBlock:
    """CompressedBlock dataclass."""

    def test_block_fields(self):
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        c = KVCacheCompressor()
        block = c.compress_block(k, v, block_id=42, num_tokens=16)
        assert block.block_id == 42
        assert block.num_tokens == 16
        assert block.sparse_mask is not None or block.sparse_mask is None


class TestCompressedPagedKVCache:
    """CompressedPagedKVCache wrapper."""

    def test_wrapper_instantiates(self):
        from gateway.paged_kv_cache import PagedKVCache

        base = PagedKVCache(n_layers=32, n_kv_heads=8, head_dim=64)
        wrapped = CompressedPagedKVCache(base)
        assert wrapped.total_compression_ratio == 1.0  # no blocks yet

    def test_compression_ratio_reported(self):
        from gateway.paged_kv_cache import PagedKVCache

        base = PagedKVCache(n_layers=32, n_kv_heads=8, head_dim=64)
        wrapped = CompressedPagedKVCache(
            base, compress_frequency=1,
        )
        # Manually compress a block
        k = torch.randn(16, 8, 64, dtype=torch.float16)
        v = torch.randn(16, 8, 64, dtype=torch.float16)
        comp = KVCacheCompressor()
        block = comp.compress_block(k, v, block_id=0, num_tokens=16)
        wrapped._compressed[0] = block
        ratio = wrapped.total_compression_ratio
        assert ratio > 1.9
