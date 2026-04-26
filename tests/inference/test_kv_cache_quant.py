"""Tests for KV cache quantization module."""

import torch

from src.inference.kv_cache_quant import (
    KVCacheQuantConfig,
    QuantizedKVCache,
    compute_quantization_error,
    dequantize_asymmetric,
    dequantize_symmetric,
    detect_outliers,
    quantize_asymmetric,
    quantize_symmetric,
)

# Standard test dimensions
B, N_HEADS, T, HEAD_DIM = 2, 4, 8, 16
GROUP_SIZE = 16  # HEAD_DIM must be divisible by GROUP_SIZE


def _rand_kv():
    """Return random key and value tensors."""
    return (
        torch.randn(B, N_HEADS, T, HEAD_DIM),
        torch.randn(B, N_HEADS, T, HEAD_DIM),
    )


# 1. KVCacheQuantConfig defaults
class TestKVCacheQuantConfig:
    def test_defaults(self):
        cfg = KVCacheQuantConfig()
        assert cfg.bits == 8
        assert cfg.group_size == 64
        assert cfg.symmetric is True
        assert cfg.per_channel is False
        assert cfg.outlier_threshold == 6.0


# 2. quantize_symmetric returns integer tensor and scales
class TestQuantizeSymmetric:
    def test_returns_int_and_scales(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, s = quantize_symmetric(t, bits=8, group_size=GROUP_SIZE)
        assert q.dtype == torch.int8
        assert s.dtype == torch.float32

    # 3. dequantize_symmetric reconstructs approximately (low MSE)
    def test_roundtrip_low_mse(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, s = quantize_symmetric(t, bits=8, group_size=GROUP_SIZE)
        recon = dequantize_symmetric(q, s)
        mse = (t - recon).pow(2).mean().item()
        assert mse < 0.01, f"MSE too high: {mse}"

    # 4. quantize_symmetric 8-bit values in range [-127, 127]
    def test_8bit_range(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, _ = quantize_symmetric(t, bits=8, group_size=GROUP_SIZE)
        assert q.min().item() >= -127
        assert q.max().item() <= 127


# 5. quantize_asymmetric returns 3 tensors
class TestQuantizeAsymmetric:
    def test_returns_three_tensors(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, s, zp = quantize_asymmetric(t, bits=8, group_size=GROUP_SIZE)
        assert q is not None
        assert s is not None
        assert zp is not None

    # 6. dequantize_asymmetric reconstructs approximately
    def test_roundtrip_low_mse(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, s, zp = quantize_asymmetric(t, bits=8, group_size=GROUP_SIZE)
        recon = dequantize_asymmetric(q, s, zp)
        mse = (t - recon).pow(2).mean().item()
        assert mse < 0.01, f"MSE too high: {mse}"


# 7. detect_outliers returns boolean mask of correct shape
class TestDetectOutliers:
    def test_mask_shape(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        mask = detect_outliers(t, threshold=6.0)
        assert mask.shape == t.shape
        assert mask.dtype == torch.bool

    # 8. detect_outliers finds extreme values
    def test_finds_extremes(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        # Inject an obvious outlier
        t[0, 0, 0, 0] = 1000.0
        mask = detect_outliers(t, threshold=6.0)
        assert mask[0, 0, 0, 0].item() is True


# 9-12. QuantizedKVCache tests
class TestQuantizedKVCache:
    # 9. update increases length
    def test_update_increases_length(self):
        cache = QuantizedKVCache(KVCacheQuantConfig(group_size=GROUP_SIZE))
        assert cache.length() == 0
        keys, values = _rand_kv()
        cache.update(keys, values)
        assert cache.length() == T

    # 10. get returns correct shapes after update
    def test_get_shapes(self):
        cache = QuantizedKVCache(KVCacheQuantConfig(group_size=GROUP_SIZE))
        keys, values = _rand_kv()
        cache.update(keys, values)
        k_out, v_out = cache.get()
        assert k_out.shape == (B, N_HEADS, T, HEAD_DIM)
        assert v_out.shape == (B, N_HEADS, T, HEAD_DIM)

    # 11. clear resets to 0 length
    def test_clear(self):
        cache = QuantizedKVCache(KVCacheQuantConfig(group_size=GROUP_SIZE))
        keys, values = _rand_kv()
        cache.update(keys, values)
        cache.clear()
        assert cache.length() == 0

    # 12. multiple updates concatenate correctly
    def test_multiple_updates(self):
        cache = QuantizedKVCache(KVCacheQuantConfig(group_size=GROUP_SIZE))
        k1, v1 = _rand_kv()
        k2, v2 = _rand_kv()
        cache.update(k1, v1)
        cache.update(k2, v2)
        assert cache.length() == T * 2
        k_out, v_out = cache.get()
        assert k_out.shape == (B, N_HEADS, T * 2, HEAD_DIM)

    # 14. memory_savings shows compression > 1
    def test_memory_savings(self):
        cache = QuantizedKVCache(KVCacheQuantConfig(group_size=GROUP_SIZE))
        keys, values = _rand_kv()
        cache.update(keys, values)
        savings = cache.memory_savings()
        assert savings["compression_ratio"] > 1.0
        assert savings["fp16_bytes"] > 0
        assert savings["quantized_bytes"] > 0


# 13. compute_quantization_error returns correct keys
class TestComputeQuantizationError:
    def test_returns_correct_keys(self):
        t = torch.randn(B, N_HEADS, T, HEAD_DIM)
        q, s = quantize_symmetric(t, bits=8, group_size=GROUP_SIZE)
        recon = dequantize_symmetric(q, s)
        err = compute_quantization_error(t, recon)
        assert "mse" in err
        assert "max_error" in err
        assert "snr_db" in err
        assert err["mse"] >= 0
        assert err["snr_db"] > 0  # should have positive SNR
