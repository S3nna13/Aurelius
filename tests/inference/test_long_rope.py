from __future__ import annotations

import math

import pytest
import torch

from src.inference.long_rope import LongRoPEConfig, LongRoPEEmbedding


def make_emb(dim=64, base=10000.0, original_max=4096, extended_max=131072,
             short_factor=None, long_factor=None):
    cfg = LongRoPEConfig(
        dim=dim,
        base=base,
        original_max=original_max,
        extended_max=extended_max,
        short_factor=short_factor,
        long_factor=long_factor,
    )
    return LongRoPEEmbedding(cfg)


class TestLongRoPEConfig:
    def test_defaults(self):
        cfg = LongRoPEConfig()
        assert cfg.dim == 128
        assert cfg.base == 10000.0
        assert cfg.original_max == 4096
        assert cfg.extended_max == 131072
        assert cfg.short_factor is None
        assert cfg.long_factor is None


class TestLongRoPEEmbeddingInit:
    def test_default_factors_shape(self):
        emb = make_emb(dim=64)
        assert emb.short_factor.shape == (32,)
        assert emb.long_factor.shape == (32,)

    def test_default_factors_linspace(self):
        emb = make_emb(dim=64)
        assert math.isclose(emb.short_factor[0].item(), 1.0, rel_tol=1e-5)
        assert math.isclose(emb.short_factor[-1].item(), 2.0, rel_tol=1e-5)
        assert math.isclose(emb.long_factor[0].item(), 1.0, rel_tol=1e-5)

    def test_custom_short_factor(self):
        half = 32
        sf = [1.5] * half
        emb = make_emb(dim=64, short_factor=sf)
        assert torch.allclose(emb.short_factor, torch.tensor(sf))

    def test_custom_long_factor(self):
        half = 32
        lf = [2.0] * half
        emb = make_emb(dim=64, long_factor=lf)
        assert torch.allclose(emb.long_factor, torch.tensor(lf))

    def test_wrong_short_factor_length_raises(self):
        with pytest.raises(ValueError, match="short_factor length"):
            make_emb(dim=64, short_factor=[1.0] * 10)

    def test_wrong_long_factor_length_raises(self):
        with pytest.raises(ValueError, match="long_factor length"):
            make_emb(dim=64, long_factor=[1.0] * 5)


class TestGetFactors:
    def test_short_ctx_uses_short_factor(self):
        half = 32
        sf = [1.1] * half
        lf = [3.3] * half
        emb = make_emb(dim=64, short_factor=sf, long_factor=lf, original_max=512)
        factors = emb._get_factors(512)
        assert torch.allclose(factors, torch.tensor(sf))

    def test_long_ctx_uses_long_factor(self):
        half = 32
        sf = [1.1] * half
        lf = [3.3] * half
        emb = make_emb(dim=64, short_factor=sf, long_factor=lf, original_max=512)
        factors = emb._get_factors(513)
        assert torch.allclose(factors, torch.tensor(lf))

    def test_boundary_uses_short_factor(self):
        half = 32
        sf = [1.2] * half
        lf = [4.0] * half
        emb = make_emb(dim=64, short_factor=sf, long_factor=lf, original_max=1024)
        factors = emb._get_factors(1024)
        assert torch.allclose(factors, torch.tensor(sf))


class TestLongRoPEForward:
    def test_forward_output_shapes(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(32)
        assert cos.shape == (32, 32)
        assert sin.shape == (32, 32)

    def test_forward_long_seq_shapes(self):
        emb = make_emb(dim=64, original_max=512)
        cos, sin = emb.forward(1024)
        assert cos.shape == (1024, 32)
        assert sin.shape == (1024, 32)

    def test_forward_cos_sin_range(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(64)
        assert cos.abs().max().item() <= 1.0 + 1e-5
        assert sin.abs().max().item() <= 1.0 + 1e-5

    def test_forward_position_zero(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(8)
        assert torch.allclose(cos[0], torch.ones(32))
        assert torch.allclose(sin[0], torch.zeros(32))

    def test_short_vs_long_ctx_different_freqs(self):
        half = 32
        sf = [1.0] * half
        lf = [4.0] * half
        emb = make_emb(dim=64, short_factor=sf, long_factor=lf, original_max=256)
        cos_short, _ = emb.forward(256)
        cos_long, _ = emb.forward(257)
        assert not torch.allclose(cos_short[:256], cos_long[:256])

    def test_device_argument_cpu(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(8, device=torch.device("cpu"))
        assert cos.device.type == "cpu"
        assert sin.device.type == "cpu"


class TestLongRoPEApplyRotary:
    def test_output_shapes(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(10)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_rot, k_rot = LongRoPEEmbedding.apply_rotary(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_preserves_norm(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(6)
        q = torch.randn(1, 2, 6, 64)
        k = torch.randn(1, 2, 6, 64)
        q_rot, k_rot = LongRoPEEmbedding.apply_rotary(q, k, cos, sin)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4)
