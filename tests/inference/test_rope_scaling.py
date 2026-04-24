from __future__ import annotations

import math

import pytest
import torch

from src.inference.rope_scaling import RoPEConfig, RotaryEmbedding


def make_emb(scaling_type="none", scaling_factor=1.0, dim=64, base=10000.0,
             max_position=4096, original_max_position=4096):
    cfg = RoPEConfig(
        dim=dim,
        base=base,
        max_position=max_position,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
        original_max_position=original_max_position,
    )
    return RotaryEmbedding(cfg)


class TestRoPEConfig:
    def test_defaults(self):
        cfg = RoPEConfig()
        assert cfg.dim == 128
        assert cfg.base == 10000.0
        assert cfg.scaling_type == "none"
        assert cfg.scaling_factor == 1.0

    def test_custom_values(self):
        cfg = RoPEConfig(dim=64, base=500.0, scaling_type="linear", scaling_factor=2.0)
        assert cfg.dim == 64
        assert cfg.base == 500.0
        assert cfg.scaling_type == "linear"


class TestRotaryEmbeddingNone:
    def test_inv_freq_shape(self):
        emb = make_emb(dim=64)
        assert emb.inv_freq.shape == (32,)

    def test_inv_freq_values(self):
        emb = make_emb(dim=64, base=10000.0)
        half = 32
        expected_0 = 1.0 / (10000.0 ** (0.0 / 64))
        assert math.isclose(emb.inv_freq[0].item(), expected_0, rel_tol=1e-5)
        expected_last = 1.0 / (10000.0 ** (62.0 / 64))
        assert math.isclose(emb.inv_freq[-1].item(), expected_last, rel_tol=1e-5)

    def test_forward_shapes(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(16)
        assert cos.shape == (16, 32)
        assert sin.shape == (16, 32)

    def test_forward_cos_sin_range(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(32)
        assert cos.abs().max().item() <= 1.0 + 1e-5
        assert sin.abs().max().item() <= 1.0 + 1e-5

    def test_forward_position_zero_is_identity(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(10)
        assert torch.allclose(cos[0], torch.ones(32))
        assert torch.allclose(sin[0], torch.zeros(32))


class TestLinearScaling:
    def test_linear_scales_inv_freq(self):
        base_emb = make_emb(scaling_type="none", scaling_factor=1.0, dim=64)
        scaled_emb = make_emb(scaling_type="linear", scaling_factor=2.0, dim=64)
        assert torch.allclose(scaled_emb.inv_freq * 2.0, base_emb.inv_freq, atol=1e-5)

    def test_linear_factor_one_equals_none(self):
        emb_none = make_emb(scaling_type="none", dim=64)
        emb_lin = make_emb(scaling_type="linear", scaling_factor=1.0, dim=64)
        assert torch.allclose(emb_none.inv_freq, emb_lin.inv_freq, atol=1e-6)

    def test_linear_forward_shape(self):
        emb = make_emb(scaling_type="linear", scaling_factor=4.0, dim=64)
        cos, sin = emb.forward(64)
        assert cos.shape == (64, 32)


class TestNTKScaling:
    def test_ntk_changes_inv_freq(self):
        base_emb = make_emb(scaling_type="none", dim=64)
        ntk_emb = make_emb(scaling_type="ntk", scaling_factor=2.0, dim=64)
        assert not torch.allclose(base_emb.inv_freq, ntk_emb.inv_freq)

    def test_ntk_factor_one_close_to_none(self):
        emb_none = make_emb(scaling_type="none", dim=64)
        emb_ntk = make_emb(scaling_type="ntk", scaling_factor=1.0, dim=64)
        assert torch.allclose(emb_none.inv_freq, emb_ntk.inv_freq, atol=1e-5)

    def test_ntk_adjusted_base_formula(self):
        dim, base, factor = 64, 10000.0, 4.0
        adjusted_base = base * (factor ** (dim / (dim - 2)))
        half = dim // 2
        exponents = torch.arange(0, half, dtype=torch.float32) * 2.0 / dim
        expected = 1.0 / (adjusted_base ** exponents)
        emb = make_emb(scaling_type="ntk", scaling_factor=factor, dim=dim, base=base)
        assert torch.allclose(emb.inv_freq, expected, atol=1e-5)


class TestYaRNScaling:
    def test_yarn_forward_returns_correct_shape(self):
        emb = make_emb(scaling_type="yarn", scaling_factor=4.0, dim=64,
                       max_position=16384, original_max_position=4096)
        cos, sin = emb.forward(128)
        assert cos.shape == (128, 32)

    def test_yarn_changes_inv_freq_vs_none(self):
        emb_none = make_emb(scaling_type="none", dim=64, scaling_factor=1.0)
        emb_yarn = make_emb(scaling_type="yarn", scaling_factor=4.0, dim=64,
                            max_position=16384, original_max_position=4096)
        assert not torch.allclose(emb_none.inv_freq, emb_yarn.inv_freq)

    def test_yarn_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling_type"):
            make_emb(scaling_type="bogus")


class TestApplyRotary:
    def test_output_shapes(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(10)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_rot, k_rot = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_preserves_norm(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(8)
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 8, 64)
        q_rot, k_rot = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4)

    def test_apply_rotary_at_pos_zero_identity(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(5)
        q = torch.randn(1, 1, 5, 64)
        k = torch.randn(1, 1, 5, 64)
        q_rot, _ = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        assert torch.allclose(q[:, :, 0, :], q_rot[:, :, 0, :], atol=1e-5)

    def test_device_argument(self):
        emb = make_emb(dim=64)
        cos, sin = emb.forward(8, device=torch.device("cpu"))
        assert cos.device.type == "cpu"
