"""Tests for GroupedQueryAttention and GQAConfig."""

from __future__ import annotations

import pytest
import torch

from src.inference.grouped_query_attention import GQAConfig, GroupedQueryAttention

SMALL = dict(n_heads=4, n_kv_heads=2, head_dim=8, d_model=32)
BATCH, SEQ = 2, 8


def make_module(**kwargs) -> GroupedQueryAttention:
    cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in GQAConfig.__dataclass_fields__}
    d_model = kwargs.pop("d_model", 32)
    cfg = GQAConfig(**cfg_kwargs)
    return GroupedQueryAttention(cfg, d_model=d_model)


class TestGQAConfig:
    def test_defaults(self):
        cfg = GQAConfig()
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8
        assert cfg.head_dim == 128
        assert cfg.dropout == 0.0
        assert cfg.causal is True

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            GQAConfig(n_heads=5, n_kv_heads=3)

    def test_mqa_is_valid(self):
        cfg = GQAConfig(n_heads=8, n_kv_heads=1)
        assert cfg.n_kv_heads == 1


class TestGroupedQueryAttentionInit:
    def test_projection_shapes(self):
        gqa = make_module(**SMALL)
        assert gqa.q_proj.weight.shape == (SMALL["n_heads"] * SMALL["head_dim"], SMALL["d_model"])
        assert gqa.k_proj.weight.shape == (
            SMALL["n_kv_heads"] * SMALL["head_dim"],
            SMALL["d_model"],
        )
        assert gqa.v_proj.weight.shape == (
            SMALL["n_kv_heads"] * SMALL["head_dim"],
            SMALL["d_model"],
        )
        assert gqa.o_proj.weight.shape == (SMALL["d_model"], SMALL["n_heads"] * SMALL["head_dim"])

    def test_n_groups_property(self):
        gqa = make_module(**SMALL)
        assert gqa.n_groups == SMALL["n_heads"] // SMALL["n_kv_heads"]

    def test_is_nn_module(self):
        gqa = make_module(**SMALL)
        assert isinstance(gqa, torch.nn.Module)


class TestGroupedQueryAttentionForward:
    def test_output_shape(self):
        gqa = make_module(**SMALL)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        out, cache = gqa(x)
        assert out.shape == (BATCH, SEQ, SMALL["d_model"])

    def test_cache_shapes(self):
        gqa = make_module(**SMALL)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        _, (k, v) = gqa(x)
        assert k.shape == (BATCH, SMALL["n_kv_heads"], SEQ, SMALL["head_dim"])
        assert v.shape == (BATCH, SMALL["n_kv_heads"], SEQ, SMALL["head_dim"])

    def test_kv_cache_concat(self):
        gqa = make_module(**SMALL)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        _, (k1, v1) = gqa(x)

        x2 = torch.randn(BATCH, 1, SMALL["d_model"])
        out2, (k2, v2) = gqa(x2, kv_cache=(k1, v1))
        assert k2.shape[2] == SEQ + 1
        assert v2.shape[2] == SEQ + 1
        assert out2.shape == (BATCH, 1, SMALL["d_model"])

    def test_deterministic_no_dropout(self):
        gqa = make_module(**SMALL)
        gqa.train(False)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        out1, _ = gqa(x)
        out2, _ = gqa(x)
        assert torch.allclose(out1, out2)

    def test_batch_independence(self):
        gqa = make_module(**SMALL)
        gqa.train(False)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        out_full, _ = gqa(x)
        out_b0, _ = gqa(x[:1])
        assert torch.allclose(out_full[:1], out_b0, atol=1e-5)

    def test_causal_mask_effect(self):
        gqa_causal = make_module(**{**SMALL, "causal": True})
        gqa_causal.train(False)
        x = torch.randn(1, 4, SMALL["d_model"])
        with torch.no_grad():
            out1, _ = gqa_causal(x)
            x_mod = x.clone()
            x_mod[0, 3] += 100.0
            out2, _ = gqa_causal(x_mod)
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-4), (
            "Causal masking should prevent future tokens from affecting past outputs"
        )

    def test_attention_mask_accepted(self):
        gqa = make_module(**SMALL)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"])
        mask = torch.zeros(BATCH, 1, SEQ, SEQ)
        out, _ = gqa(x, attention_mask=mask)
        assert out.shape == (BATCH, SEQ, SMALL["d_model"])

    def test_mqa_single_kv_head(self):
        cfg = GQAConfig(n_heads=4, n_kv_heads=1, head_dim=8, causal=True)
        gqa = GroupedQueryAttention(cfg, d_model=32)
        x = torch.randn(2, 6, 32)
        out, (k, v) = gqa(x)
        assert out.shape == (2, 6, 32)
        assert k.shape == (2, 1, 6, 8)

    def test_no_bias_in_projections(self):
        gqa = make_module(**SMALL)
        assert gqa.q_proj.bias is None
        assert gqa.k_proj.bias is None
        assert gqa.v_proj.bias is None
        assert gqa.o_proj.bias is None

    def test_multi_step_cache_accumulation(self):
        gqa = make_module(**SMALL)
        gqa.train(False)
        x_init = torch.randn(1, 4, SMALL["d_model"])
        _, cache = gqa(x_init)
        for step in range(3):
            x_step = torch.randn(1, 1, SMALL["d_model"])
            _, cache = gqa(x_step, kv_cache=cache)
        assert cache[0].shape[2] == 4 + 3

    def test_gradient_flows(self):
        gqa = make_module(**SMALL)
        x = torch.randn(BATCH, SEQ, SMALL["d_model"], requires_grad=True)
        out, _ = gqa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
