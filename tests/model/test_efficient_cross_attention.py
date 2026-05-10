"""Tests for efficient cross-attention variants."""

import pytest
import torch
from aurelius.model.efficient_cross_attention import (
    CrossAttentionLayer,
    GatedCrossAttention,
    LinearCrossAttention,
    PerceiverCrossAttention,
)

# ── shared test dimensions ──────────────────────────────────────────────────
B = 2  # batch size
M = 6  # latent / query sequence length
N = 10  # context sequence length
D_MODEL = 32
D_CONTEXT = 16
N_HEADS = 4


# ── PerceiverCrossAttention ─────────────────────────────────────────────────


class TestPerceiverCrossAttention:
    @pytest.fixture
    def model(self):
        return PerceiverCrossAttention(D_MODEL, D_CONTEXT, n_heads=N_HEADS)

    @pytest.fixture
    def inputs(self):
        latent = torch.randn(B, M, D_MODEL)
        context = torch.randn(B, N, D_CONTEXT)
        return latent, context

    def test_output_shape(self, model, inputs):
        """Test 1: output shape is (B, M, d_latent)."""
        latent, context = inputs
        out = model(latent, context)
        assert out.shape == (B, M, D_MODEL)

    def test_output_finite(self, model, inputs):
        """Test 2: output contains no NaN or Inf."""
        latent, context = inputs
        out = model(latent, context)
        assert torch.isfinite(out).all()

    def test_different_sequence_lengths(self, model):
        """Test 3: works when M != N (and both != each other)."""
        latent = torch.randn(B, 3, D_MODEL)
        context = torch.randn(B, 20, D_CONTEXT)
        out = model(latent, context)
        assert out.shape == (B, 3, D_MODEL)

    def test_gradient_flows(self, model, inputs):
        """Test 4: gradients reach both latent and context projections."""
        latent, context = inputs
        latent = latent.requires_grad_(True)
        context = context.requires_grad_(True)
        out = model(latent, context)
        out.sum().backward()
        assert latent.grad is not None
        assert context.grad is not None
        assert torch.isfinite(latent.grad).all()


# ── GatedCrossAttention ─────────────────────────────────────────────────────


class TestGatedCrossAttention:
    @pytest.fixture
    def model(self):
        return GatedCrossAttention(D_MODEL, D_CONTEXT, n_heads=N_HEADS)

    @pytest.fixture
    def inputs(self):
        x = torch.randn(B, M, D_MODEL)
        context = torch.randn(B, N, D_CONTEXT)
        return x, context

    def test_output_shape(self, model, inputs):
        """Test 5: output shape is (B, T, d_model)."""
        x, context = inputs
        out = model(x, context)
        assert out.shape == (B, M, D_MODEL)

    def test_gate_zero_at_init(self, model, inputs):
        """Test 6: gate initialised to 0 → tanh(0)=0 → output equals input."""
        x, context = inputs
        with torch.no_grad():
            out = model(x, context)
        assert torch.allclose(out, x, atol=1e-6), (
            "At init gate=0 so tanh(gate)=0 and output should equal input"
        )

    def test_gate_nonzero_changes_output(self, model, inputs):
        """Test 7: after manually setting gate != 0, output differs from input."""
        x, context = inputs
        with torch.no_grad():
            model.gate.fill_(1.0)
            out = model(x, context)
        # With a non-zero gate the cross-attention contribution is added,
        # so output should differ from the original input (unless cross_out is
        # identically zero, which is vanishingly unlikely for random inputs).
        assert not torch.allclose(out, x, atol=1e-6)

    def test_output_finite(self, model, inputs):
        """Test 8: output contains no NaN or Inf."""
        x, context = inputs
        out = model(x, context)
        assert torch.isfinite(out).all()


# ── LinearCrossAttention ────────────────────────────────────────────────────


class TestLinearCrossAttention:
    @pytest.fixture
    def model(self):
        return LinearCrossAttention(D_MODEL, D_CONTEXT)

    @pytest.fixture
    def inputs(self):
        x = torch.randn(B, M, D_MODEL)
        context = torch.randn(B, N, D_CONTEXT)
        return x, context

    def test_output_shape(self, model, inputs):
        """Test 9: output shape is (B, M, d_model)."""
        x, context = inputs
        out = model(x, context)
        assert out.shape == (B, M, D_MODEL)

    def test_output_finite(self, model, inputs):
        """Test 10: output contains no NaN or Inf."""
        x, context = inputs
        out = model(x, context)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, model, inputs):
        """Test 11: gradients flow through LinearCrossAttention."""
        x, context = inputs
        x = x.requires_grad_(True)
        context = context.requires_grad_(True)
        out = model(x, context)
        out.sum().backward()
        assert x.grad is not None
        assert context.grad is not None
        assert torch.isfinite(x.grad).all()


# ── CrossAttentionLayer ─────────────────────────────────────────────────────


class TestCrossAttentionLayer:
    @pytest.fixture
    def inputs(self):
        x = torch.randn(B, M, D_MODEL)
        context = torch.randn(B, N, D_CONTEXT)
        return x, context

    def test_gated_output_shape(self, inputs):
        """Test 12: gated variant output shape is (B, T, d_model)."""
        model = CrossAttentionLayer(D_MODEL, D_CONTEXT, n_heads=N_HEADS, variant="gated")
        x, context = inputs
        out = model(x, context)
        assert out.shape == (B, M, D_MODEL)

    def test_linear_output_shape(self, inputs):
        """Test 13: linear variant output shape is (B, T, d_model)."""
        model = CrossAttentionLayer(D_MODEL, D_CONTEXT, n_heads=N_HEADS, variant="linear")
        x, context = inputs
        out = model(x, context)
        assert out.shape == (B, M, D_MODEL)

    def test_gradient_flows(self, inputs):
        """Test 14: gradients flow through CrossAttentionLayer (gated variant)."""
        model = CrossAttentionLayer(D_MODEL, D_CONTEXT, n_heads=N_HEADS, variant="gated")
        x, context = inputs
        x = x.requires_grad_(True)
        out = model(x, context)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_context_produces_different_output(self, inputs):
        """Test 15: different context tensors produce different outputs."""
        # Use non-zero gate so context actually contributes
        model = CrossAttentionLayer(D_MODEL, D_CONTEXT, n_heads=N_HEADS, variant="gated")
        with torch.no_grad():
            model.cross_attn.gate.fill_(1.0)

        x, context1 = inputs
        context2 = torch.randn(B, N, D_CONTEXT)

        with torch.no_grad():
            out1 = model(x, context1)
            out2 = model(x, context2)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Different contexts should produce different outputs"
        )
