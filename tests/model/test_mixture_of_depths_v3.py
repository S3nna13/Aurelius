"""Tests for Mixture of Depths v3 (mixture_of_depths_v3.py)."""

from __future__ import annotations

import math

import torch
from aurelius.model.mixture_of_depths_v3 import (
    MoDBlock,
    MoDConfig,
    MoDModel,
    TokenRouter,
)

# ---------------------------------------------------------------------------
# Tiny dimensions used throughout
# ---------------------------------------------------------------------------
D = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
VOCAB = 64
B = 2
T = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(capacity: float = 0.5) -> MoDConfig:
    return MoDConfig(d_model=D, capacity=capacity, n_heads=N_HEADS, d_ff=D_FF)


def make_x(b: int = B, t: int = T) -> torch.Tensor:
    return torch.randn(b, t, D)


# ---------------------------------------------------------------------------
# TokenRouter tests
# ---------------------------------------------------------------------------


class TestTokenRouter:
    def test_router_weights_shape(self):
        """router_weights must be (B, T, 1)."""
        router = TokenRouter(D, capacity=0.5)
        x = make_x()
        weights, mask, aux = router(x)
        assert weights.shape == (B, T, 1)

    def test_selected_mask_shape(self):
        """selected_mask must be (B, T) bool tensor."""
        router = TokenRouter(D, capacity=0.5)
        x = make_x()
        _, mask, _ = router(x)
        assert mask.shape == (B, T)
        assert mask.dtype == torch.bool

    def test_selected_mask_count(self):
        """Each batch item must have exactly ceil(capacity * T) True values."""
        capacity = 0.5
        k_expected = math.ceil(capacity * T)
        router = TokenRouter(D, capacity=capacity)
        x = make_x()
        _, mask, _ = router(x)
        for b in range(B):
            assert mask[b].sum().item() == k_expected

    def test_selected_mask_count_capacity_0625(self):
        """Check count for a non-round capacity fraction."""
        capacity = 0.625
        k_expected = math.ceil(capacity * T)
        router = TokenRouter(D, capacity=capacity)
        x = make_x()
        _, mask, _ = router(x)
        for b in range(B):
            assert mask[b].sum().item() == k_expected

    def test_aux_loss_is_scalar_and_nonneg(self):
        """aux_loss must be a non-negative scalar."""
        router = TokenRouter(D, capacity=0.5)
        x = make_x()
        _, _, aux = router(x)
        assert aux.shape == ()
        assert aux.item() >= 0.0

    def test_capacity_1_all_selected(self):
        """With capacity=1.0, all T tokens must be selected in every batch item."""
        router = TokenRouter(D, capacity=1.0)
        x = make_x()
        _, mask, _ = router(x)
        assert mask.all()

    def test_capacity_very_small(self):
        """With capacity=1/T, only 1 token is selected per batch item."""
        capacity = 1.0 / T  # ceil(capacity*T) == 1
        router = TokenRouter(D, capacity=capacity)
        x = make_x()
        _, mask, _ = router(x)
        for b in range(B):
            assert mask[b].sum().item() == 1

    def test_router_differentiable(self):
        """Gradient must flow back through the router weights."""
        router = TokenRouter(D, capacity=0.5)
        x = make_x().requires_grad_(True)
        weights, _, aux = router(x)
        loss = weights.sum() + aux
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# MoDBlock tests
# ---------------------------------------------------------------------------


class TestMoDBlock:
    def test_output_shape(self):
        """MoDBlock output must match input shape."""
        block = MoDBlock(make_config())
        x = make_x()
        out, _ = block(x)
        assert out.shape == x.shape

    def test_output_finite(self):
        """MoDBlock output must contain no NaN or Inf values."""
        block = MoDBlock(make_config())
        x = make_x()
        out, _ = block(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        """Gradients must flow through MoDBlock to the input."""
        block = MoDBlock(make_config())
        x = make_x().requires_grad_(True)
        out, aux = block(x)
        (out.sum() + aux).backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_aux_loss_scalar_nonneg(self):
        """aux_loss from MoDBlock must be a non-negative scalar."""
        block = MoDBlock(make_config())
        x = make_x()
        _, aux = block(x)
        assert aux.shape == ()
        assert aux.item() >= 0.0


# ---------------------------------------------------------------------------
# MoDModel tests
# ---------------------------------------------------------------------------


class TestMoDModel:
    def _make_model(self, capacity: float = 0.5) -> MoDModel:
        return MoDModel(
            d_model=D,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF,
            capacity=capacity,
            vocab_size=VOCAB,
        )

    def test_output_shape(self):
        """MoDModel logits must be (B, T, vocab_size)."""
        model = self._make_model()
        ids = torch.randint(0, VOCAB, (B, T))
        logits, _ = model(ids)
        assert logits.shape == (B, T, VOCAB)

    def test_total_aux_loss_nonneg(self):
        """total_aux_loss must be a non-negative scalar."""
        model = self._make_model()
        ids = torch.randint(0, VOCAB, (B, T))
        _, aux = model(ids)
        assert aux.shape == ()
        assert aux.item() >= 0.0

    def test_gradient_flows_to_embedding(self):
        """Gradients must reach the embedding weight."""
        model = self._make_model()
        ids = torch.randint(0, VOCAB, (B, T))
        logits, aux = model(ids)
        (logits.sum() + aux).backward()
        assert model.embedding.weight.grad is not None

    def test_b1_t1(self):
        """Model must work with a single token (B=1, T=1)."""
        model = self._make_model()
        ids = torch.randint(0, VOCAB, (1, 1))
        logits, aux = model(ids)
        assert logits.shape == (1, 1, VOCAB)
        assert aux.shape == ()

    def test_output_finite(self):
        """MoDModel logits must be finite."""
        model = self._make_model()
        ids = torch.randint(0, VOCAB, (B, T))
        logits, _ = model(ids)
        assert torch.isfinite(logits).all()
