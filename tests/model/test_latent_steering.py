"""Tests for src/model/latent_steering.py

Covers:
- SteeringConfig defaults
- extract_hidden_states shape (B, T, d_model)
- extract_hidden_states with negative layer_idx
- compute_steering_vector shape (d_model,)
- apply_steering_hook returns handle with remove() method
- lerp_hidden output shape matches input
- lerp_hidden alpha=0 -> no change
- LatentSteerer.enable / disable hook registration
- LatentSteerer.steer_generate returns token tensor
- "add" method changes hidden states vs unsteered
"""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.latent_steering import (
    SteeringConfig,
    LatentSteerer,
    apply_steering_hook,
    compute_steering_vector,
    extract_hidden_states,
    lerp_hidden,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture(scope="module")
def model():
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, TINY_CFG.vocab_size, (2, 8))


@pytest.fixture(scope="module")
def steering_vec():
    torch.manual_seed(42)
    return torch.randn(TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# SteeringConfig tests
# ---------------------------------------------------------------------------

class TestSteeringConfig:
    def test_default_alpha(self):
        cfg = SteeringConfig()
        assert cfg.alpha == 1.0

    def test_default_layer_idx(self):
        cfg = SteeringConfig()
        assert cfg.layer_idx == -1

    def test_default_method(self):
        cfg = SteeringConfig()
        assert cfg.method == "add"

    def test_custom_values(self):
        cfg = SteeringConfig(alpha=2.5, layer_idx=0, method="project")
        assert cfg.alpha == 2.5
        assert cfg.layer_idx == 0
        assert cfg.method == "project"

    def test_lerp_method(self):
        cfg = SteeringConfig(method="lerp")
        assert cfg.method == "lerp"


# ---------------------------------------------------------------------------
# extract_hidden_states tests
# ---------------------------------------------------------------------------

class TestExtractHiddenStates:
    def test_shape_layer_0(self, model, input_ids):
        B, T = input_ids.shape
        h = extract_hidden_states(model, input_ids, layer_idx=0)
        assert h.shape == (B, T, TINY_CFG.d_model)

    def test_shape_layer_1(self, model, input_ids):
        B, T = input_ids.shape
        h = extract_hidden_states(model, input_ids, layer_idx=1)
        assert h.shape == (B, T, TINY_CFG.d_model)

    def test_negative_layer_idx(self, model, input_ids):
        """Negative layer index should resolve to the last layer."""
        B, T = input_ids.shape
        h_neg = extract_hidden_states(model, input_ids, layer_idx=-1)
        h_pos = extract_hidden_states(model, input_ids, layer_idx=TINY_CFG.n_layers - 1)
        assert h_neg.shape == (B, T, TINY_CFG.d_model)
        assert torch.allclose(h_neg, h_pos)

    def test_returns_tensor(self, model, input_ids):
        h = extract_hidden_states(model, input_ids, layer_idx=0)
        assert isinstance(h, torch.Tensor)

    def test_out_of_range_layer_idx(self, model, input_ids):
        with pytest.raises(IndexError):
            extract_hidden_states(model, input_ids, layer_idx=99)


# ---------------------------------------------------------------------------
# compute_steering_vector tests
# ---------------------------------------------------------------------------

class TestComputeSteeringVector:
    def test_shape(self, model):
        torch.manual_seed(1)
        pos = torch.randint(0, TINY_CFG.vocab_size, (2, 8))
        neg = torch.randint(0, TINY_CFG.vocab_size, (2, 8))
        vec = compute_steering_vector(model, pos, neg, layer_idx=0)
        assert vec.shape == (TINY_CFG.d_model,)

    def test_shape_negative_idx(self, model):
        torch.manual_seed(2)
        pos = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
        neg = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
        vec = compute_steering_vector(model, pos, neg, layer_idx=-1)
        assert vec.shape == (TINY_CFG.d_model,)

    def test_identical_inputs_zero_vector(self, model):
        """Same pos and neg -> steering vector should be (near) zero."""
        torch.manual_seed(3)
        ids = torch.randint(0, TINY_CFG.vocab_size, (2, 6))
        vec = compute_steering_vector(model, ids, ids, layer_idx=0)
        assert torch.allclose(vec, torch.zeros_like(vec), atol=1e-5)

    def test_returns_tensor(self, model):
        pos = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
        neg = torch.randint(0, TINY_CFG.vocab_size, (1, 4))
        vec = compute_steering_vector(model, pos, neg, layer_idx=0)
        assert isinstance(vec, torch.Tensor)


# ---------------------------------------------------------------------------
# apply_steering_hook tests
# ---------------------------------------------------------------------------

class TestApplySteeringHook:
    def test_returns_handle_with_remove(self, model, steering_vec):
        handle = apply_steering_hook(model, steering_vec, alpha=1.0,
                                     method="add", layer_idx=0)
        assert hasattr(handle, "remove")
        assert callable(handle.remove)
        handle.remove()

    def test_hook_changes_output_add(self, model, input_ids, steering_vec):
        """Output logits should differ after attaching an 'add' hook."""
        with torch.no_grad():
            _, logits_before, _ = model(input_ids)

        handle = apply_steering_hook(model, steering_vec, alpha=1.0,
                                     method="add", layer_idx=0)
        with torch.no_grad():
            _, logits_after, _ = model(input_ids)
        handle.remove()

        assert not torch.allclose(logits_before, logits_after)

    def test_hook_changes_output_project(self, model, input_ids, steering_vec):
        """Output logits should differ after attaching a 'project' hook."""
        with torch.no_grad():
            _, logits_before, _ = model(input_ids)

        handle = apply_steering_hook(model, steering_vec, alpha=1.0,
                                     method="project", layer_idx=0)
        with torch.no_grad():
            _, logits_after, _ = model(input_ids)
        handle.remove()

        assert not torch.allclose(logits_before, logits_after)

    def test_hook_removed_after_remove(self, model, input_ids, steering_vec):
        """After remove(), output should return to baseline."""
        with torch.no_grad():
            _, logits_baseline, _ = model(input_ids)

        handle = apply_steering_hook(model, steering_vec, alpha=5.0,
                                     method="add", layer_idx=0)
        handle.remove()

        with torch.no_grad():
            _, logits_after_remove, _ = model(input_ids)

        assert torch.allclose(logits_baseline, logits_after_remove)

    def test_lerp_method_hook(self, model, input_ids, steering_vec):
        """'lerp' method hook should also change outputs."""
        with torch.no_grad():
            _, logits_before, _ = model(input_ids)

        handle = apply_steering_hook(model, steering_vec, alpha=1.0,
                                     method="lerp", layer_idx=0)
        with torch.no_grad():
            _, logits_after, _ = model(input_ids)
        handle.remove()

        assert not torch.allclose(logits_before, logits_after)

    def test_invalid_method_raises(self, model, steering_vec, input_ids):
        handle = apply_steering_hook(model, steering_vec, alpha=1.0,
                                     method="invalid_method", layer_idx=0)
        with pytest.raises(ValueError):
            with torch.no_grad():
                model(input_ids)
        handle.remove()


# ---------------------------------------------------------------------------
# lerp_hidden tests
# ---------------------------------------------------------------------------

class TestLerpHidden:
    def test_output_shape(self):
        h = torch.randn(2, 8, 64)
        direction = torch.randn(64)
        out = lerp_hidden(h, direction, alpha=1.0)
        assert out.shape == h.shape

    def test_alpha_zero_no_change(self):
        h = torch.randn(2, 8, 64)
        direction = torch.randn(64)
        out = lerp_hidden(h, direction, alpha=0.0)
        assert torch.allclose(out, h)

    def test_nonzero_alpha_changes_values(self):
        h = torch.randn(2, 8, 64)
        direction = torch.randn(64)
        out = lerp_hidden(h, direction, alpha=2.0)
        assert not torch.allclose(out, h)

    def test_zero_direction_returns_unchanged(self):
        """If the direction has zero norm, output equals input."""
        h = torch.randn(2, 8, 64)
        direction = torch.zeros(64)
        out = lerp_hidden(h, direction, alpha=1.0)
        assert torch.allclose(out, h)


# ---------------------------------------------------------------------------
# LatentSteerer tests
# ---------------------------------------------------------------------------

class TestLatentSteerer:
    def _make_steerer(self, model, method="add"):
        cfg = SteeringConfig(alpha=2.0, layer_idx=-1, method=method)
        return LatentSteerer(model, cfg)

    def test_enable_registers_hook(self, model, steering_vec):
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        steerer.enable()
        assert steerer._handle is not None
        steerer.disable()

    def test_disable_removes_hook(self, model, steering_vec):
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        steerer.enable()
        steerer.disable()
        assert steerer._handle is None

    def test_enable_without_vector_raises(self, model):
        steerer = self._make_steerer(model)
        with pytest.raises(RuntimeError):
            steerer.enable()

    def test_steer_generate_returns_tensor(self, model, steering_vec, input_ids):
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        output = steerer.steer_generate(input_ids, max_new_tokens=3)
        assert isinstance(output, torch.Tensor)

    def test_steer_generate_output_longer_than_input(self, model, steering_vec, input_ids):
        B, T = input_ids.shape
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        output = steerer.steer_generate(input_ids, max_new_tokens=5)
        assert output.shape[0] == B
        assert output.shape[1] >= T + 1

    def test_steer_generate_disables_hook_after(self, model, steering_vec, input_ids):
        """steer_generate() should leave the hook disabled when called fresh."""
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        steerer.steer_generate(input_ids, max_new_tokens=2)
        assert steerer._handle is None

    def test_double_enable_replaces_hook(self, model, steering_vec):
        steerer = self._make_steerer(model)
        steerer.set_steering_vector(steering_vec)
        steerer.enable()
        handle1 = steerer._handle
        steerer.enable()  # should replace without error
        handle2 = steerer._handle
        assert handle1 is not handle2
        steerer.disable()
