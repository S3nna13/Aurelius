"""Tests for src/model/latent_interp_utils.py"""

from __future__ import annotations

import torch

from src.model.latent_interp_utils import (
    InterpolationPath,
    LatentMixer,
    lerp_states,
    slerp,
    slerp_states,
)

D = 32
B = 4


# ---------------------------------------------------------------------------
# 1. slerp output shape
# ---------------------------------------------------------------------------


def test_slerp_output_shape():
    """slerp of two (D,) tensors should return shape (D,)."""
    v0 = torch.randn(D)
    v1 = torch.randn(D)
    result = slerp(v0, v1, 0.5)
    assert result.shape == (D,)


# ---------------------------------------------------------------------------
# 2. lerp endpoints equal v0 at t=0 and v1 at t=1
# ---------------------------------------------------------------------------


def test_lerp_states_t0_equals_v0():
    """lerp_states at t=0 should return states identical to states_a."""
    a = [torch.randn(B, D) for _ in range(3)]
    b = [torch.randn(B, D) for _ in range(3)]
    result = lerp_states(a, b, 0.0)
    for ra, a_i in zip(result, a):
        assert torch.allclose(ra, a_i)


def test_lerp_states_t1_equals_v1():
    """lerp_states at t=1 should return states identical to states_b."""
    a = [torch.randn(B, D) for _ in range(3)]
    b = [torch.randn(B, D) for _ in range(3)]
    result = lerp_states(a, b, 1.0)
    for rb, b_i in zip(result, b):
        assert torch.allclose(rb, b_i)


# ---------------------------------------------------------------------------
# 3. slerp endpoints
# ---------------------------------------------------------------------------


def test_slerp_t0_direction_matches_v0():
    """slerp at t=0 should have same direction as v0."""
    torch.manual_seed(0)
    v0 = torch.randn(D)
    v1 = torch.randn(D)
    result = slerp(v0, v1, 0.0)
    eps = 1e-6
    cos_sim = torch.dot(result / (result.norm() + eps), v0 / (v0.norm() + eps))
    assert cos_sim.item() > 0.999


def test_slerp_t1_direction_matches_v1():
    """slerp at t=1 should have same direction as v1."""
    torch.manual_seed(1)
    v0 = torch.randn(D)
    v1 = torch.randn(D)
    result = slerp(v0, v1, 1.0)
    eps = 1e-6
    cos_sim = torch.dot(result / (result.norm() + eps), v1 / (v1.norm() + eps))
    assert cos_sim.item() > 0.999


# ---------------------------------------------------------------------------
# 4. Parallel vector fallback
# ---------------------------------------------------------------------------


def test_slerp_parallel_vector_fallback():
    """Parallel vectors should fall back to lerp without NaN."""
    v = torch.randn(D)
    result = slerp(v, v.clone(), 0.5)
    assert not torch.isnan(result).any()
    # Should point in the same direction as v
    eps = 1e-6
    cos_sim = torch.dot(result / (result.norm() + eps), v / (v.norm() + eps))
    assert cos_sim.item() > 0.99


# ---------------------------------------------------------------------------
# 5. slerp_states list length
# ---------------------------------------------------------------------------


def test_slerp_states_list_length():
    """slerp_states should return a list of the same length as inputs."""
    n_layers = 6
    a = [torch.randn(B, D) for _ in range(n_layers)]
    b = [torch.randn(B, D) for _ in range(n_layers)]
    result = slerp_states(a, b, 0.5)
    assert len(result) == n_layers


# ---------------------------------------------------------------------------
# 6. LatentMixer output shape
# ---------------------------------------------------------------------------


def test_latent_mixer_output_shape():
    """LatentMixer.forward should preserve input shape."""
    mixer = LatentMixer()
    h_a = torch.randn(B, D)
    h_b = torch.randn(B, D)
    out = mixer(h_a, h_b)
    assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# 7. LatentMixer gradient
# ---------------------------------------------------------------------------


def test_latent_mixer_gradient():
    """Gradients should flow through LatentMixer to alpha."""
    mixer = LatentMixer()
    h_a = torch.randn(B, D)
    h_b = torch.randn(B, D)
    out = mixer(h_a, h_b)
    out.sum().backward()
    assert mixer.alpha.grad is not None


# ---------------------------------------------------------------------------
# 8. InterpolationPath length
# ---------------------------------------------------------------------------


def test_interpolation_path_length():
    """generate_path should return exactly `steps` tensors."""
    steps = 7
    path_gen = InterpolationPath(steps=steps, mode="slerp")
    v0 = torch.randn(D)
    v1 = torch.randn(D)
    path = path_gen.generate_path(v0, v1)
    assert len(path) == steps


# ---------------------------------------------------------------------------
# 9. Mixed precision float32
# ---------------------------------------------------------------------------


def test_slerp_float32_output():
    """slerp should return float32 given float32 inputs."""
    v0 = torch.randn(D, dtype=torch.float32)
    v1 = torch.randn(D, dtype=torch.float32)
    result = slerp(v0, v1, 0.5)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# 10. t=0.5 midpoint norm is between v0 and v1 norms
# ---------------------------------------------------------------------------


def test_slerp_midpoint_norm_between_endpoints():
    """At t=0.5 the result norm should be roughly between v0 and v1 norms."""
    torch.manual_seed(2)
    v0 = torch.randn(D)
    v1 = torch.randn(D)
    mid = slerp(v0, v1, 0.5)

    norm0 = v0.norm().item()
    norm1 = v1.norm().item()
    norm_mid = mid.norm().item()

    lo = min(norm0, norm1) * 0.5
    hi = max(norm0, norm1) * 2.0
    assert lo <= norm_mid <= hi, f"mid norm {norm_mid:.4f} not between {norm0:.4f} and {norm1:.4f}"
