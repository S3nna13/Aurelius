"""Integration tests for PARL reward via ALIGNMENT_REGISTRY."""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Test 1 — "parl" key is in ALIGNMENT_REGISTRY
# ---------------------------------------------------------------------------

def test_parl_in_registry():
    from src.alignment import ALIGNMENT_REGISTRY

    assert "parl" in ALIGNMENT_REGISTRY, (
        '"parl" must be registered in ALIGNMENT_REGISTRY'
    )


# ---------------------------------------------------------------------------
# Test 2 — construct from registry, call with tensors, correct shape
# ---------------------------------------------------------------------------

def test_construct_from_registry_and_call():
    from src.alignment import ALIGNMENT_REGISTRY

    PARLRewardCls = ALIGNMENT_REGISTRY["parl"]
    reward_fn = PARLRewardCls(lambda1=1.0, lambda2=1.0, total_steps=10_000)

    B = 6
    r_perf = torch.rand(B)
    r_parallel = torch.rand(B)
    r_finish = torch.rand(B)

    out = reward_fn(r_perf, r_parallel, r_finish, step=0)

    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"
    assert torch.isfinite(out).all(), "Output must be finite"


# ---------------------------------------------------------------------------
# Test 3 — regression guard: existing registry keys are still present
# ---------------------------------------------------------------------------

def test_existing_registry_keys_intact():
    """Ensure adding 'parl' did not remove any pre-existing registry entries."""
    from src.alignment import ALIGNMENT_REGISTRY

    # Keys that must survive across all cycles
    required_keys = {"prm", "adversarial_code_battle", "constitution_dimensions"}
    missing = required_keys - set(ALIGNMENT_REGISTRY.keys())
    assert not missing, (
        f"Registry regression: the following keys disappeared: {missing}"
    )
