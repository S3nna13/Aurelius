"""Integration tests for CrossStageDistillation registry wiring.

Verifies:
1. "cross_stage_distillation" key is present in ALIGNMENT_REGISTRY.
2. The class can be constructed from the registry and .loss() returns a finite scalar.
3. Pre-existing alignment registry keys are still present (regression guard).
"""

import torch

from src.alignment import ALIGNMENT_REGISTRY


def test_registry_contains_cross_stage_distillation():
    """ALIGNMENT_REGISTRY must expose the cross_stage_distillation key."""
    assert "cross_stage_distillation" in ALIGNMENT_REGISTRY, (
        f"'cross_stage_distillation' not found in ALIGNMENT_REGISTRY. "
        f"Present keys: {list(ALIGNMENT_REGISTRY.keys())}"
    )


def test_construct_from_registry_and_call_loss():
    """Construct CrossStageDistillation from registry; .loss() must return a finite scalar."""
    cls = ALIGNMENT_REGISTRY["cross_stage_distillation"]
    csd = cls(alpha=0.1)

    B, T, V = 2, 8, 32
    rl_loss = torch.tensor(1.0)
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)

    out = csd.loss(rl_loss, student, teacher)

    assert out.shape == torch.Size([]), (
        f"Expected scalar output, got shape {out.shape}"
    )
    assert torch.isfinite(out), f"Expected finite output, got {out}"


def test_existing_registry_keys_still_present():
    """Regression guard: pre-existing keys must not be removed by the new wiring."""
    expected_keys = ["prm", "adversarial_code_battle", "constitution_dimensions", "parl", "toggle"]
    for key in expected_keys:
        assert key in ALIGNMENT_REGISTRY, (
            f"Pre-existing registry key '{key}' was removed — regression detected."
        )
