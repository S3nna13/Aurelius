"""Integration tests for the GRM entry in ALIGNMENT_REGISTRY.

Verifies:
1. "grm" key exists in ALIGNMENT_REGISTRY after import.
2. The class can be constructed from the registry and score a 4-dim dict.
3. Pre-existing registry keys are undisturbed.
"""
from __future__ import annotations

import torch
import pytest

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.grm import DIMENSIONS, GenerativeRewardModel


# ---------------------------------------------------------------------------
# 1. "grm" in ALIGNMENT_REGISTRY
# ---------------------------------------------------------------------------

def test_grm_key_in_registry():
    assert "grm" in ALIGNMENT_REGISTRY, (
        "'grm' not found in ALIGNMENT_REGISTRY — check src/alignment/__init__.py"
    )


def test_grm_registry_value_is_class():
    cls = ALIGNMENT_REGISTRY["grm"]
    assert cls is GenerativeRewardModel


# ---------------------------------------------------------------------------
# 2. Construct from registry, score with 4-dim dict, check shape
# ---------------------------------------------------------------------------

def test_registry_construct_and_score():
    cls = ALIGNMENT_REGISTRY["grm"]
    grm = cls()  # default config

    dim_scores = {d: 0.75 for d in DIMENSIONS}
    result = grm.score(dim_scores)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert 0.0 <= result.item() <= 1.0


def test_registry_construct_with_custom_config():
    from src.alignment.grm import GRMConfig
    cls = ALIGNMENT_REGISTRY["grm"]
    config = GRMConfig(
        weights={"helpfulness": 2.0, "adherence": 1.0,
                 "relevance": 1.0, "detail": 0.0},
        mode="grm",
    )
    grm = cls(config)
    scores = {d: 1.0 for d in DIMENSIONS}
    result = grm.score(scores)
    # All provided scores = 1.0; normalised weights sum to 1 → result = 1.0
    assert result.item() == pytest.approx(1.0, abs=1e-5)


def test_registry_rule_mode_via_registry():
    """Hybrid rule mode works when constructed through the registry."""
    cls = ALIGNMENT_REGISTRY["grm"]
    from src.alignment.grm import GRMConfig
    grm = cls(GRMConfig(mode="rule"))
    result = grm.score({"helpfulness": 0.1}, rule_reward=0.55)
    assert result.item() == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 3. Existing registry keys are undisturbed
# ---------------------------------------------------------------------------

def test_existing_prm_key_intact():
    assert "prm" in ALIGNMENT_REGISTRY, (
        "Pre-existing 'prm' key was removed from ALIGNMENT_REGISTRY"
    )


def test_existing_constitution_dimensions_key_intact():
    assert "constitution_dimensions" in ALIGNMENT_REGISTRY


def test_existing_adversarial_code_battle_key_intact():
    assert "adversarial_code_battle" in ALIGNMENT_REGISTRY
