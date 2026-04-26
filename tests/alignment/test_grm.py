"""Unit tests for src/alignment/grm.py.

Covers all 12 required scenarios from the spec plus additional edge cases.
"""

from __future__ import annotations

import pytest
import torch

from src.alignment.grm import DIMENSIONS, GenerativeRewardModel, GRMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_scores(value: float = 0.5) -> dict:
    return {d: value for d in DIMENSIONS}


def _make_grm(**kw) -> GenerativeRewardModel:
    return GenerativeRewardModel(GRMConfig(**kw))


# ---------------------------------------------------------------------------
# 1. Output shape [B] for batch inputs
# ---------------------------------------------------------------------------


def test_output_shape_scalar():
    """score() returns a scalar tensor (0-dim)."""
    grm = GenerativeRewardModel()
    result = grm.score(_uniform_scores(0.6))
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0


def test_output_shape_tensor_inputs():
    """score() works when dim_scores contain tensors (e.g. [B] shaped)."""
    grm = GenerativeRewardModel()
    scores = {d: torch.tensor([0.5, 0.7, 0.9]) for d in DIMENSIONS}
    result = grm.score(scores)
    # weighted sum of 4 equal-weighted identical tensors → same shape
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# 2. Output range [0, 1] on normal inputs
# ---------------------------------------------------------------------------


def test_output_range_normal():
    grm = GenerativeRewardModel()
    result = grm.score(_uniform_scores(0.75))
    assert 0.0 <= result.item() <= 1.0


def test_output_range_extreme_valid():
    grm = GenerativeRewardModel()
    assert grm.score(_uniform_scores(0.0)).item() == pytest.approx(0.0)
    assert grm.score(_uniform_scores(1.0)).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Weights normalised: custom weights that don't sum to 1 → valid output
# ---------------------------------------------------------------------------


def test_weights_normalised_arbitrary_sum():
    config = GRMConfig(weights={d: 10.0 for d in DIMENSIONS})
    grm = GenerativeRewardModel(config)
    # sum(weights) = 40; each normalised weight = 0.25
    result = grm.score(_uniform_scores(0.8))
    assert result.item() == pytest.approx(0.8, abs=1e-5)


def test_weights_normalised_asymmetric():
    config = GRMConfig(
        weights={"helpfulness": 3.0, "adherence": 1.0, "relevance": 0.0, "detail": 0.0}
    )
    grm = GenerativeRewardModel(config)
    # normalised: helpfulness=0.75, adherence=0.25
    scores = {"helpfulness": 1.0, "adherence": 0.0, "relevance": 0.5, "detail": 0.5}
    result = grm.score(scores)
    assert result.item() == pytest.approx(0.75, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. mode="rule" with rule_reward → returns rule_reward unchanged
# ---------------------------------------------------------------------------


def test_mode_rule_with_reward_float():
    grm = _make_grm(mode="rule")
    result = grm.score(_uniform_scores(1.0), rule_reward=0.42)
    assert result.item() == pytest.approx(0.42)


def test_mode_rule_with_reward_tensor():
    grm = _make_grm(mode="rule")
    rr = torch.tensor(0.99)
    result = grm.score(_uniform_scores(0.0), rule_reward=rr)
    assert result.item() == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 5. mode="rule" without rule_reward → falls through to GRM scoring
# ---------------------------------------------------------------------------


def test_mode_rule_no_rule_reward_fallback():
    grm = _make_grm(mode="rule")
    # No rule_reward → GRM path is used
    result = grm.score(_uniform_scores(0.6), rule_reward=None)
    assert result.item() == pytest.approx(0.6, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. Adversarial: all dim_scores = zero tensors → output finite
# ---------------------------------------------------------------------------


def test_adversarial_all_zero():
    grm = GenerativeRewardModel()
    scores = {d: torch.tensor(0.0) for d in DIMENSIONS}
    result = grm.score(scores)
    assert torch.isfinite(result)
    assert result.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Adversarial: dim_scores out of [0, 1] → clamped output
# ---------------------------------------------------------------------------


def test_adversarial_scores_above_one():
    grm = GenerativeRewardModel()
    result = grm.score(_uniform_scores(5.0))
    assert result.item() == pytest.approx(1.0)


def test_adversarial_scores_below_zero():
    grm = GenerativeRewardModel()
    result = grm.score(_uniform_scores(-3.0))
    assert result.item() == pytest.approx(0.0)


def test_adversarial_mixed_out_of_range():
    grm = GenerativeRewardModel()
    scores = {"helpfulness": 2.0, "adherence": -1.0, "relevance": 0.5, "detail": 0.5}
    result = grm.score(scores)
    assert 0.0 <= result.item() <= 1.0


# ---------------------------------------------------------------------------
# 8. Identical candidates → same score
# ---------------------------------------------------------------------------


def test_identical_candidates_same_score():
    grm = GenerativeRewardModel()
    s = _uniform_scores(0.65)
    r1 = grm.score(s)
    r2 = grm.score(s)
    assert r1.item() == pytest.approx(r2.item())


# ---------------------------------------------------------------------------
# 9. single_dim: only one dimension provided → weighted correctly
# ---------------------------------------------------------------------------


def test_single_dim_only():
    # Give full weight to helpfulness, zero to others
    config = GRMConfig(
        weights={"helpfulness": 1.0, "adherence": 0.0, "relevance": 0.0, "detail": 0.0}
    )
    grm = GenerativeRewardModel(config)
    result = grm.score({"helpfulness": 0.9})
    assert result.item() == pytest.approx(0.9, abs=1e-5)


def test_single_dim_equal_weights():
    """One dim out of four at equal weight → 0.25 * value."""
    grm = GenerativeRewardModel()
    result = grm.score({"helpfulness": 1.0})
    assert result.item() == pytest.approx(0.25, abs=1e-5)


# ---------------------------------------------------------------------------
# 10. missing_dim: dimension in weights but not in dim_scores → 0 contribution
# ---------------------------------------------------------------------------


def test_missing_dim_contributes_zero():
    grm = GenerativeRewardModel()  # weights on all 4 dims
    # Only provide 3 dims; "detail" is missing
    scores = {"helpfulness": 1.0, "adherence": 1.0, "relevance": 1.0}
    result = grm.score(scores)
    # 3 * 0.25 * 1.0 = 0.75
    assert result.item() == pytest.approx(0.75, abs=1e-5)


# ---------------------------------------------------------------------------
# 11. mode="grm" ignores rule_reward
# ---------------------------------------------------------------------------


def test_mode_grm_ignores_rule_reward():
    grm = _make_grm(mode="grm")
    result_no_rr = grm.score(_uniform_scores(0.5))
    result_with_rr = grm.score(_uniform_scores(0.5), rule_reward=0.999)
    assert result_no_rr.item() == pytest.approx(result_with_rr.item())


# ---------------------------------------------------------------------------
# 12. Determinism: same inputs → same output
# ---------------------------------------------------------------------------


def test_determinism():
    grm = GenerativeRewardModel()
    scores = {"helpfulness": 0.7, "adherence": 0.8, "relevance": 0.6, "detail": 0.9}
    results = [grm.score(scores).item() for _ in range(10)]
    assert all(r == pytest.approx(results[0]) for r in results)
