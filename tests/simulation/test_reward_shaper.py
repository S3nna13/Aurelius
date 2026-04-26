"""Tests for src/simulation/reward_shaper.py — 10+ tests."""

import pytest

from src.simulation.reward_shaper import (
    PotentialFunction,
    RewardShaper,
    RewardShaperConfig,
    RewardShapeType,
    _DefaultPotential,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_shaper():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.DENSE)
    return RewardShaper(cfg)


@pytest.fixture
def clip_shaper():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.CLIP, clip_range=(-1.0, 1.0))
    return RewardShaper(cfg)


@pytest.fixture
def curiosity_shaper():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.CURIOSITY, curiosity_beta=1.0)
    return RewardShaper(cfg)


class _DistancePotential:
    """Simple potential: negative manhattan distance to (4, 4)."""

    def __call__(self, state: dict) -> float:
        return -(abs(state.get("x", 0) - 4) + abs(state.get("y", 0) - 4))


# ---------------------------------------------------------------------------
# 1. DENSE: pass-through within clip_range
# ---------------------------------------------------------------------------


def test_dense_passthrough(dense_shaper):
    r = dense_shaper.shape(0.5, {}, {}, done=False)
    assert r == pytest.approx(0.5)


def test_dense_clip_upper(dense_shaper):
    # Default clip_range is (-10, 10)
    r = dense_shaper.shape(50.0, {}, {}, done=False)
    assert r == pytest.approx(10.0)


def test_dense_clip_lower(dense_shaper):
    r = dense_shaper.shape(-50.0, {}, {}, done=False)
    assert r == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# 2. SPARSE
# ---------------------------------------------------------------------------


def test_sparse_zero_on_non_terminal():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.SPARSE)
    shaper = RewardShaper(cfg)
    r = shaper.shape(5.0, {}, {}, done=False)
    assert r == pytest.approx(0.0)


def test_sparse_preserves_on_terminal():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.SPARSE)
    shaper = RewardShaper(cfg)
    r = shaper.shape(5.0, {}, {}, done=True)
    assert r == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 3. POTENTIAL_BASED
# ---------------------------------------------------------------------------


def test_potential_based_shaping():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.POTENTIAL_BASED, gamma=0.99)
    phi = _DistancePotential()
    shaper = RewardShaper(cfg, potential_fn=phi)
    s = {"x": 0, "y": 0}  # phi = -(4+4) = -8
    s2 = {"x": 1, "y": 0}  # phi = -(3+4) = -7
    raw = -0.01
    expected = raw + 0.99 * (-7) - (-8)
    r = shaper.shape(raw, s, s2, done=False)
    assert r == pytest.approx(expected, abs=1e-5)


def test_potential_based_default_zero_potential():
    cfg = RewardShaperConfig(shape_type=RewardShapeType.POTENTIAL_BASED, gamma=0.99)
    shaper = RewardShaper(cfg)
    # With zero potential, shaped == raw
    r = shaper.shape(1.0, {"x": 0}, {"x": 1}, done=False)
    assert r == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. CURIOSITY
# ---------------------------------------------------------------------------


def test_curiosity_bonus_first_visit(curiosity_shaper):
    # beta=1.0, count=1 → bonus = 1/sqrt(1) = 1.0
    r = curiosity_shaper.shape(0.0, {}, {"x": 0}, done=False)
    assert r == pytest.approx(1.0)


def test_curiosity_bonus_decreases_with_visits(curiosity_shaper):
    state = {"x": 42}
    r1 = curiosity_shaper.shape(0.0, {}, state, done=False)
    r2 = curiosity_shaper.shape(0.0, {}, state, done=False)
    assert r2 < r1


def test_curiosity_reset_clears_counts(curiosity_shaper):
    state = {"x": 1}
    curiosity_shaper.shape(0.0, {}, state, done=False)
    curiosity_shaper.shape(0.0, {}, state, done=False)
    curiosity_shaper.reset_visit_counts()
    # After reset, first visit again → bonus = 1.0
    r = curiosity_shaper.shape(0.0, {}, state, done=False)
    assert r == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. CLIP
# ---------------------------------------------------------------------------


def test_clip_strategy(clip_shaper):
    assert clip_shaper.shape(5.0, {}, {}, done=False) == pytest.approx(1.0)
    assert clip_shaper.shape(-5.0, {}, {}, done=False) == pytest.approx(-1.0)
    assert clip_shaper.shape(0.3, {}, {}, done=False) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 6. PotentialFunction protocol
# ---------------------------------------------------------------------------


def test_default_potential_returns_zero():
    phi = _DefaultPotential()
    assert phi({"x": 99}) == pytest.approx(0.0)


def test_custom_potential_satisfies_protocol():
    phi = _DistancePotential()
    assert isinstance(phi, PotentialFunction)


# ---------------------------------------------------------------------------
# 7. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RewardShaperConfig()
    assert cfg.shape_type == RewardShapeType.DENSE
    assert cfg.clip_range == (-10.0, 10.0)
    assert cfg.gamma == pytest.approx(0.99)
    assert cfg.curiosity_beta == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 8. RewardShapeType enum values
# ---------------------------------------------------------------------------


def test_shape_type_values():
    assert RewardShapeType.DENSE == "dense"
    assert RewardShapeType.SPARSE == "sparse"
    assert RewardShapeType.POTENTIAL_BASED == "potential_based"
    assert RewardShapeType.CURIOSITY == "curiosity"
    assert RewardShapeType.CLIP == "clip"
