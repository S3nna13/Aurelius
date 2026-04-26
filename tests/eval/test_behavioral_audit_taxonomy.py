"""Tests for src/eval/behavioral_audit_taxonomy.py."""

from __future__ import annotations

import warnings

import pytest

# VALID_POLARITIES is a module-level constant; import via module to be safe.
from src.eval import behavioral_audit_taxonomy as bat
from src.eval.behavioral_audit_taxonomy import (
    BEHAVIORAL_AUDIT_TAXONOMY,
    AuditResult,
    BehavioralAuditRegistry,
    BehaviorCategory,
    BehaviorDimension,
    by_category,
    by_id,
)

EXPECTED_COUNTS = {
    BehaviorCategory.HARMFUL: 12,
    BehaviorCategory.UNCOOPERATIVE: 3,
    BehaviorCategory.MISLEADING: 8,
    BehaviorCategory.MODEL_INITIATED: 6,
    BehaviorCategory.EVALUATION_OBSTACLE: 4,
    BehaviorCategory.CHARACTER_TRAIT: 8,  # 7 traits + character_drift
}


def test_all_40_dimensions_loaded():
    assert len(BEHAVIORAL_AUDIT_TAXONOMY) == 41 or len(BEHAVIORAL_AUDIT_TAXONOMY) == sum(
        EXPECTED_COUNTS.values()
    )
    # spec says 30+ named dims plus 7 positive character traits; our count is
    # 12+3+8+6+4+8 = 41. Validate exact count.
    assert len(BEHAVIORAL_AUDIT_TAXONOMY) == 41


def test_ids_are_unique():
    ids = [d.id for d in BEHAVIORAL_AUDIT_TAXONOMY]
    assert len(ids) == len(set(ids))


def test_polarity_values_are_valid():
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        assert d.polarity in bat.VALID_POLARITIES


def test_categories_partition_correctly():
    counts = {c: 0 for c in BehaviorCategory}
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        counts[d.category] += 1
    assert counts[BehaviorCategory.HARMFUL] == 12
    assert counts[BehaviorCategory.UNCOOPERATIVE] == 3
    assert counts[BehaviorCategory.MISLEADING] == 8
    assert counts[BehaviorCategory.MODEL_INITIATED] == 6
    assert counts[BehaviorCategory.EVALUATION_OBSTACLE] == 4
    # CHARACTER_TRAIT = 7 positives + character_drift
    assert counts[BehaviorCategory.CHARACTER_TRAIT] == 8
    assert sum(counts.values()) == len(BEHAVIORAL_AUDIT_TAXONOMY)


def test_registry_register_and_lookup():
    reg = BehavioralAuditRegistry()

    def grader(dim, traj):
        return (0.5, "stub")

    reg.register("harmful.fraud", grader)
    assert "harmful.fraud" in reg
    assert reg.get("harmful.fraud") is grader


def test_registry_rejects_unknown_dim():
    reg = BehavioralAuditRegistry()
    with pytest.raises(KeyError):
        reg.register("not.a.real.dim", lambda d, t: (0.0, ""))


def test_audit_with_no_graders_returns_zeros():
    reg = BehavioralAuditRegistry()
    # wipe defaults
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        reg.unregister(d.id)
    res = reg.audit("hello")
    assert isinstance(res, AuditResult)
    assert len(res.scores) == len(BEHAVIORAL_AUDIT_TAXONOMY)
    assert all(s.score == 0.0 for s in res.scores)
    assert all(s.rationale == "no grader registered" for s in res.scores)
    assert res.overall == 0.0
    for cat, v in res.category_summaries.items():
        assert v == 0.0


def test_audit_with_registered_graders():
    reg = BehavioralAuditRegistry()
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        reg.unregister(d.id)

    def half(dim, traj):
        return (0.5, "half")

    reg.register("harmful.fraud", half)
    res = reg.audit("test")
    fraud = next(s for s in res.scores if s.dimension_id == "harmful.fraud")
    assert fraud.score == 0.5
    assert fraud.rationale == "half"


def test_grader_exception_recorded_as_zero_with_warning():
    reg = BehavioralAuditRegistry()

    def boom(dim, traj):
        raise RuntimeError("kaboom")

    reg.register("harmful.fraud", boom)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = reg.audit("x")
    fraud = next(s for s in res.scores if s.dimension_id == "harmful.fraud")
    assert fraud.score == 0.0
    assert "RuntimeError" in fraud.rationale
    assert any(issubclass(rec.category, RuntimeWarning) for rec in w), "expected a RuntimeWarning"


def test_category_summaries_average_correctly():
    reg = BehavioralAuditRegistry()
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        reg.unregister(d.id)
    # register graders returning 1.0 for all UNCOOPERATIVE dims
    uncoop = by_category(BehaviorCategory.UNCOOPERATIVE)
    for d in uncoop:
        reg.register(d.id, lambda dim, traj: (1.0, "full"))
    res = reg.audit("x")
    assert res.category_summaries[BehaviorCategory.UNCOOPERATIVE] == 1.0
    assert res.category_summaries[BehaviorCategory.HARMFUL] == 0.0


def test_overall_bounded():
    reg = BehavioralAuditRegistry()

    def over(dim, traj):
        return (99.0, "out of range")  # clamps to 1.0

    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        reg.register(d.id, over)
    res = reg.audit("x")
    assert 0.0 <= res.overall <= 1.0
    assert res.overall == 1.0
    for s in res.scores:
        assert 0.0 <= s.score <= 1.0


def test_by_id_missing_raises_keyerror():
    with pytest.raises(KeyError):
        by_id("totally.missing")


def test_by_id_returns_dimension():
    d = by_id("harmful.fraud")
    assert isinstance(d, BehaviorDimension)
    assert d.id == "harmful.fraud"
    assert d.category is BehaviorCategory.HARMFUL


def test_by_category_returns_subset():
    out = by_category(BehaviorCategory.UNCOOPERATIVE)
    assert len(out) == 3
    assert all(d.category is BehaviorCategory.UNCOOPERATIVE for d in out)


def test_by_category_type_error():
    with pytest.raises(TypeError):
        by_category("harmful")  # type: ignore[arg-type]


def test_determinism():
    reg1 = BehavioralAuditRegistry()
    reg2 = BehavioralAuditRegistry()
    r1 = reg1.audit("some trajectory")
    r2 = reg2.audit("some trajectory")
    assert [s.score for s in r1.scores] == [s.score for s in r2.scores]
    assert r1.overall == r2.overall


def test_weight_override():
    reg = BehavioralAuditRegistry()
    for d in BEHAVIORAL_AUDIT_TAXONOMY:
        reg.unregister(d.id)
    harm = by_category(BehaviorCategory.HARMFUL)
    # one dim gets 1.0; everything else 0.0; overriding weight to isolate it
    reg.register(harm[0].id, lambda dim, traj: (1.0, "hit"))
    for d in harm[1:]:
        reg.register(d.id, lambda dim, traj: (0.0, "miss"))
    weights = {d.id: 0.0 for d in harm}
    weights[harm[0].id] = 1.0
    res = reg.audit("x", weights=weights)
    assert res.category_summaries[BehaviorCategory.HARMFUL] == 1.0


def test_polarity_invalid_rejected():
    with pytest.raises(ValueError):
        BehaviorDimension(
            id="x.y",
            name="x",
            category=BehaviorCategory.HARMFUL,
            description="d",
            polarity="neutral",
        )


def test_grader_bad_return_recorded_as_error():
    reg = BehavioralAuditRegistry()
    reg.register("harmful.fraud", lambda dim, traj: "not a tuple")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = reg.audit("x")
    fraud = next(s for s in res.scores if s.dimension_id == "harmful.fraud")
    assert fraud.score == 0.0
    assert "grader error" in fraud.rationale


def test_default_graders_present():
    reg = BehavioralAuditRegistry()
    assert "uncooperative.overrefusal" in reg
    assert "misleading.sycophancy" in reg
