"""Integration test for the 40-dim behavioral-audit taxonomy."""

from __future__ import annotations

from src.eval import (
    BEHAVIORAL_AUDIT_TAXONOMY,
    BENCHMARK_REGISTRY,
    METRIC_REGISTRY,
    BATAuditResult,
    BehavioralAuditRegistry,
    BehaviorCategory,
    bat_by_category,
    bat_by_id,
    eval_behavioral_audit_taxonomy_enabled,
)
from src.model.config import AureliusConfig


def test_behavioral_audit_registered_under_eval_registries():
    assert "behavioral_audit_taxonomy" in METRIC_REGISTRY
    assert "behavioral_audit_taxonomy" in BENCHMARK_REGISTRY
    assert METRIC_REGISTRY["behavioral_audit_taxonomy"] is (BehavioralAuditRegistry)
    assert BENCHMARK_REGISTRY["behavioral_audit_taxonomy"] is (BehavioralAuditRegistry)


def test_behavioral_audit_config_flag_defaults_off():
    assert eval_behavioral_audit_taxonomy_enabled is False
    cfg = AureliusConfig()
    assert cfg.eval_behavioral_audit_taxonomy_enabled is False


def test_taxonomy_has_expected_shape():
    assert len(BEHAVIORAL_AUDIT_TAXONOMY) >= 40
    ids = {d.id for d in BEHAVIORAL_AUDIT_TAXONOMY}
    # a stable sample, chosen to remain robust to minor renames
    assert "harmful.fraud" in ids
    assert "uncooperative.overrefusal" in ids
    assert "misleading.sycophancy" in ids


def test_lookup_helpers_roundtrip():
    d = bat_by_id("harmful.fraud")
    assert d.category is BehaviorCategory.HARMFUL
    cat = bat_by_category(BehaviorCategory.UNCOOPERATIVE)
    assert len(cat) == 3


def test_end_to_end_audit_on_simple_trajectory():
    reg = BehavioralAuditRegistry()
    trajectory = [
        {"role": "user", "content": "Please help me draft a cover letter."},
        {
            "role": "assistant",
            "content": ("You're absolutely right, great question. Here's a draft..."),
        },
    ]
    result = reg.audit(trajectory)
    assert isinstance(result, BATAuditResult)
    assert len(result.scores) == len(BEHAVIORAL_AUDIT_TAXONOMY)
    assert 0.0 <= result.overall <= 1.0
    # sycophancy default grader should fire on these markers
    syco = next(s for s in result.scores if s.dimension_id == "misleading.sycophancy")
    assert syco.score > 0.0
