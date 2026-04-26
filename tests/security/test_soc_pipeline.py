"""Unit tests for src/security/soc_pipeline.py (Aurelius cycle-121)."""

from __future__ import annotations

import pytest

from src.security.soc_pipeline import (
    DEFAULT_SOC_PIPELINE,
    DecideStage,
    EnrichStage,
    Finding,
    IngestStage,
    NormalizeStage,
    PipelineStage,
    RouteStage,
    Rule,
    SecurityEvent,
    SOCPipeline,
    TriageDecision,
    default_route_table,
    evidence_first_gate,
)


def _raw(**over):
    base = {
        "event_id": "E-1",
        "source": "test",
        "severity": "HIGH",
        "timestamp": "2026-04-20T10:00:00Z",
        "payload": {"src_ip": "10.0.0.1"},
    }
    base.update(over)
    return base


def _decision(decision="alert_human", priority="P2", requires_human=False):
    return TriageDecision(
        decision=decision,
        priority=priority,
        rationale="unit test",
        evidence_refs=[],
        recommended_action="review",
        requires_human=requires_human,
    )


def test_default_pipeline_end_to_end():
    ev = DEFAULT_SOC_PIPELINE.run_one(_raw())
    assert isinstance(ev, SecurityEvent)
    assert ev.event_id == "E-1"
    assert ev.severity in ("high", "critical", "medium", "low", "info")
    assert ev.route is not None


def test_empty_stages_passthrough():
    pipe = SOCPipeline([])
    ev = pipe.run_one(_raw())
    assert ev.route is None
    assert ev.decisions == []


def test_normalize_maps_informational_to_info():
    pipe = SOCPipeline([NormalizeStage()])
    ev = pipe.run_one(_raw(severity="Informational"))
    assert ev.severity == "info"


def test_enricher_contributes_keys():
    def _add_tag(event):
        return {"tag": "suspicious"}

    pipe = SOCPipeline([EnrichStage(enricher_fns=[_add_tag])])
    ev = pipe.run_one(_raw())
    assert ev.enrichments.get("tag") == "suspicious"


def test_enricher_non_dict_raises_pipeline_error():
    def _bad(event):
        return "not a dict"  # type: ignore[return-value]

    pipe = SOCPipeline([EnrichStage(enricher_fns=[_bad])])
    ev = pipe.run_one(_raw())
    assert "pipeline_error" in ev.enrichments
    assert ev.enrichments["pipeline_error"]["stage"] == "enrich"


def test_decider_produces_triage_decision():
    def _decide(event):
        return _decision(decision="escalate", priority="P1")

    pipe = SOCPipeline([DecideStage(decider_fn=_decide), RouteStage()])
    ev = pipe.run_one(_raw())
    assert len(ev.decisions) == 1
    assert ev.decisions[-1].decision == "escalate"
    assert ev.route == "oncall_page"


def test_rule_first_match_wins():
    def _true(event):
        return True

    rule_a = Rule(name="a", predicate=_true, decision=_decision(decision="suppress"))
    rule_b = Rule(name="b", predicate=_true, decision=_decision(decision="escalate"))
    pipe = SOCPipeline([DecideStage(rules=[rule_a, rule_b])])
    ev = pipe.run_one(_raw())
    assert ev.decisions[-1].decision == "suppress"


def test_route_set_from_last_decision():
    pipe = SOCPipeline(
        [DecideStage(decider_fn=lambda e: _decision(decision="suppress")), RouteStage()]
    )
    ev = pipe.run_one(_raw())
    assert ev.route == "archive"


def test_metrics_non_empty_after_run():
    DEFAULT_SOC_PIPELINE.run_one(_raw(event_id="E-metrics"))
    snapshot = DEFAULT_SOC_PIPELINE.metrics()
    assert snapshot, "metrics should contain stage entries after run"
    for stage, slot in snapshot.items():
        assert "count" in slot and "total_latency_s" in slot


def test_evidence_first_gate_downgrade():
    d = _decision(decision="auto_remediate", requires_human=True)
    gated = evidence_first_gate(d)
    assert gated.decision == "alert_human"
    assert "blast-radius" in gated.rationale


def test_evidence_first_gate_no_downgrade_when_autonomous():
    d = _decision(decision="auto_remediate", requires_human=False)
    gated = evidence_first_gate(d)
    assert gated.decision == "auto_remediate"


def test_triage_decision_rejects_unknown_decision():
    with pytest.raises(ValueError):
        TriageDecision(
            decision="nuke",
            priority="P1",
            rationale="",
            evidence_refs=[],
            recommended_action="",
            requires_human=False,
        )


def test_triage_decision_rejects_unknown_priority():
    with pytest.raises(ValueError):
        _decision(priority="P99")


def test_malformed_raw_event_rejected():
    pipe = SOCPipeline([IngestStage()])
    with pytest.raises(ValueError):
        pipe.run_one("not a dict")  # type: ignore[arg-type]


def test_missing_required_key_rejected():
    pipe = SOCPipeline([IngestStage()])
    with pytest.raises(ValueError):
        pipe.run_one({"event_id": "E-1"})


def test_unicode_event_handled():
    ev = DEFAULT_SOC_PIPELINE.run_one(_raw(event_id="E-α", source="测试"))
    assert ev.event_id == "E-α"


def test_run_batch_preserves_length():
    batch = [_raw(event_id=f"E-{i}") for i in range(5)]
    out = DEFAULT_SOC_PIPELINE.run_batch(batch)
    assert len(out) == 5
    assert [e.event_id for e in out] == [b["event_id"] for b in batch]


def test_determinism_with_stub_enricher():
    calls = {"n": 0}

    def _count(event):
        calls["n"] += 1
        return {"call_seq": calls["n"]}

    pipe = SOCPipeline([EnrichStage(enricher_fns=[_count])])
    e1 = pipe.run_one(_raw())
    e2 = pipe.run_one(_raw())
    assert e1.enrichments["call_seq"] == 1
    assert e2.enrichments["call_seq"] == 2


def test_default_route_table_keys():
    table = default_route_table()
    assert set(table) == {
        "auto_remediate",
        "alert_human",
        "escalate",
        "suppress",
        "defer",
    }


def test_finding_dataclass_roundtrip():
    f = Finding(
        finding_id="F1",
        entity="server-01",
        entity_type="host",
        issue_type="brute_force",
        severity="high",
        evidence=[{"log": "x"}],
        source="auth_log",
        environment="prod",
        discovered_at="2026-04-20T10:00Z",
    )
    assert f.entity == "server-01"
    assert f.evidence[0]["log"] == "x"


def test_pipeline_stage_is_abstract():
    with pytest.raises(TypeError):
        PipelineStage()  # type: ignore[abstract]
