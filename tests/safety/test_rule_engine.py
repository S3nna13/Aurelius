"""Tests for src.safety.rule_engine (Guard0)."""

from __future__ import annotations

import warnings

import pytest

from src.safety.rule_engine import (
    Rule,
    RuleDomain,
    RuleEngine,
    RuleEvaluationReport,
    RuleSeverity,
    RuleViolation,
    SEED_RULES,
    VALID_CHECK_TYPES,
)


def _mkinfo(**overrides):
    base = {
        "name": "test",
        "domain": "tool_safety",
        "severity": "high",
        "confidence": 0.8,
        "owasp_agentic": None,
        "standards": ["OWASP-ASI"],
    }
    base.update(overrides)
    return base


def _mkrule(rid="T-001", *, info=None, check=None):
    return Rule(
        id=rid,
        info=info or _mkinfo(),
        check=check
        or {
            "type": "prompt_contains",
            "pattern": r"rm -rf /",
            "message": "dangerous rm",
            "config": None,
        },
    )


def test_schema_rejects_bad_id():
    with pytest.raises(ValueError, match="alphanumeric"):
        _mkrule(rid="bad id!")
    with pytest.raises(ValueError):
        _mkrule(rid="")


def test_schema_rejects_missing_info_keys():
    bad_info = {"name": "x", "domain": "tool_safety", "severity": "high"}
    with pytest.raises(ValueError, match="missing required key"):
        _mkrule(info=bad_info)


def test_schema_rejects_unknown_check_type():
    with pytest.raises(ValueError, match="unknown check type"):
        _mkrule(
            check={
                "type": "telepathy",
                "pattern": "x",
                "message": "",
                "config": None,
            }
        )


def test_schema_rejects_unknown_domain_and_severity():
    with pytest.raises(ValueError, match="unknown domain"):
        _mkrule(info=_mkinfo(domain="bogus"))
    with pytest.raises(ValueError, match="unknown severity"):
        _mkrule(info=_mkinfo(severity="omega"))


def test_schema_confidence_bounds():
    with pytest.raises(ValueError, match="confidence"):
        _mkrule(info=_mkinfo(confidence=1.5))
    with pytest.raises(ValueError, match="confidence"):
        _mkrule(info=_mkinfo(confidence=-0.01))
    _mkrule(info=_mkinfo(confidence=0.0))
    _mkrule(info=_mkinfo(confidence=1.0))


def test_invalid_regex_raises_clear_error():
    with pytest.raises(ValueError, match="invalid regex"):
        _mkrule(
            check={
                "type": "prompt_contains",
                "pattern": "([unclosed",
                "message": "bad",
                "config": None,
            }
        )


def test_prompt_contains_positive_and_negative():
    rule = _mkrule(
        check={
            "type": "prompt_contains",
            "pattern": r"rm -rf",
            "message": "m",
            "config": None,
        }
    )
    eng = RuleEngine([rule])
    r_pos = eng.evaluate({"prompt": "please rm -rf /tmp now"})
    assert r_pos.failed == 1 and r_pos.passed == 0
    assert r_pos.violations[0].rule_id == "T-001"
    assert "rm -rf" in r_pos.violations[0].evidence_snippet

    r_neg = eng.evaluate({"prompt": "hello world"})
    assert r_neg.failed == 0 and r_neg.passed == 1


def test_prompt_missing_positive_and_negative():
    rule = _mkrule(
        rid="T-missing",
        check={
            "type": "prompt_missing",
            "pattern": r"human-approved",
            "message": "needs approval",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    r_pos = eng.evaluate({"prompt": "delete production database"})
    assert r_pos.failed == 1
    r_neg = eng.evaluate({"prompt": "delete production database [human-approved]"})
    assert r_neg.failed == 0 and r_neg.passed == 1


def test_code_matches_positive_and_negative():
    rule = _mkrule(
        rid="T-code",
        info=_mkinfo(domain="code_execution", severity="critical"),
        check={
            "type": "code_matches",
            "pattern": r"\bDYNAMIC_CALL\s*\(",
            "message": "dyn",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    r_pos = eng.evaluate({"code": "x = DYNAMIC_CALL('1+1')"})
    assert r_pos.failed == 1
    r_neg = eng.evaluate({"code": "x = 2"})
    assert r_neg.failed == 0 and r_neg.passed == 1


def test_config_matches_positive_and_negative():
    rule = _mkrule(
        rid="T-cfg",
        info=_mkinfo(domain="supply_chain", severity="medium"),
        check={
            "type": "config_matches",
            "pattern": r"pip install (?!.*==)",
            "message": "unpinned",
            "config": {"key": "setup.install_cmd"},
        },
    )
    eng = RuleEngine([rule])
    r_pos = eng.evaluate({"config": {"setup": {"install_cmd": "pip install requests"}}})
    assert r_pos.failed == 1
    r_neg = eng.evaluate({"config": {"setup": {"install_cmd": "pip install requests==2.31.0"}}})
    assert r_neg.failed == 0 and r_neg.passed == 1
    r_none = eng.evaluate({"config": {"other": 1}})
    assert r_none.failed == 0 and r_none.passed == 1


def test_agent_property_positive_and_negative():
    rule = _mkrule(
        rid="T-ident",
        info=_mkinfo(domain="identity_access", severity="high"),
        check={
            "type": "agent_property",
            "pattern": r"^admin$",
            "message": "admin role",
            "config": {"key": "role"},
        },
    )
    eng = RuleEngine([rule])
    r_pos = eng.evaluate({"agent_properties": {"role": "admin"}})
    assert r_pos.failed == 1
    r_neg = eng.evaluate({"agent_properties": {"role": "reader"}})
    assert r_neg.failed == 0 and r_neg.passed == 1


def test_no_check_rule_never_fires_but_is_counted():
    rule = _mkrule(
        rid="T-nocheck",
        info=_mkinfo(domain="reliability_bounds", severity="info"),
        check={
            "type": "no_check",
            "pattern": None,
            "message": "doc-only",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    r = eng.evaluate({})
    assert r.failed == 0 and r.passed == 1 and r.evaluated_rules == 1


def test_by_domain_and_by_severity_filters():
    eng = RuleEngine(list(SEED_RULES))
    code = eng.by_domain("code_execution")
    assert len(code) >= 2
    assert all(r.domain == RuleDomain.CODE_EXECUTION for r in code)

    crits = eng.by_severity(RuleSeverity.CRITICAL)
    assert len(crits) >= 2
    assert all(r.severity == RuleSeverity.CRITICAL for r in crits)


def test_evaluate_empty_rules_returns_no_violations():
    eng = RuleEngine([])
    r = eng.evaluate({"prompt": "anything"})
    assert r.violations == []
    assert r.evaluated_rules == 0 and r.passed == 0 and r.failed == 0


def test_evaluate_skips_rules_with_missing_context_key():
    rule = _mkrule(
        rid="T-skip",
        check={
            "type": "prompt_contains",
            "pattern": r"x",
            "message": "",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = eng.evaluate({})
    assert r.failed == 0 and r.passed == 0
    assert r.evaluated_rules == 1
    assert any("skipping" in str(ww.message) for ww in w)


def test_from_dicts_validates_and_constructs():
    rd = {
        "id": "AS-XYZ-1",
        "info": _mkinfo(domain="data_leakage", severity="critical"),
        "check": {
            "type": "prompt_contains",
            "pattern": "password",
            "message": "leaked pw",
            "config": None,
        },
    }
    eng = RuleEngine.from_dicts([rd])
    assert len(eng.rules) == 1
    with pytest.raises(ValueError, match="missing key"):
        RuleEngine.from_dicts([{"id": "X", "info": {}}])
    with pytest.raises(TypeError):
        RuleEngine.from_dicts("not-a-list")  # type: ignore[arg-type]


def test_seed_rules_loaded_and_span_multiple_domains():
    assert len(SEED_RULES) == 10
    domains = {r.domain for r in SEED_RULES}
    assert len(domains) >= 4
    ids = [r.id for r in SEED_RULES]
    assert len(set(ids)) == len(ids)
    for r in SEED_RULES:
        assert r.check["type"] in VALID_CHECK_TYPES


def test_seed_rules_detect_private_key_header():
    eng = RuleEngine(list(SEED_RULES))
    code = "secret = '-----BEGIN RSA PRIVATE KEY-----\\nAAAA\\n-----END'\n"
    r = eng.evaluate({"code": code})
    fired = {v.rule_id for v in r.violations}
    assert "AS-LEAK-001" in fired


def test_evaluate_is_deterministic():
    eng = RuleEngine(list(SEED_RULES))
    ctx = {
        "prompt": "ls; AKIAABCDEFGHIJKLMNOP",
        "code": "x = 1",
        "agent_properties": {"role": "admin", "self_modifying": "true"},
    }
    r1 = eng.evaluate(ctx)
    r2 = eng.evaluate(ctx)
    assert [v.rule_id for v in r1.violations] == [v.rule_id for v in r2.violations]
    assert r1.passed == r2.passed and r1.failed == r2.failed


def test_unicode_evidence_snippet():
    rule = _mkrule(
        rid="T-uni",
        check={
            "type": "prompt_contains",
            "pattern": r"secret",
            "message": "",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    r = eng.evaluate({"prompt": "\u524d\u7f6e secret \u65e5\u672c\u8a9e"})
    assert r.failed == 1
    snip = r.violations[0].evidence_snippet
    assert "secret" in snip
    snip.encode("utf-8")


def test_adversarial_quoted_pattern_content():
    rule = _mkrule(
        rid="T-adv",
        check={
            "type": "prompt_contains",
            "pattern": r"\$\(whoami\)",
            "message": "cmd sub",
            "config": None,
        },
    )
    eng = RuleEngine([rule])
    r = eng.evaluate({"prompt": "run $(whoami) and then exit"})
    assert r.failed == 1
    assert "$(whoami)" in r.violations[0].evidence_snippet


def test_report_totals_consistent_with_violations():
    eng = RuleEngine(list(SEED_RULES))
    r = eng.evaluate(
        {
            "prompt": "hello AKIAABCDEFGHIJKLMNOP human-approved",
            "code": "x=1",
            "config": {"setup": {"install_cmd": "pip install foo"}},
            "agent_properties": {"role": "reader", "self_modifying": "false"},
        }
    )
    assert len(r.violations) == r.failed
    assert r.passed + r.failed <= r.evaluated_rules
    assert r.evaluated_rules == len(SEED_RULES)


def test_report_invariant_violation_detected():
    with pytest.raises(ValueError, match="exceeds"):
        RuleEvaluationReport(violations=[], evaluated_rules=1, passed=1, failed=1)


def test_violation_dataclass_fields():
    rule = _mkrule()
    eng = RuleEngine([rule])
    r = eng.evaluate({"prompt": "rm -rf /tmp"})
    assert isinstance(r.violations[0], RuleViolation)
    v = r.violations[0]
    assert v.severity == RuleSeverity.HIGH
    assert v.domain == RuleDomain.TOOL_SAFETY
    assert 0.0 <= v.confidence <= 1.0
