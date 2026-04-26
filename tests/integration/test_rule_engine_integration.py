"""Integration tests for the Guard0 rule engine.

Checks that:
  * RuleEngine is registered in SAFETY_FILTER_REGISTRY.
  * The ``safety_rule_engine_enabled`` config flag exists and defaults to False.
  * SEED_RULES round-trip through from_dicts -> evaluate.
"""

from __future__ import annotations

from src.model.config import AureliusConfig
from src.safety import (
    RULE_ENGINE_SEED_RULES,
    SAFETY_FILTER_REGISTRY,
    RuleEngine,
)


def test_rule_engine_registered_in_safety_filter_registry():
    assert "rule_engine" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["rule_engine"] is RuleEngine


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert hasattr(cfg, "safety_rule_engine_enabled")
    assert cfg.safety_rule_engine_enabled is False


def test_seed_rules_instantiated_and_evaluable():
    eng = RuleEngine(list(RULE_ENGINE_SEED_RULES))
    ctx = {
        "prompt": "list files; cat /etc/passwd",
        "code": "x = 1",
        "config": {"setup": {"install_cmd": "pip install pkg==1.0"}},
        "agent_properties": {"role": "reader", "self_modifying": "false"},
    }
    report = eng.evaluate(ctx)
    assert report.evaluated_rules == len(RULE_ENGINE_SEED_RULES)
    # Shell-metacharacter rule should fire on the prompt.
    fired = {v.rule_id for v in report.violations}
    assert "AS-TOOL-001" in fired
