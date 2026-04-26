"""Integration: CompactionTriggerManager wired into LONGCONTEXT_STRATEGY_REGISTRY."""

from __future__ import annotations

from src.longcontext import LONGCONTEXT_STRATEGY_REGISTRY
from src.longcontext.compaction_trigger import (
    DEFAULT_TIERS,
    CompactionTriggerManager,
)
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag


def test_compaction_trigger_registered():
    assert "compaction_trigger" in LONGCONTEXT_STRATEGY_REGISTRY
    assert LONGCONTEXT_STRATEGY_REGISTRY["compaction_trigger"] is CompactionTriggerManager


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.longcontext_compaction_trigger_enabled is False


def test_config_flag_settable():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="longcontext.compaction_trigger", enabled=True))
    cfg = AureliusConfig()
    assert cfg.longcontext_compaction_trigger_enabled is True


def test_default_tiers_match_mythos_thresholds():
    # Mythos System Card pp.189, 192 — fast at 50k, slow at 200k.
    names = {t.name: t.threshold_tokens for t in DEFAULT_TIERS}
    assert names.get("fast") == 50_000
    assert names.get("slow") == 200_000


def test_feature_flag_off_preserves_default_behavior():
    FEATURE_FLAG_REGISTRY._flags.pop("longcontext.compaction_trigger", None)
    cfg = AureliusConfig()
    assert cfg.longcontext_compaction_trigger_enabled is False
    assert "compaction_trigger" in LONGCONTEXT_STRATEGY_REGISTRY
