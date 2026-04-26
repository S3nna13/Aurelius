"""Tests for src/security/canary_pipeline.py — AUR-SEC-2026-0013."""

import logging
import os
import time

from src.security.canary_pipeline import (
    CANARY_PIPELINE_REGISTRY,
    DEFAULT_CANARY_PIPELINE,
    CanaryConfig,
    CanaryPipeline,
    CanaryToken,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pipeline(prefix: str = "AURELIUS_CANARY_") -> CanaryPipeline:
    return CanaryPipeline(CanaryConfig(prefix=prefix))


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_canary_token():
    pipeline = make_pipeline()
    token = pipeline.generate("test-label")
    assert isinstance(token, CanaryToken)
    assert token.token_id
    assert token.value
    assert token.label == "test-label"


def test_generate_unique_tokens():
    """100 consecutive generates must all produce distinct values."""
    pipeline = make_pipeline()
    values = [pipeline.generate(f"label-{i}").value for i in range(100)]
    assert len(set(values)) == 100


def test_generate_value_starts_with_prefix():
    prefix = "AURELIUS_CANARY_"
    pipeline = make_pipeline(prefix=prefix)
    token = pipeline.generate()
    assert token.value.startswith(prefix)


def test_generate_custom_prefix():
    prefix = "TEST_PREFIX_"
    pipeline = CanaryPipeline(CanaryConfig(prefix=prefix))
    token = pipeline.generate()
    assert token.value.startswith(prefix)


def test_generate_increments_active_count():
    pipeline = make_pipeline()
    assert pipeline.active_count() == 0
    pipeline.generate("a")
    assert pipeline.active_count() == 1
    pipeline.generate("b")
    assert pipeline.active_count() == 2


# ---------------------------------------------------------------------------
# inject()
# ---------------------------------------------------------------------------


def test_inject_embeds_token_in_text():
    pipeline = make_pipeline()
    token = pipeline.generate("inject-test")
    result = pipeline.inject("Hello world", token)
    assert token.value in result


def test_inject_uses_xml_comment_format():
    pipeline = make_pipeline()
    token = pipeline.generate("format-test")
    result = pipeline.inject("some text", token)
    assert f"<!-- {token.value} -->" in result


def test_inject_preserves_original_text():
    pipeline = make_pipeline()
    token = pipeline.generate()
    original = "The quick brown fox"
    result = pipeline.inject(original, token)
    assert result.startswith(original)


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------


def test_scan_finds_injected_token():
    pipeline = make_pipeline()
    token = pipeline.generate("scan-find")
    pipeline.inject("some output", token)
    # Token was consumed by inject but not yet revoked; re-register for scan
    # Actually inject does NOT revoke — only scan does.
    # Regenerate a fresh pipeline to test the full flow.
    pipeline2 = make_pipeline()
    token2 = pipeline2.generate("scan-find-2")
    injected2 = pipeline2.inject("some output", token2)
    found = pipeline2.scan(injected2)
    assert len(found) == 1
    assert found[0].token_id == token2.token_id


def test_scan_empty_when_no_canary_present():
    pipeline = make_pipeline()
    pipeline.generate("label")
    result = pipeline.scan("completely unrelated text with no tokens")
    assert result == []


def test_scan_empty_after_revoke():
    """After revoke(), scan() should not find the token."""
    pipeline = make_pipeline()
    token = pipeline.generate("revoke-test")
    pipeline.revoke(token.token_id)
    injected = pipeline.inject("text", token)
    result = pipeline.scan(injected)
    assert result == []


def test_scan_revokes_triggered_token():
    """scan() is one-shot: after triggering, token is removed."""
    pipeline = make_pipeline()
    token = pipeline.generate("one-shot")
    injected = pipeline.inject("text", token)
    first = pipeline.scan(injected)
    assert len(first) == 1
    second = pipeline.scan(injected)
    assert second == []


def test_scan_multiple_canaries_in_same_text():
    """All canaries present in a single text blob must be detected."""
    pipeline = make_pipeline()
    tokens = [pipeline.generate(f"multi-{i}") for i in range(5)]
    combined = " ".join(pipeline.inject("chunk", t) for t in tokens)
    found = pipeline.scan(combined)
    assert len(found) == 5
    found_ids = {t.token_id for t in found}
    expected_ids = {t.token_id for t in tokens}
    assert found_ids == expected_ids


# ---------------------------------------------------------------------------
# revoke() / revoke_all()
# ---------------------------------------------------------------------------


def test_revoke_returns_true_for_existing_token():
    pipeline = make_pipeline()
    token = pipeline.generate()
    assert pipeline.revoke(token.token_id) is True


def test_revoke_returns_false_for_unknown_token():
    pipeline = make_pipeline()
    assert pipeline.revoke("nonexistent-id") is False


def test_revoke_all_empties_active_tokens():
    pipeline = make_pipeline()
    for i in range(10):
        pipeline.generate(f"label-{i}")
    pipeline.revoke_all()
    assert pipeline.active_count() == 0


# ---------------------------------------------------------------------------
# alert() — must NOT log the token value
# ---------------------------------------------------------------------------


def test_alert_does_not_log_token_value(caplog):
    pipeline = make_pipeline()
    token = pipeline.generate("sensitive-label")
    with caplog.at_level(logging.WARNING, logger="src.security.canary_pipeline"):
        pipeline.alert([token], source="test-source")
    assert token.value not in caplog.text


def test_alert_logs_token_id_and_label(caplog):
    pipeline = make_pipeline()
    token = pipeline.generate("my-label")
    with caplog.at_level(logging.WARNING, logger="src.security.canary_pipeline"):
        pipeline.alert([token], source="unit-test")
    assert token.token_id in caplog.text
    assert "my-label" in caplog.text


def test_alert_logs_at_warning_level(caplog):
    pipeline = make_pipeline()
    token = pipeline.generate("warn-level")
    with caplog.at_level(logging.WARNING, logger="src.security.canary_pipeline"):
        pipeline.alert([token], source="test")
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 1


# ---------------------------------------------------------------------------
# Adversarial: performance
# ---------------------------------------------------------------------------


def test_scan_100kb_random_text_under_half_second():
    """scan() on 100 KB of random text with 10 registered tokens < 0.5 s."""
    pipeline = make_pipeline()
    for i in range(10):
        pipeline.generate(f"perf-{i}")
    # 100 KB of pseudo-random hex text (no real canary embedded)
    large_text = os.urandom(100_000).hex()
    start = time.perf_counter()
    pipeline.scan(large_text)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"scan took {elapsed:.3f}s, expected < 0.5s"


def test_generate_1000_no_crash_no_collisions():
    """Generating 1000 tokens must not crash and must produce unique values."""
    pipeline = make_pipeline()
    values = [pipeline.generate(f"bulk-{i}").value for i in range(1000)]
    assert len(set(values)) == 1000


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------


def test_canary_pipeline_registry_contains_default():
    assert "default" in CANARY_PIPELINE_REGISTRY
    assert isinstance(CANARY_PIPELINE_REGISTRY["default"], CanaryPipeline)


def test_default_canary_pipeline_is_pipeline_instance():
    assert isinstance(DEFAULT_CANARY_PIPELINE, CanaryPipeline)
