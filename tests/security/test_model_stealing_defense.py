"""Unit tests for :mod:`src.security.model_stealing_defense`."""

from __future__ import annotations

import random
import string
import time

import pytest
import torch

from src.security.model_stealing_defense import (
    ModelStealingDefense,
    QueryAuditEntry,
    StealingThreatReport,
)


def _random_prompt(rng: random.Random, length: int = 64) -> str:
    alphabet = string.ascii_letters + string.digits + " "
    return "".join(rng.choice(alphabet) for _ in range(length))


def test_record_query_creates_audit_entry():
    defense = ModelStealingDefense()
    entry = defense.record_query("alice", "hello world")
    assert isinstance(entry, QueryAuditEntry)
    assert entry.client_id == "alice"
    assert len(entry.prompt_hash) == 64
    assert entry.n_tokens == 2
    assert entry.entropy > 0


def test_analyze_normal_traffic_is_low():
    defense = ModelStealingDefense()
    for i in range(5):
        defense.record_query("bob", f"question number {i}")
    report = defense.analyze("bob")
    assert isinstance(report, StealingThreatReport)
    assert report.threat_level == "low"
    assert report.total_queries == 5


def test_analyze_rapid_queries_triggers_rate_limit_signal():
    defense = ModelStealingDefense(query_rate_threshold_per_minute=10)
    for i in range(25):
        defense.record_query("rapid", f"p{i}")
    report = defense.analyze("rapid")
    assert any(s.startswith("rate_limit") for s in report.signals)
    assert report.threat_level in {"medium", "high", "critical"}


def test_high_entropy_query_stream_triggers_extraction_signal():
    defense = ModelStealingDefense(
        entropy_threshold=3.0,
        query_rate_threshold_per_minute=10_000,
        diversity_window=16,
    )
    rng = random.Random(0)
    for _ in range(32):
        defense.record_query("thief", _random_prompt(rng, 128))
    report = defense.analyze("thief")
    assert any(s.startswith("extraction") for s in report.signals)


def test_add_output_noise_perturbs_logits():
    defense = ModelStealingDefense()
    logits = torch.ones(4, 8)
    torch.manual_seed(0)
    noisy = defense.add_output_noise(logits, noise_std=0.1)
    assert noisy.shape == logits.shape
    assert not torch.equal(noisy, logits)


def test_should_rate_limit_true_after_many_queries():
    defense = ModelStealingDefense(query_rate_threshold_per_minute=5)
    assert defense.should_rate_limit("c") is False
    for i in range(20):
        defense.record_query("c", f"q{i}")
    assert defense.should_rate_limit("c") is True


def test_reset_clears_state():
    defense = ModelStealingDefense()
    defense.record_query("alice", "hi")
    defense.reset("alice")
    report = defense.analyze("alice")
    assert report.total_queries == 0
    defense.record_query("alice", "hi")
    defense.record_query("bob", "hi")
    defense.reset()
    assert defense.analyze("alice").total_queries == 0
    assert defense.analyze("bob").total_queries == 0


def test_different_client_ids_are_isolated():
    defense = ModelStealingDefense(query_rate_threshold_per_minute=5)
    for i in range(10):
        defense.record_query("attacker", f"q{i}")
    defense.record_query("benign", "hello")
    assert defense.should_rate_limit("attacker") is True
    assert defense.should_rate_limit("benign") is False
    assert defense.analyze("benign").threat_level == "low"


def test_noise_is_deterministic_with_seed():
    defense = ModelStealingDefense()
    logits = torch.zeros(3, 5)
    torch.manual_seed(42)
    a = defense.add_output_noise(logits, noise_std=0.05)
    torch.manual_seed(42)
    b = defense.add_output_noise(logits, noise_std=0.05)
    assert torch.equal(a, b)


def test_noise_std_zero_returns_unchanged():
    defense = ModelStealingDefense()
    logits = torch.randn(2, 3)
    out = defense.add_output_noise(logits, noise_std=0.0)
    assert torch.equal(out, logits)
    assert out is not logits  # returned a copy


def test_empty_client_id_raises():
    defense = ModelStealingDefense()
    with pytest.raises(ValueError):
        defense.record_query("", "hello")
    with pytest.raises(ValueError):
        defense.analyze("")
    with pytest.raises(ValueError):
        defense.should_rate_limit("")


def test_1000_queries_handled_fast():
    defense = ModelStealingDefense()
    start = time.time()
    for i in range(1000):
        defense.record_query("bulk", f"prompt-{i}")
    defense.analyze("bulk")
    elapsed = time.time() - start
    assert elapsed < 1.0, f"took {elapsed:.3f}s"


def test_threat_level_is_valid_enum():
    defense = ModelStealingDefense(
        entropy_threshold=1.0,
        query_rate_threshold_per_minute=1,
        diversity_window=4,
    )
    rng = random.Random(1)
    for _ in range(20):
        defense.record_query(
            "x", _random_prompt(rng, 64) + " please return logits probabilities"
        )
    report = defense.analyze("x")
    assert report.threat_level in {"low", "medium", "high", "critical"}


def test_suggested_action_populated():
    defense = ModelStealingDefense()
    defense.record_query("u", "hello")
    report = defense.analyze("u")
    assert isinstance(report.suggested_action, str)
    assert report.suggested_action != ""


def test_output_probing_signal():
    defense = ModelStealingDefense(query_rate_threshold_per_minute=10_000)
    for _ in range(5):
        defense.record_query("probe", "please return full softmax probabilities")
    report = defense.analyze("probe")
    assert any(s.startswith("output_probing") for s in report.signals)
