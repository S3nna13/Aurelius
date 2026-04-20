"""Tests for synthetic_jailbreak_generator."""

from __future__ import annotations

import pytest

from src.eval.synthetic_jailbreak_generator import (
    SyntheticJailbreakGenerator,
    score_probe_naive_refusal,
)


def test_generate_deterministic_with_seed():
    g = SyntheticJailbreakGenerator()
    a = g.generate(20, seed=123)
    b = g.generate(20, seed=123)
    assert [p.text for p in a] == [p.text for p in b]
    assert all(isinstance(p.template_id, int) for p in a)


def test_generate_zero():
    assert SyntheticJailbreakGenerator().generate(0, seed=0) == []


def test_generate_payload_substitution():
    g = SyntheticJailbreakGenerator(default_payload="X")
    out = g.generate(50, seed=0, payload="Y")
    assert any("Y" in p.text for p in out)


def test_generate_rejects_negative_n():
    with pytest.raises(ValueError):
        SyntheticJailbreakGenerator().generate(-1)


def test_generate_rejects_huge_n():
    with pytest.raises(ValueError):
        SyntheticJailbreakGenerator().generate(20_000)


def test_generate_type_errors():
    with pytest.raises(TypeError):
        SyntheticJailbreakGenerator().generate("3")  # type: ignore[arg-type]


def test_default_payload_type_guard():
    with pytest.raises(TypeError):
        SyntheticJailbreakGenerator(default_payload=123)  # type: ignore[arg-type]


def test_score_refusal_nonempty():
    s = score_probe_naive_refusal("I cannot help with that policy violation.")
    assert 0.0 <= s <= 1.0


def test_score_refusal_empty_string():
    assert score_probe_naive_refusal("") == 0.0


def test_score_refusal_type_error():
    with pytest.raises(TypeError):
        score_probe_naive_refusal(None)  # type: ignore[arg-type]


def test_sanitize_strips_control_chars():
    g = SyntheticJailbreakGenerator()
    # n=1 should not crash on weird payload
    out = g.generate(1, seed=0, payload="a\x00b\x1fc")
    assert "\x00" not in out[0].text


def test_adversarial_jailbreak_phrase_in_templates():
    g = SyntheticJailbreakGenerator()
    batch = g.generate(200, seed=99)
    texts = " ".join(p.text for p in batch).lower()
    assert "ignore" in texts or "dan" in texts or "override" in texts
