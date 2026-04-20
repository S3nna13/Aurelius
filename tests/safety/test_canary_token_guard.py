"""Unit tests for ``src.safety.canary_token_guard``."""

from __future__ import annotations

import re

import pytest

from src.safety.canary_token_guard import CanaryDetection, CanaryTokenGuard


# --------------------------------------------------------------------- generate


def test_generate_uniqueness_across_many_calls() -> None:
    g = CanaryTokenGuard()
    tokens = {g.generate() for _ in range(2048)}
    # 16 bytes = 128 bits of entropy; 2048 draws should be fully unique.
    assert len(tokens) == 2048


def test_generate_format_matches_spec() -> None:
    g = CanaryTokenGuard()
    tok = g.generate(prefix="CANARY", entropy_bytes=16)
    assert re.fullmatch(r"CANARY-[0-9a-f]{32}-END", tok) is not None


def test_generate_custom_prefix_and_entropy() -> None:
    g = CanaryTokenGuard()
    tok = g.generate(prefix="TRIP_WIRE", entropy_bytes=24)
    assert re.fullmatch(r"TRIP_WIRE-[0-9a-f]{48}-END", tok) is not None


def test_generate_rejects_bad_prefix() -> None:
    g = CanaryTokenGuard()
    with pytest.raises(ValueError):
        g.generate(prefix="")
    with pytest.raises(ValueError):
        g.generate(prefix="has space")
    with pytest.raises(ValueError):
        g.generate(prefix="bad-dash")  # dash not in allowed charset


def test_generate_rejects_low_entropy() -> None:
    g = CanaryTokenGuard()
    with pytest.raises(ValueError):
        g.generate(entropy_bytes=4)


def test_generate_non_string_prefix_rejected() -> None:
    g = CanaryTokenGuard()
    with pytest.raises(TypeError):
        g.generate(prefix=123)  # type: ignore[arg-type]


# ------------------------------------------------------------ system-prompt inject


def test_inject_into_empty_system_prompt() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    out = g.inject_into_system_prompt("", tok)
    assert out != ""
    assert tok in out


def test_inject_into_populated_system_prompt_preserves_prefix() -> None:
    g = CanaryTokenGuard()
    base = "You are a helpful assistant."
    tok = g.generate()
    out = g.inject_into_system_prompt(base, tok)
    assert out.startswith(base)
    assert tok in out
    assert out.count(tok) == 1


def test_inject_non_string_inputs_rejected() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    with pytest.raises(TypeError):
        g.inject_into_system_prompt(None, tok)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        g.inject_into_system_prompt("sys", 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        g.inject_into_system_prompt("sys", "")


# ----------------------------------------------------------------------- wrap


def test_wrap_tool_output_surrounds_with_sentinels() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    wrapped = g.wrap_tool_output("payload from tool", tok)
    assert wrapped.count(tok) == 2  # once in open, once in close
    assert "payload from tool" in wrapped


def test_wrap_tool_output_multiple_times_stacks() -> None:
    g = CanaryTokenGuard()
    t1 = g.generate()
    t2 = g.generate()
    inner = g.wrap_tool_output("x", t1)
    outer = g.wrap_tool_output(inner, t2)
    assert outer.count(t1) == 2
    assert outer.count(t2) == 2


# ----------------------------------------------------------------------- scan


def test_scan_no_leak() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    det = g.scan("A perfectly normal assistant reply.", {tok})
    assert isinstance(det, CanaryDetection)
    assert det.leaked is False
    assert det.leaked_tokens == []
    assert det.first_offset == -1
    assert det.context == ""


def test_scan_single_leak_captures_context_and_offset() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False)
    tok = g.generate()
    body = "lorem ipsum " * 3 + tok + " trailing text"
    active = {tok}
    det = g.scan(body, active)
    assert det.leaked is True
    assert det.leaked_tokens == [tok]
    assert det.first_offset == body.index(tok)
    assert tok in det.context
    # rotate_on_leak=False means token is still active.
    assert tok in active


def test_scan_multiple_tokens_all_reported_sorted() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False)
    t1 = g.generate()
    t2 = g.generate()
    body = f"junk {t2} middle {t1} end"
    det = g.scan(body, {t1, t2})
    assert det.leaked is True
    assert det.leaked_tokens == sorted([t1, t2])
    # first_offset should be position of t2 (it appears first)
    assert det.first_offset == body.index(t2)


def test_scan_partial_substring_is_not_a_leak() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()  # e.g. CANARY-<32hex>-END
    # Emit a strict prefix of the token — an adversarial "near miss".
    partial = tok[: len(tok) - 5]
    assert partial != tok
    det = g.scan(f"model said {partial}", {tok})
    assert det.leaked is False


def test_scan_unicode_output() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False)
    tok = g.generate()
    body = "こんにちは 🌸 " + tok + " 🚀 مرحبا"
    det = g.scan(body, {tok})
    assert det.leaked is True
    assert tok in det.context


def test_scan_empty_output() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    det = g.scan("", {tok})
    assert det.leaked is False


def test_scan_large_output_over_100kb() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False)
    tok = g.generate()
    filler = "a" * 120_000
    body = filler + tok + filler
    det = g.scan(body, {tok})
    assert det.leaked is True
    assert det.first_offset == len(filler)


def test_scan_rejects_non_string_output() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    with pytest.raises(TypeError):
        g.scan(b"bytes not str", {tok})  # type: ignore[arg-type]


def test_scan_rejects_non_set_active_tokens() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    with pytest.raises(TypeError):
        g.scan("hi", [tok])  # type: ignore[arg-type]


def test_scan_rejects_non_string_token_element() -> None:
    g = CanaryTokenGuard()
    with pytest.raises(TypeError):
        g.scan("hi", {123})  # type: ignore[arg-type]


def test_scan_rejects_empty_string_token() -> None:
    g = CanaryTokenGuard()
    with pytest.raises(ValueError):
        g.scan("hi", {""})


# ---------------------------------------------------------------- rotation flag


def test_rotate_on_leak_removes_token_from_active_set() -> None:
    g = CanaryTokenGuard(rotate_on_leak=True)
    tok = g.generate()
    active = {tok}
    det = g.scan(f"oops I said {tok}", active)
    assert det.leaked is True
    assert tok not in active


def test_rotate_on_leak_false_preserves_token() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False)
    tok = g.generate()
    active = {tok}
    g.scan(f"oops {tok}", active)
    assert tok in active


# ------------------------------------------------------------------- history


def test_history_bounded_by_size() -> None:
    g = CanaryTokenGuard(history_size=4)
    tok = g.generate()
    for _ in range(10):
        g.scan("clean output", {tok})
    assert len(g.history) == 4


def test_history_records_both_leak_and_clean() -> None:
    g = CanaryTokenGuard(rotate_on_leak=False, history_size=8)
    tok = g.generate()
    g.scan("clean", {tok})
    g.scan(f"leak {tok}", {tok})
    g.scan("clean again", {tok})
    hist = g.history
    assert len(hist) == 3
    assert [d.leaked for d in hist] == [False, True, False]


def test_clear_history() -> None:
    g = CanaryTokenGuard()
    tok = g.generate()
    g.scan("x", {tok})
    assert len(g.history) == 1
    g.clear_history()
    assert len(g.history) == 0


# ------------------------------------------------------------------- determinism
# NOTE: ``secrets`` deliberately draws from os.urandom and cannot be seeded;
# deterministic generation is infeasible by design (see PEP 506). We therefore
# assert determinism via token-equality: the same token string always scans the
# same way regardless of when it was produced.


def test_scan_is_deterministic_given_same_inputs() -> None:
    g1 = CanaryTokenGuard(rotate_on_leak=False)
    g2 = CanaryTokenGuard(rotate_on_leak=False)
    tok = "CANARY-deadbeefcafebabe0123456789abcdef01-END"
    body = f"preamble {tok} postamble"
    d1 = g1.scan(body, {tok})
    d2 = g2.scan(body, {tok})
    assert d1 == d2


# ------------------------------------------------------------- constructor guards


def test_constructor_rejects_bad_history_size() -> None:
    with pytest.raises(ValueError):
        CanaryTokenGuard(history_size=0)
    with pytest.raises(ValueError):
        CanaryTokenGuard(history_size=-3)


def test_constructor_rejects_non_bool_rotate() -> None:
    with pytest.raises(TypeError):
        CanaryTokenGuard(rotate_on_leak="yes")  # type: ignore[arg-type]
