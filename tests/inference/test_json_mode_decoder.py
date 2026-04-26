"""Unit tests for JSON-mode constrained decoder."""

from __future__ import annotations

import pytest
import torch

from src.inference.json_mode_decoder import (
    MAX_DEPTH,
    JSONDecoderState,
    JSONMaskBuilder,
    is_valid_json_prefix,
)

VOCAB = ["{", "}", "[", "]", ",", ":", '"', "a", "b", "1", "2", "3", " "]


def _idx(tok: str) -> int:
    return VOCAB.index(tok)


def _feed(builder: JSONMaskBuilder, tokens: list[str]) -> JSONDecoderState:
    state = builder.reset()
    for tok in tokens:
        state = builder.update(state, tok)
    return state


def test_initial_state_allows_only_open_containers() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = builder.reset()
    mask = builder.get_mask(state)
    assert mask[_idx("{")].item() is True
    assert mask[_idx("[")].item() is True
    assert mask[_idx("}")].item() is False
    assert mask[_idx("]")].item() is False
    assert mask[_idx(",")].item() is False
    assert mask[_idx(":")].item() is False
    assert mask[_idx("a")].item() is False


def test_after_brace_allows_quote_or_close() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = _feed(builder, ["{"])
    mask = builder.get_mask(state)
    assert mask[_idx('"')].item() is True
    assert mask[_idx("}")].item() is True
    assert mask[_idx("{")].item() is False
    assert mask[_idx(",")].item() is False


def test_expecting_value_after_colon() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = _feed(builder, ["{", '"', "a", '"', ":"])
    mask = builder.get_mask(state)
    # A value can start with "{", "[", quote or a digit.
    assert mask[_idx("{")].item() is True
    assert mask[_idx("[")].item() is True
    assert mask[_idx('"')].item() is True
    assert mask[_idx("1")].item() is True
    # Not a close.
    assert mask[_idx("}")].item() is False
    assert mask[_idx("]")].item() is False


def test_nested_object_step_by_step() -> None:
    builder = JSONMaskBuilder(VOCAB)
    seq = ["{", '"', "a", '"', ":", "{", "}", "}"]
    state = builder.reset()
    for tok in seq:
        state = builder.update(state, tok)
    assert state.done is True
    assert state.stack == []


def test_array_round_trip() -> None:
    builder = JSONMaskBuilder(VOCAB)
    seq = ["[", "1", ",", "2", ",", "3", "]"]
    state = builder.reset()
    for tok in seq:
        state = builder.update(state, tok)
    assert state.done is True


def test_string_content_allows_any_char_except_unescaped_quote() -> None:
    builder = JSONMaskBuilder(VOCAB)
    # Enter array then open string value.
    state = _feed(builder, ["[", '"'])
    mask = builder.get_mask(state)
    # Inside a string, letters and digits are allowed.
    assert mask[_idx("a")].item() is True
    assert mask[_idx("b")].item() is True
    assert mask[_idx("1")].item() is True
    # The quote token closes the string - admissible.
    assert mask[_idx('"')].item() is True
    # Structural tokens are also fine as string content.
    assert mask[_idx("{")].item() is True


def test_numbers_admissible_in_value_position() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = _feed(builder, ["["])
    mask = builder.get_mask(state)
    assert mask[_idx("1")].item() is True
    assert mask[_idx("2")].item() is True
    assert mask[_idx("3")].item() is True


def test_comma_between_array_elements() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = _feed(builder, ["[", "1"])
    mask = builder.get_mask(state)
    # After "1" we are in a number; "," terminates it and is valid.
    assert mask[_idx(",")].item() is True
    assert mask[_idx("]")].item() is True
    # A trailing comma right after "[" with no element is NOT valid:
    state2 = _feed(builder, ["["])
    mask2 = builder.get_mask(state2)
    assert mask2[_idx(",")].item() is False


def test_is_valid_json_prefix_basic() -> None:
    assert is_valid_json_prefix("{ ") is True
    assert is_valid_json_prefix("}{") is False
    assert is_valid_json_prefix("[1,2") is True
    assert is_valid_json_prefix("[,") is False


def test_mask_logits_sets_minus_inf() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = builder.reset()
    logits = torch.zeros(len(VOCAB))
    masked = builder.mask_logits(logits, state)
    assert masked[_idx("{")].item() == 0.0
    assert masked[_idx("[")].item() == 0.0
    assert masked[_idx("}")].item() == float("-inf")
    assert masked[_idx(",")].item() == float("-inf")


def test_determinism() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state1 = _feed(builder, ["{", '"', "a", '"', ":", "1"])
    state2 = _feed(builder, ["{", '"', "a", '"', ":", "1"])
    m1 = builder.get_mask(state1)
    m2 = builder.get_mask(state2)
    assert torch.equal(m1, m2)


def test_escape_handling_keeps_in_string() -> None:
    vocab = ["{", "}", '"', "\\", "a", ":", ","]
    builder = JSONMaskBuilder(vocab)
    # Open object, key start, feed "a\" which keeps us in string
    # because the backslash escapes the quote.
    state = builder.reset()
    state = builder.update(state, "{")
    state = builder.update(state, '"')
    assert state.in_string is True
    state = builder.update(state, "\\")
    assert state.in_string is True
    assert state.escape is True
    state = builder.update(state, '"')
    # Still in string after escaped quote.
    assert state.in_string is True
    assert state.escape is False


def test_depth_overflow_raises() -> None:
    builder = JSONMaskBuilder(["[", "]"])
    state = builder.reset()
    for _ in range(MAX_DEPTH):
        state = builder.update(state, "[")
    with pytest.raises(OverflowError):
        builder.update(state, "[")


def test_mask_logits_shape_validation() -> None:
    builder = JSONMaskBuilder(VOCAB)
    state = builder.reset()
    with pytest.raises(ValueError):
        builder.mask_logits(torch.zeros(len(VOCAB) + 1), state)
    with pytest.raises(ValueError):
        builder.mask_logits(torch.zeros(2, len(VOCAB)), state)
