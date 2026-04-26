"""Unit tests for src.serving.structured_output_decoder.

Covers all key public surfaces:
 - StructuredOutputDecoder.is_valid_prefix
 - StructuredOutputDecoder.is_complete
 - StructuredOutputDecoder.build_token_mask_from_schema
 - StructuredOutputDecoder.constrained_logits
 - TokenTrie
 - GrammarConstrainedDecoder
 - Adversarial / edge cases
"""

from __future__ import annotations

import pytest
import torch

from src.serving.structured_output_decoder import (
    STRUCTURED_OUTPUT_REGISTRY,
    GrammarConstrainedDecoder,
    JsonParseState,
    StructuredOutputDecoder,
    TokenTrie,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
EOS_ID = 0


# Tiny vocabulary: indices 0..255, token strings are single ASCII characters
# plus a handful of JSON structural tokens placed at specific indices.
def _make_vocab() -> list[str]:
    vocab: list[str] = []
    # index 0 → EOS (empty string sentinel)
    vocab.append("")
    # index 1..127 → ASCII printable characters
    for i in range(1, 128):
        vocab.append(chr(i))
    # pad to 256 with two-char strings
    for i in range(128, 256):
        vocab.append(f"t{i}")
    return vocab


VOCAB = _make_vocab()


# Token ID helpers (index into VOCAB list)
def _tok(ch: str) -> int:
    return VOCAB.index(ch)


@pytest.fixture()
def decoder() -> StructuredOutputDecoder:
    return StructuredOutputDecoder(vocab_size=VOCAB_SIZE, eos_token_id=EOS_ID)


# ---------------------------------------------------------------------------
# 1. is_valid_prefix — string schema
# ---------------------------------------------------------------------------


def test_is_valid_prefix_string_empty(decoder):
    """Empty partial is always a valid prefix for any schema."""
    assert decoder.is_valid_prefix({"type": "string"}, "") is True


def test_is_valid_prefix_string_open_quote(decoder):
    """An opening quote is a valid start of a JSON string."""
    assert decoder.is_valid_prefix({"type": "string"}, '"') is True


def test_is_valid_prefix_string_partial(decoder):
    """Partial string body (open, no close) is a valid prefix."""
    assert decoder.is_valid_prefix({"type": "string"}, '"hello') is True


def test_is_valid_prefix_string_complete(decoder):
    """Fully closed string is a valid prefix (and a complete value)."""
    assert decoder.is_valid_prefix({"type": "string"}, '"hello"') is True


def test_is_valid_prefix_string_wrong_type(decoder):
    """A bare number is not a valid prefix for a string schema."""
    assert decoder.is_valid_prefix({"type": "string"}, "42") is False


# ---------------------------------------------------------------------------
# 2. is_valid_prefix — object schema with required fields
# ---------------------------------------------------------------------------


def test_is_valid_prefix_object_opening_brace(decoder):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert decoder.is_valid_prefix(schema, "{") is True


def test_is_valid_prefix_object_partial_key(decoder):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert decoder.is_valid_prefix(schema, '{"name') is True


def test_is_valid_prefix_object_bad_start(decoder):
    schema = {"type": "object"}
    assert decoder.is_valid_prefix(schema, "[") is False


# ---------------------------------------------------------------------------
# 3. is_valid_prefix — array schema
# ---------------------------------------------------------------------------


def test_is_valid_prefix_array_opening_bracket(decoder):
    schema = {"type": "array", "items": {"type": "number"}}
    assert decoder.is_valid_prefix(schema, "[") is True


def test_is_valid_prefix_array_partial(decoder):
    schema = {"type": "array", "items": {"type": "number"}}
    assert decoder.is_valid_prefix(schema, "[1, 2") is True


def test_is_valid_prefix_array_bad_start(decoder):
    schema = {"type": "array"}
    assert decoder.is_valid_prefix(schema, "{") is False


# ---------------------------------------------------------------------------
# 4. is_valid_prefix — enum schema
# ---------------------------------------------------------------------------


def test_is_valid_prefix_enum_valid_prefix(decoder):
    schema = {"enum": ["cat", "dog", "bird"]}
    # "cat" serialises to '"cat"' → '"ca' is a valid prefix
    assert decoder.is_valid_prefix(schema, '"ca') is True


def test_is_valid_prefix_enum_invalid_value(decoder):
    schema = {"enum": ["cat", "dog", "bird"]}
    assert decoder.is_valid_prefix(schema, '"fish"') is False


def test_is_valid_prefix_enum_exact_match(decoder):
    schema = {"enum": ["yes", "no"]}
    assert decoder.is_valid_prefix(schema, '"yes"') is True


# ---------------------------------------------------------------------------
# 5. is_complete — valid complete JSON
# ---------------------------------------------------------------------------


def test_is_complete_string(decoder):
    assert decoder.is_complete({"type": "string"}, '"hello"') is True


def test_is_complete_number(decoder):
    assert decoder.is_complete({"type": "number"}, "3.14") is True


def test_is_complete_boolean_true(decoder):
    assert decoder.is_complete({"type": "boolean"}, "true") is True


def test_is_complete_null(decoder):
    assert decoder.is_complete({"type": "null"}, "null") is True


def test_is_complete_object(decoder):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    assert decoder.is_complete(schema, '{"x": 1}') is True


def test_is_complete_object_missing_required(decoder):
    schema = {"type": "object", "required": ["x"]}
    assert decoder.is_complete(schema, "{}") is False


def test_is_complete_enum(decoder):
    schema = {"enum": [1, 2, 3]}
    assert decoder.is_complete(schema, "2") is True
    assert decoder.is_complete(schema, "5") is False


# ---------------------------------------------------------------------------
# 6. is_complete — invalid / partial JSON
# ---------------------------------------------------------------------------


def test_is_complete_rejects_partial(decoder):
    assert decoder.is_complete({"type": "string"}, '"hello') is False


def test_is_complete_rejects_wrong_type(decoder):
    assert decoder.is_complete({"type": "integer"}, '"hello"') is False


def test_is_complete_rejects_malformed_json(decoder):
    assert decoder.is_complete({"type": "object"}, "{bad}") is False


# ---------------------------------------------------------------------------
# 7. build_token_mask_from_schema — shape and dtype
# ---------------------------------------------------------------------------


def test_build_token_mask_shape_and_dtype(decoder):
    schema = {"type": "string"}
    mask = decoder.build_token_mask_from_schema(schema, "", VOCAB)
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (VOCAB_SIZE,)


def test_build_token_mask_has_allowed_tokens(decoder):
    schema = {"type": "string"}
    mask = decoder.build_token_mask_from_schema(schema, "", VOCAB)
    # At minimum the opening quote should be allowed.
    assert mask.any().item() is True


# ---------------------------------------------------------------------------
# 8. constrained_logits — disallowed tokens → -inf, allowed unchanged
# ---------------------------------------------------------------------------


def test_constrained_logits_disallowed_are_neg_inf(decoder):
    schema = {"type": "boolean"}
    logits = torch.zeros(1, VOCAB_SIZE)
    out = decoder.constrained_logits(logits, schema, "", VOCAB)
    mask = decoder.build_token_mask_from_schema(schema, "", VOCAB)
    # Every forbidden position must be exactly -inf.
    assert torch.all(out[0, ~mask] == float("-inf")).item()


def test_constrained_logits_allowed_unchanged(decoder):
    schema = {"type": "boolean"}
    logits = torch.ones(1, VOCAB_SIZE) * 5.0
    out = decoder.constrained_logits(logits, schema, "", VOCAB)
    mask = decoder.build_token_mask_from_schema(schema, "", VOCAB)
    if mask.any():
        assert torch.all(out[0, mask] == 5.0).item()


# ---------------------------------------------------------------------------
# 9. constrained_logits — never all-inf
# ---------------------------------------------------------------------------


def test_constrained_logits_never_all_inf(decoder):
    """Even with an impossible partial, EOS fallback prevents all-inf rows."""
    schema = {"type": "boolean"}
    logits = torch.zeros(1, VOCAB_SIZE)
    # Force a state where nothing should match by using a bogus partial.
    out = decoder.constrained_logits(logits, schema, "XYZABC!@#", VOCAB)
    assert not torch.all(out == float("-inf")).item()


# ---------------------------------------------------------------------------
# 10. TokenTrie — insert and prefix_match
# ---------------------------------------------------------------------------


def test_token_trie_exact_match():
    vocab = ["cat", "car", "card", "dog"]
    trie = TokenTrie(vocab)
    ids = trie.prefix_match("car")
    # Should find "car" (index 1) and "card" (index 2)
    assert 1 in ids
    assert 2 in ids
    assert 0 not in ids  # "cat" shares "ca" but not "car"


def test_token_trie_no_match():
    vocab = ["cat", "dog"]
    trie = TokenTrie(vocab)
    assert trie.prefix_match("xyz") == []


def test_token_trie_empty_prefix():
    vocab = ["a", "b", "c"]
    trie = TokenTrie(vocab)
    # Empty prefix should match all tokens
    ids = trie.prefix_match("")
    assert sorted(ids) == [0, 1, 2]


def test_token_trie_has_prefix():
    vocab = ["hello", "world"]
    trie = TokenTrie(vocab)
    assert trie.has_prefix("hel") is True
    assert trie.has_prefix("xyz") is False


# ---------------------------------------------------------------------------
# 11. GrammarConstrainedDecoder — basic grammar enforcement
# ---------------------------------------------------------------------------


def test_grammar_decoder_basic():
    vocab = ["", "a", "b", "c", "{", "}"]
    grammar = {
        "start": ["{"],
        "in_obj": ["a", "b", "}"],
    }
    decoder = GrammarConstrainedDecoder(
        vocab_size=len(vocab),
        eos_token_id=0,
        grammar_states=grammar,
        terminal_states={"in_obj"},
        initial_state="start",
    )
    mask = decoder.build_token_mask(vocab, state="start")
    assert mask.shape == (len(vocab),)
    # Only "{" (index 4) and not EOS should be set in "start"
    assert mask[4].item() is True  # "{"
    assert mask[1].item() is False  # "a" not allowed in start


def test_grammar_decoder_terminal_state_allows_eos():
    vocab = ["", "a", "b", "}"]
    grammar = {
        "start": ["a"],
        "done": ["}"],
    }
    decoder = GrammarConstrainedDecoder(
        vocab_size=len(vocab),
        eos_token_id=0,
        grammar_states=grammar,
        terminal_states={"done"},
        initial_state="start",
    )
    mask = decoder.build_token_mask(vocab, state="done")
    assert mask[0].item() is True  # EOS allowed in terminal state


def test_grammar_decoder_constrained_logits():
    vocab = ["", "a", "b"]
    grammar = {"start": ["a"]}
    decoder = GrammarConstrainedDecoder(
        vocab_size=len(vocab),
        eos_token_id=0,
        grammar_states=grammar,
        initial_state="start",
    )
    logits = torch.zeros(len(vocab))
    out = decoder.constrained_logits(logits, vocab)
    assert out[2] == float("-inf")  # "b" not allowed
    assert out[1] != float("-inf")  # "a" is allowed


# ---------------------------------------------------------------------------
# 12. Adversarial: malformed schema → ValueError (no silent fallback)
# ---------------------------------------------------------------------------


def test_malformed_schema_anyof_not_list(decoder):
    with pytest.raises(ValueError, match="list"):
        decoder.is_valid_prefix({"anyOf": "not-a-list"}, '"x"')


def test_malformed_schema_ref_raises(decoder):
    with pytest.raises(ValueError, match="\\$ref"):
        decoder.is_valid_prefix({"$ref": "#/definitions/Foo"}, "1")


def test_malformed_grammar_initial_state_missing():
    with pytest.raises(ValueError, match="initial_state"):
        GrammarConstrainedDecoder(
            vocab_size=4,
            eos_token_id=0,
            grammar_states={"other": ["a"]},
            initial_state="missing",
        )


# ---------------------------------------------------------------------------
# 13. Nested object schema
# ---------------------------------------------------------------------------


def test_is_complete_nested_object(decoder):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "required": ["a"],
    }
    assert decoder.is_complete(schema, '{"a": 42}') is True
    assert decoder.is_complete(schema, '{"a": "wrong"}') is False
    assert decoder.is_complete(schema, "{}") is False


def test_is_valid_prefix_nested_object(decoder):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
    }
    assert decoder.is_valid_prefix(schema, '{"a"') is True
    assert decoder.is_valid_prefix(schema, '{"a": 1') is True


# ---------------------------------------------------------------------------
# 14. anyOf schema
# ---------------------------------------------------------------------------


def test_is_valid_prefix_anyof(decoder):
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert decoder.is_valid_prefix(schema, '"hello') is True  # matches string branch
    assert decoder.is_valid_prefix(schema, "42") is True  # matches number branch
    assert decoder.is_valid_prefix(schema, "{") is False  # neither


def test_is_complete_anyof(decoder):
    schema = {"anyOf": [{"type": "string"}, {"type": "boolean"}]}
    assert decoder.is_complete(schema, '"yes"') is True
    assert decoder.is_complete(schema, "true") is True
    assert decoder.is_complete(schema, "42") is False


# ---------------------------------------------------------------------------
# 15. STRUCTURED_OUTPUT_REGISTRY — public registry check
# ---------------------------------------------------------------------------


def test_registry_keys():
    assert "json_schema" in STRUCTURED_OUTPUT_REGISTRY
    assert "grammar" in STRUCTURED_OUTPUT_REGISTRY
    assert STRUCTURED_OUTPUT_REGISTRY["json_schema"] is StructuredOutputDecoder
    assert STRUCTURED_OUTPUT_REGISTRY["grammar"] is GrammarConstrainedDecoder


# ---------------------------------------------------------------------------
# 16. JsonParseState enum re-exported on decoder class
# ---------------------------------------------------------------------------


def test_json_parse_state_accessible_on_decoder():
    assert StructuredOutputDecoder.JsonParseState is JsonParseState
    assert JsonParseState.START is not None
    assert JsonParseState.COMPLETE is not None
