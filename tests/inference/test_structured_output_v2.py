"""Tests for src/inference/structured_output_v2.py

Tiny config: small vocabs, short JSON strings.
"""

from __future__ import annotations

import json

import torch

from src.inference.structured_output_v2 import (
    JSONModeDecoder,
    OutputSchema,
    StructuredOutputParser,
    build_json_grammar_logit_mask,
    extract_json_from_text,
    is_valid_json,
    validate_schema,
)

# ---------------------------------------------------------------------------
# OutputSchema defaults
# ---------------------------------------------------------------------------


def test_output_schema_defaults():
    schema = OutputSchema()
    assert schema.schema_type == "json"
    assert schema.required_keys == []
    assert schema.value_types == {}
    assert schema.max_tokens == 512


def test_output_schema_custom():
    schema = OutputSchema(
        schema_type="json",
        required_keys=["id", "label"],
        value_types={"id": "int", "label": "str"},
        max_tokens=256,
    )
    assert schema.required_keys == ["id", "label"]
    assert schema.value_types["id"] == "int"
    assert schema.max_tokens == 256


# ---------------------------------------------------------------------------
# is_valid_json
# ---------------------------------------------------------------------------


def test_is_valid_json_valid_object():
    assert is_valid_json('{"a": 1}') is True


def test_is_valid_json_valid_array():
    assert is_valid_json("[1, 2, 3]") is True


def test_is_valid_json_valid_primitives():
    assert is_valid_json('"hello"') is True
    assert is_valid_json("42") is True
    assert is_valid_json("true") is True
    assert is_valid_json("null") is True


def test_is_valid_json_invalid():
    assert is_valid_json("{bad json}") is False
    assert is_valid_json("") is False
    assert is_valid_json("hello world") is False


# ---------------------------------------------------------------------------
# extract_json_from_text
# ---------------------------------------------------------------------------


def test_extract_json_finds_object():
    text = 'Here is the result: {"name": "Alice", "age": 30} and trailing text.'
    result = extract_json_from_text(text)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["name"] == "Alice"
    assert parsed["age"] == 30


def test_extract_json_finds_array():
    text = "Output: [1, 2, 3] done."
    result = extract_json_from_text(text)
    assert result is not None
    assert json.loads(result) == [1, 2, 3]


def test_extract_json_returns_none_when_absent():
    assert extract_json_from_text("no json here") is None
    assert extract_json_from_text("") is None


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


def test_validate_schema_passes_all_keys():
    schema = OutputSchema(required_keys=["name", "score"], value_types={"score": "int"})
    valid, errors = validate_schema({"name": "Bob", "score": 99}, schema)
    assert valid is True
    assert errors == []


def test_validate_schema_missing_key_fails():
    schema = OutputSchema(required_keys=["name", "score"])
    valid, errors = validate_schema({"name": "Bob"}, schema)
    assert valid is False
    assert any("score" in e for e in errors)


def test_validate_schema_wrong_type_fails():
    schema = OutputSchema(value_types={"count": "int"})
    valid, errors = validate_schema({"count": "not-an-int"}, schema)
    assert valid is False
    assert any("count" in e for e in errors)


def test_validate_schema_bool_not_int():
    schema = OutputSchema(value_types={"flag": "int"})
    valid, errors = validate_schema({"flag": True}, schema)
    assert valid is False


def test_validate_schema_all_types_correct():
    schema = OutputSchema(
        value_types={
            "name": "str",
            "score": "float",
            "active": "bool",
            "tags": "list",
            "meta": "dict",
        }
    )
    data = {"name": "test", "score": 3.14, "active": True, "tags": ["a"], "meta": {"x": 1}}
    valid, errors = validate_schema(data, schema)
    assert valid is True, errors


# ---------------------------------------------------------------------------
# build_json_grammar_logit_mask
# ---------------------------------------------------------------------------


def test_mask_shape():
    vocab = ['"', "{", "[", "a", ",", "}", "]"]
    mask = build_json_grammar_logit_mask("", vocab)
    assert mask.shape == (len(vocab),)
    assert mask.dtype == torch.float32


def test_mask_values_zero_or_neg_inf():
    vocab = ['"', "{", "[", "a", ",", "}"]
    mask = build_json_grammar_logit_mask("", vocab)
    for val in mask.tolist():
        assert val == 0.0 or val == float("-inf")


def test_mask_expect_value_allows_open_brace():
    vocab = ["{", "a", " "]
    mask = build_json_grammar_logit_mask("", vocab)
    # '{' starts a value, should be allowed
    assert mask[0] == 0.0


# ---------------------------------------------------------------------------
# StructuredOutputParser
# ---------------------------------------------------------------------------


def test_parser_parse_returns_dict():
    schema = OutputSchema(required_keys=["x"])
    parser = StructuredOutputParser(schema)
    result = parser.parse('Result: {"x": 42}')
    assert result is not None
    assert result["x"] == 42


def test_parser_parse_returns_none_missing_key():
    schema = OutputSchema(required_keys=["x"])
    parser = StructuredOutputParser(schema)
    assert parser.parse('{"y": 1}') is None


def test_parser_parse_returns_none_invalid():
    schema = OutputSchema()
    parser = StructuredOutputParser(schema)
    assert parser.parse("not json at all") is None


def test_parser_repair_adds_closing_brace():
    parser = StructuredOutputParser(OutputSchema())
    repaired = parser.repair('{"key": "value"')
    assert is_valid_json(repaired)
    assert json.loads(repaired)["key"] == "value"


def test_parser_repair_removes_trailing_comma():
    parser = StructuredOutputParser(OutputSchema())
    repaired = parser.repair('{"a": 1,}')
    assert is_valid_json(repaired)


def test_parser_batch_parse_length():
    schema = OutputSchema(required_keys=["v"])
    parser = StructuredOutputParser(schema)
    texts = ['{"v": 1}', "not json", '{"v": 2}', "{}"]
    results = parser.batch_parse(texts)
    assert len(results) == 4
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert results[3] is None  # missing required "v"


# ---------------------------------------------------------------------------
# JSONModeDecoder
# ---------------------------------------------------------------------------


def test_json_mode_decoder_returns_string():
    vocab = ['"', "{", "}", ":", "k", "v", ",", " ", "<eos>"]
    eos_token_id = 8
    token_sequence = [1, 0, 4, 0, 3, 0, 5, 0, 2, eos_token_id]
    call_count = [0]

    def mock_model_fn(input_ids: torch.Tensor) -> torch.Tensor:
        idx = call_count[0]
        call_count[0] += 1
        logits = torch.full((len(vocab),), float("-inf"))
        t = token_sequence[idx] if idx < len(token_sequence) else eos_token_id
        logits[t] = 1.0
        return logits

    schema = OutputSchema()
    decoder = JSONModeDecoder(mock_model_fn, schema, vocab, eos_token_id=eos_token_id)
    result = decoder.decode(torch.tensor([0], dtype=torch.long), max_tokens=20)
    assert isinstance(result, str)
