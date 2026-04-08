import pytest
import torch
from src.inference.structured_output import (
    JSONConstraint, PartialJSONValidator, _is_json_prefix, JSONSchemaLogitProcessor
)

def test_is_json_prefix_empty():
    assert _is_json_prefix("") is True

def test_is_json_prefix_complete_object():
    assert _is_json_prefix('{"key": "value"}') is True

def test_is_json_prefix_partial_object():
    assert _is_json_prefix('{"key":') is True  # incomplete but valid prefix

def test_is_json_prefix_partial_string():
    assert _is_json_prefix('"hello') is True  # incomplete string

def test_is_json_prefix_invalid():
    assert _is_json_prefix('}{invalid') is False

def test_partial_validator_tracks_state():
    v = PartialJSONValidator()
    v.append('{"name": "test"}')
    assert v.is_complete()

def test_partial_validator_incomplete_not_complete():
    v = PartialJSONValidator()
    v.append('{"name":')
    assert not v.is_complete()

def test_partial_validator_prefix_valid():
    v = PartialJSONValidator()
    v.append('{"key":')
    assert v.is_valid_prefix(' "value"')

def test_json_logit_processor_masks_invalid():
    vocab = {0: "hello", 1: '"', 2: "}", 3: "xyz{{", 4: "true"}
    constraint = JSONConstraint(schema={"type": "object"})
    proc = JSONSchemaLogitProcessor(vocab, constraint)
    logits = torch.zeros(5)
    result = proc(logits, torch.tensor([]))
    # "xyz{{" is clearly invalid JSON continuation
    assert result[3] == float("-inf")

def test_json_logit_processor_allows_valid():
    vocab = {0: '"', 1: "}", 2: "true", 3: "xyz{{"}
    constraint = JSONConstraint(schema={"type": "object"})
    proc = JSONSchemaLogitProcessor(vocab, constraint)
    logits = torch.zeros(4)
    result = proc(logits, torch.tensor([]))
    # At least some tokens should be allowed
    assert (result > float("-inf")).any()
