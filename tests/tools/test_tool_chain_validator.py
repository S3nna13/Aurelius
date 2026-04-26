"""Tests for tool_chain_validator."""
from __future__ import annotations

import pytest

from src.tools.tool_chain_validator import (
    ToolChainError,
    ToolChainValidator,
    TOOL_CHAIN_VALIDATOR_REGISTRY,
)


class TestToolChainValidatorValidateChain:
    def test_valid_chain(self):
        v = ToolChainValidator()
        chain = [
            {"name": "search", "args": {"query": "hello"}},
            {"name": "summarize", "args": {"text": "${search.output}"}},
        ]
        assert v.validate_chain(chain) == []

    def test_valid_chain_at_max_length(self):
        v = ToolChainValidator()
        chain = [{"name": f"tool_{i}", "args": {}} for i in range(32)]
        assert v.validate_chain(chain) == []

    def test_missing_name_key(self):
        v = ToolChainValidator()
        chain = [{"args": {}}]
        errors = v.validate_chain(chain)
        assert any("missing required key 'name'" in e for e in errors)

    def test_missing_args_key(self):
        v = ToolChainValidator()
        chain = [{"name": "tool"}]
        errors = v.validate_chain(chain)
        assert any("missing required key 'args'" in e for e in errors)

    def test_missing_both_keys(self):
        v = ToolChainValidator()
        chain = [{}]
        errors = v.validate_chain(chain)
        assert any("missing required key 'name'" in e for e in errors)
        assert any("missing required key 'args'" in e for e in errors)

    def test_invalid_name_empty(self):
        v = ToolChainValidator()
        chain = [{"name": "", "args": {}}]
        errors = v.validate_chain(chain)
        assert any("non-empty" in e for e in errors)

    def test_invalid_name_too_long(self):
        v = ToolChainValidator()
        chain = [{"name": "a" * 65, "args": {}}]
        errors = v.validate_chain(chain)
        assert any("exceeds 64" in e for e in errors)

    def test_invalid_name_bad_chars(self):
        v = ToolChainValidator()
        chain = [{"name": "bad name!", "args": {}}]
        errors = v.validate_chain(chain)
        assert any("invalid characters" in e for e in errors)

    def test_invalid_name_non_string(self):
        v = ToolChainValidator()
        chain = [{"name": 123, "args": {}}]
        errors = v.validate_chain(chain)
        assert any("must be a string" in e for e in errors)

    def test_circular_dependency(self):
        v = ToolChainValidator()
        chain = [
            {"name": "a", "args": {"x": "$b"}},
            {"name": "b", "args": {"x": "$a"}},
        ]
        errors = v.validate_chain(chain)
        assert any("circular" in e.lower() for e in errors)

    def test_max_length_exceeded(self):
        v = ToolChainValidator()
        chain = [{"name": f"tool_{i}", "args": {}} for i in range(33)]
        errors = v.validate_chain(chain)
        assert any("exceeds maximum length" in e.lower() for e in errors)

    def test_duplicates_not_allowed_by_default(self):
        v = ToolChainValidator()
        chain = [
            {"name": "dup", "args": {}},
            {"name": "dup", "args": {}},
        ]
        errors = v.validate_chain(chain)
        assert any("duplicate" in e.lower() for e in errors)

    def test_duplicates_allowed_with_flag(self):
        v = ToolChainValidator()
        chain = [
            {"name": "dup", "args": {}},
            {"name": "dup", "args": {}},
        ]
        assert v.validate_chain(chain, allow_duplicates=True) == []


class TestToolChainValidatorCheckAcyclic:
    def test_acyclic_chain(self):
        v = ToolChainValidator()
        chain = [
            {"name": "a", "args": {}},
            {"name": "b", "args": {"x": "$a"}},
        ]
        assert v.check_acyclic(chain) is True

    def test_cyclic_chain(self):
        v = ToolChainValidator()
        chain = [
            {"name": "a", "args": {"x": "$b"}},
            {"name": "b", "args": {"x": "$a"}},
        ]
        assert v.check_acyclic(chain) is False

    def test_self_reference_is_cycle(self):
        v = ToolChainValidator()
        chain = [{"name": "a", "args": {"x": "$a"}}]
        assert v.check_acyclic(chain) is False

    def test_chain_with_no_refs(self):
        v = ToolChainValidator()
        chain = [
            {"name": "a", "args": {}},
            {"name": "b", "args": {}},
        ]
        assert v.check_acyclic(chain) is True

    def test_longer_acyclic_chain(self):
        v = ToolChainValidator()
        chain = [
            {"name": "a", "args": {}},
            {"name": "b", "args": {"x": "$a"}},
            {"name": "c", "args": {"x": "$b", "y": "$a"}},
        ]
        assert v.check_acyclic(chain) is True


class TestToolChainError:
    def test_is_exception(self):
        assert issubclass(ToolChainError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(ToolChainError, match="boom"):
            raise ToolChainError("boom")


class TestToolChainValidatorRegistry:
    def test_exists(self):
        assert TOOL_CHAIN_VALIDATOR_REGISTRY is not None

    def test_is_instance(self):
        assert isinstance(TOOL_CHAIN_VALIDATOR_REGISTRY, ToolChainValidator)
