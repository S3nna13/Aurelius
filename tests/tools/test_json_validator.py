"""Tests for JSON validator."""
from __future__ import annotations

import pytest

from src.tools.json_validator import JSONValidator, JSONSchema


class TestJSONValidator:
    def test_missing_required_field(self):
        jv = JSONValidator()
        schema = JSONSchema(required_fields=["name", "id"])
        ok, errors = jv.validate({"name": "test"}, schema)
        assert ok is False
        assert any("id" in e for e in errors)

    def test_field_type_mismatch(self):
        jv = JSONValidator()
        schema = JSONSchema(field_types={"age": int})
        ok, errors = jv.validate({"age": "not_a_number"}, schema)
        assert ok is False

    def test_valid_data(self):
        jv = JSONValidator()
        schema = JSONSchema(required_fields=["x"], field_types={"x": int})
        ok, _ = jv.validate({"x": 42}, schema)
        assert ok is True