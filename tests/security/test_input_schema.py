"""Tests for input schema."""

from __future__ import annotations

from src.security.input_schema import InputSchema, ParamSpec


class TestInputSchema:
    def test_valid_data(self):
        schema = InputSchema(params=[ParamSpec("name", str)])
        ok, _ = schema.validate({"name": "test"})
        assert ok is True

    def test_missing_required(self):
        schema = InputSchema(params=[ParamSpec("name", str, required=True)])
        ok, errors = schema.validate({})
        assert ok is False
        assert any("name" in e for e in errors)

    def test_type_mismatch(self):
        schema = InputSchema(params=[ParamSpec("age", int)])
        ok, errors = schema.validate({"age": "not_int"})
        assert ok is False

    def test_optional_param(self):
        schema = InputSchema(params=[ParamSpec("name", str, required=False)])
        ok, _ = schema.validate({})
        assert ok is True
