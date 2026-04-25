"""Tests for src/protocol/schema_validator.py"""

import pytest
from src.protocol.schema_validator import (
    MessageSchema,
    SchemaField,
    SchemaType,
    SchemaValidator,
    ValidationError,
    PROTOCOL_REGISTRY,
)


def _simple_schema(name: str = "Msg") -> MessageSchema:
    return MessageSchema(
        name=name,
        fields=[
            SchemaField(name="text", type_name="str"),
            SchemaField(name="count", type_name="int"),
            SchemaField(name="score", type_name="float"),
            SchemaField(name="active", type_name="bool"),
            SchemaField(name="tags", type_name="list"),
            SchemaField(name="meta", type_name="dict"),
        ],
    )


# ---------------------------------------------------------------------------
# define_schema
# ---------------------------------------------------------------------------


def test_define_schema_registers():
    sv = SchemaValidator()
    sv.define_schema(_simple_schema())
    # Should not raise on validate after defining
    errors = sv.validate(
        {"text": "hi", "count": 1, "score": 0.5, "active": True, "tags": [], "meta": {}},
        "Msg",
    )
    assert errors == []


def test_validate_unknown_schema_raises_keyerror():
    sv = SchemaValidator()
    with pytest.raises(KeyError):
        sv.validate({}, "DoesNotExist")


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_missing_required_field_produces_error():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema(name="S", fields=[SchemaField("x", "str")]))
    errors = sv.validate({}, "S")
    assert len(errors) == 1
    assert errors[0].field == "x"


def test_optional_field_missing_is_ok():
    sv = SchemaValidator()
    sv.define_schema(
        MessageSchema(
            name="S",
            fields=[SchemaField("x", "str", required=False)],
        )
    )
    errors = sv.validate({}, "S")
    assert errors == []


# ---------------------------------------------------------------------------
# Type checking — correct types pass
# ---------------------------------------------------------------------------


def test_str_type_passes():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "str")]))
    assert sv.is_valid({"v": "hello"}, "S")


def test_int_type_passes():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "int")]))
    assert sv.is_valid({"v": 42}, "S")


def test_float_type_accepts_int_literal():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "float")]))
    assert sv.is_valid({"v": 3}, "S")


def test_float_type_accepts_float_literal():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "float")]))
    assert sv.is_valid({"v": 3.14}, "S")


def test_bool_type_passes():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "bool")]))
    assert sv.is_valid({"v": False}, "S")


def test_list_type_passes():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "list")]))
    assert sv.is_valid({"v": [1, 2, 3]}, "S")


def test_dict_type_passes():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "dict")]))
    assert sv.is_valid({"v": {"a": 1}}, "S")


# ---------------------------------------------------------------------------
# Type checking — wrong types fail
# ---------------------------------------------------------------------------


def test_wrong_type_produces_error():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "int")]))
    errors = sv.validate({"v": "not-an-int"}, "S")
    assert len(errors) == 1
    assert errors[0].field == "v"
    assert errors[0].value == "not-an-int"


def test_bool_not_accepted_as_int():
    """bool is a subclass of int in Python; validator must reject it for int fields."""
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "int")]))
    errors = sv.validate({"v": True}, "S")
    assert len(errors) == 1


def test_bool_not_accepted_as_float():
    sv = SchemaValidator()
    sv.define_schema(MessageSchema("S", [SchemaField("v", "float")]))
    errors = sv.validate({"v": True}, "S")
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------


def test_is_valid_true_for_valid_data():
    sv = SchemaValidator()
    sv.define_schema(_simple_schema())
    assert sv.is_valid(
        {"text": "t", "count": 1, "score": 1.0, "active": True, "tags": [], "meta": {}},
        "Msg",
    )


def test_is_valid_false_for_invalid_data():
    sv = SchemaValidator()
    sv.define_schema(_simple_schema())
    assert not sv.is_valid(
        {"text": 123, "count": 1, "score": 1.0, "active": True, "tags": [], "meta": {}},
        "Msg",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_protocol_registry_contains_schema_validator():
    assert "schema_validator" in PROTOCOL_REGISTRY
    assert isinstance(PROTOCOL_REGISTRY["schema_validator"], SchemaValidator)
