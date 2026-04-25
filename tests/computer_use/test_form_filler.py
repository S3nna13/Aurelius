"""
test_form_filler.py
Tests for form_filler.py  (>=28 tests)
"""

import pytest
from src.computer_use.form_filler import (
    FieldType,
    FormField,
    FormFillResult,
    FormFiller,
    FORM_FILLER_REGISTRY,
    REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filler(*fields: FormField) -> FormFiller:
    ff = FormFiller()
    for f in fields:
        ff.register_field(f)
    return ff


def text_field(fid: str = "name", required: bool = False) -> FormField:
    return FormField(field_id=fid, field_type=FieldType.TEXT, required=required)


def email_field(fid: str = "email", required: bool = False) -> FormField:
    return FormField(field_id=fid, field_type=FieldType.EMAIL, required=required)


def number_field(fid: str = "age", required: bool = False) -> FormField:
    return FormField(field_id=fid, field_type=FieldType.NUMBER, required=required)


def checkbox_field(fid: str = "agree") -> FormField:
    return FormField(field_id=fid, field_type=FieldType.CHECKBOX)


def select_field(fid: str = "color", options: list[str] | None = None) -> FormField:
    return FormField(
        field_id=fid,
        field_type=FieldType.SELECT,
        options=options or ["red", "green", "blue"],
    )


def textarea_field(fid: str = "bio") -> FormField:
    return FormField(field_id=fid, field_type=FieldType.TEXTAREA)


def password_field(fid: str = "pw") -> FormField:
    return FormField(field_id=fid, field_type=FieldType.PASSWORD)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

def test_registry_contains_default():
    assert "default" in FORM_FILLER_REGISTRY


def test_registry_alias():
    assert REGISTRY is FORM_FILLER_REGISTRY


def test_registry_default_is_class():
    assert FORM_FILLER_REGISTRY["default"] is FormFiller


# ---------------------------------------------------------------------------
# FormFillResult — frozen
# ---------------------------------------------------------------------------

def test_form_fill_result_frozen():
    r = FormFillResult(field_id="f", value="v", success=True)
    with pytest.raises((AttributeError, TypeError)):
        r.success = False  # type: ignore[misc]


def test_form_fill_result_defaults():
    r = FormFillResult(field_id="f", value="v", success=True)
    assert r.error == ""


# ---------------------------------------------------------------------------
# TEXT fill
# ---------------------------------------------------------------------------

def test_fill_text_success():
    ff = make_filler(text_field())
    r = ff.fill("name", "Alice")
    assert r.success is True
    assert r.value == "Alice"
    assert r.error == ""


def test_fill_text_empty_not_required():
    ff = make_filler(text_field())
    r = ff.fill("name", "")
    assert r.success is True


def test_fill_password_success():
    ff = make_filler(password_field())
    r = ff.fill("pw", "s3cr3t!")
    assert r.success is True


def test_fill_textarea_success():
    ff = make_filler(textarea_field())
    r = ff.fill("bio", "Hello world")
    assert r.success is True


# ---------------------------------------------------------------------------
# EMAIL fill
# ---------------------------------------------------------------------------

def test_fill_email_valid():
    ff = make_filler(email_field())
    r = ff.fill("email", "user@example.com")
    assert r.success is True


def test_fill_email_invalid_no_at():
    ff = make_filler(email_field())
    r = ff.fill("email", "notanemail")
    assert r.success is False
    assert "@" in r.error or "email" in r.error.lower() or "contain" in r.error.lower()


def test_fill_email_invalid_returns_value():
    ff = make_filler(email_field())
    r = ff.fill("email", "bad")
    assert r.value == "bad"
    assert r.field_id == "email"


# ---------------------------------------------------------------------------
# NUMBER fill
# ---------------------------------------------------------------------------

def test_fill_number_integer():
    ff = make_filler(number_field())
    r = ff.fill("age", "25")
    assert r.success is True


def test_fill_number_float():
    ff = make_filler(number_field())
    r = ff.fill("age", "3.14")
    assert r.success is True


def test_fill_number_negative():
    ff = make_filler(number_field())
    r = ff.fill("age", "-10")
    assert r.success is True


def test_fill_number_invalid():
    ff = make_filler(number_field())
    r = ff.fill("age", "abc")
    assert r.success is False
    assert r.error != ""


# ---------------------------------------------------------------------------
# CHECKBOX fill
# ---------------------------------------------------------------------------

def test_fill_checkbox_true():
    ff = make_filler(checkbox_field())
    r = ff.fill("agree", "true")
    assert r.success is True


def test_fill_checkbox_false():
    ff = make_filler(checkbox_field())
    r = ff.fill("agree", "false")
    assert r.success is True


def test_fill_checkbox_invalid():
    ff = make_filler(checkbox_field())
    r = ff.fill("agree", "yes")
    assert r.success is False
    assert r.error != ""


def test_fill_checkbox_invalid_bool_cased():
    ff = make_filler(checkbox_field())
    r = ff.fill("agree", "True")  # capital T — not allowed
    assert r.success is False


# ---------------------------------------------------------------------------
# SELECT fill
# ---------------------------------------------------------------------------

def test_fill_select_valid_option():
    ff = make_filler(select_field())
    r = ff.fill("color", "red")
    assert r.success is True


def test_fill_select_invalid_option():
    ff = make_filler(select_field())
    r = ff.fill("color", "purple")
    assert r.success is False
    assert r.error != ""


def test_fill_select_invalid_returns_field_id():
    ff = make_filler(select_field())
    r = ff.fill("color", "purple")
    assert r.field_id == "color"


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------

def test_required_field_empty_value_error():
    ff = make_filler(text_field(required=True))
    r = ff.fill("name", "")
    assert r.success is False
    assert "required" in r.error.lower() or r.error != ""


def test_required_field_nonempty_success():
    ff = make_filler(text_field(required=True))
    r = ff.fill("name", "Bob")
    assert r.success is True


def test_required_email_empty():
    ff = make_filler(email_field(required=True))
    r = ff.fill("email", "")
    assert r.success is False


# ---------------------------------------------------------------------------
# fill_form
# ---------------------------------------------------------------------------

def test_fill_form_returns_list():
    ff = make_filler(text_field("first"), text_field("last"))
    results = ff.fill_form({"first": "John", "last": "Doe"})
    assert len(results) == 2
    assert all(r.success for r in results)


def test_fill_form_partial_failure():
    ff = make_filler(email_field("email"), text_field("name"))
    results = ff.fill_form({"email": "bad", "name": "Alice"})
    by_id = {r.field_id: r for r in results}
    assert by_id["email"].success is False
    assert by_id["name"].success is True


def test_fill_form_empty_dict():
    ff = make_filler(text_field())
    results = ff.fill_form({})
    assert results == []


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_true():
    ff = make_filler(email_field())
    assert ff.validate("email", "a@b.com") is True


def test_validate_false():
    ff = make_filler(email_field())
    assert ff.validate("email", "notvalid") is False


def test_validate_unknown_field():
    ff = FormFiller()
    assert ff.validate("ghost", "anything") is False


# ---------------------------------------------------------------------------
# Unregistered field
# ---------------------------------------------------------------------------

def test_fill_unregistered_field():
    ff = FormFiller()
    r = ff.fill("ghost", "value")
    assert r.success is False
    assert r.error != ""
