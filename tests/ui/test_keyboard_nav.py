"""Tests for src.ui.keyboard_nav."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.keyboard_nav import (
    KEYBOARD_NAV_REGISTRY,
    KeyBinding,
    KeyBindingError,
    KeyboardNav,
)

# ---------------------------------------------------------------------------
# Built-in bindings
# ---------------------------------------------------------------------------


def test_bindings_contains_arrow_keys() -> None:
    """All four arrow keys must be registered after import."""
    for key in ("up", "down", "left", "right"):
        assert key in KeyboardNav.BINDINGS, f"missing arrow key: {key}"


def test_bindings_contains_enter() -> None:
    assert "enter" in KeyboardNav.BINDINGS


def test_bindings_contains_escape() -> None:
    assert "escape" in KeyboardNav.BINDINGS


def test_bindings_contains_slash() -> None:
    assert "/" in KeyboardNav.BINDINGS


def test_bindings_contains_question_mark() -> None:
    assert "?" in KeyboardNav.BINDINGS


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_dispatch_enter_returns_confirm() -> None:
    action = KeyboardNav.dispatch("Enter", [])
    assert action == "confirm"


def test_dispatch_escape_returns_cancel() -> None:
    action = KeyboardNav.dispatch("Escape", [])
    assert action == "cancel"


def test_dispatch_slash_returns_command_palette() -> None:
    action = KeyboardNav.dispatch("/", [])
    assert action == "command_palette"


def test_dispatch_question_mark_returns_help() -> None:
    action = KeyboardNav.dispatch("?", [])
    assert action == "help"


def test_dispatch_unknown_key_returns_none() -> None:
    result = KeyboardNav.dispatch("unknown_key_xyz")
    assert result is None


def test_dispatch_unknown_key_does_not_crash() -> None:
    # Must not raise even for exotic keys.
    KeyboardNav.dispatch("F99", ["hyper", "super"])


def test_dispatch_none_modifiers_same_as_empty() -> None:
    action_none = KeyboardNav.dispatch("Enter", None)
    action_empty = KeyboardNav.dispatch("Enter", [])
    assert action_none == action_empty


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_new_binding() -> None:
    binding = KeyBinding(
        key="_testkey_a",
        description="test binding",
        action="test_action_a",
    )
    KeyboardNav.register(binding)
    assert KeyboardNav.dispatch("_testkey_a") == "test_action_a"
    del KeyboardNav.BINDINGS["_testkey_a"]


def test_register_duplicate_raises() -> None:
    binding = KeyBinding(key="_testkey_dup", description="dup", action="dup_action")
    KeyboardNav.register(binding)
    try:
        with pytest.raises(KeyBindingError):
            KeyboardNav.register(KeyBinding(key="_testkey_dup", description="dup2", action="dup2"))
    finally:
        del KeyboardNav.BINDINGS["_testkey_dup"]


def test_register_wrong_type_raises() -> None:
    with pytest.raises(KeyBindingError):
        KeyboardNav.register("not a KeyBinding")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render_help
# ---------------------------------------------------------------------------


def test_render_help_does_not_crash() -> None:
    console = Console(record=True)
    KeyboardNav.render_help(console)


def test_render_help_output_contains_enter() -> None:
    console = Console(record=True)
    KeyboardNav.render_help(console)
    output = console.export_text()
    assert "enter" in output.lower()


# ---------------------------------------------------------------------------
# KEYBOARD_NAV_REGISTRY
# ---------------------------------------------------------------------------


def test_keyboard_nav_registry_is_dict() -> None:
    assert isinstance(KEYBOARD_NAV_REGISTRY, dict)
