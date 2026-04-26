"""Unit tests for src/agent/patch_synthesis.py (16 tests)."""

from __future__ import annotations

import pytest

from src.agent.patch_synthesis import (
    PATCH_REGISTRY,
    Patch,
    PatchError,
    PatchSynthesizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORIGINAL = """\
def greet(name):
    return "Hello, " + name

def farewell(name):
    return "Goodbye, " + name
"""

MODIFIED = """\
def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"
"""

IDENTICAL = "no changes here\n"


@pytest.fixture()
def synth() -> PatchSynthesizer:
    return PatchSynthesizer()


@pytest.fixture()
def patch_of_diff(synth) -> Patch:
    return synth.make_patch(ORIGINAL, MODIFIED, filename="greet.py")


@pytest.fixture()
def patch_str(synth, patch_of_diff) -> str:
    return synth.patch_to_str(patch_of_diff)


# ---------------------------------------------------------------------------
# make_patch
# ---------------------------------------------------------------------------


def test_make_patch_identical_zero_hunks(synth):
    patch = synth.make_patch(IDENTICAL, IDENTICAL)
    assert len(patch.hunks) == 0


def test_make_patch_different_has_hunks(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED)
    assert len(patch.hunks) >= 1


def test_make_patch_returns_patch_instance(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED)
    assert isinstance(patch, Patch)


def test_make_patch_filename_stored(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED, filename="my_file.py")
    assert patch.filename == "my_file.py"


def test_make_patch_patch_id_set(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED)
    assert isinstance(patch.patch_id, str)
    assert len(patch.patch_id) > 0


def test_make_patch_hunk_has_lines(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED)
    for hunk in patch.hunks:
        assert isinstance(hunk.lines, list)
        assert len(hunk.lines) > 0


# ---------------------------------------------------------------------------
# patch_to_str
# ---------------------------------------------------------------------------


def test_patch_to_str_starts_with_minus_minus_minus(synth, patch_of_diff):
    s = synth.patch_to_str(patch_of_diff)
    assert s.startswith("---")


def test_patch_to_str_contains_plus_plus_plus(synth, patch_of_diff):
    s = synth.patch_to_str(patch_of_diff)
    assert "+++" in s


def test_patch_to_str_contains_hunk_header(synth, patch_of_diff):
    s = synth.patch_to_str(patch_of_diff)
    assert "@@" in s


def test_patch_to_str_identical_empty(synth):
    patch = synth.make_patch(IDENTICAL, IDENTICAL)
    s = synth.patch_to_str(patch)
    assert s == ""


# ---------------------------------------------------------------------------
# apply_patch  — round-trip
# ---------------------------------------------------------------------------


def test_apply_patch_roundtrip(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED, filename="greet.py")
    recovered = synth.apply_patch(ORIGINAL, patch)
    assert recovered == MODIFIED


def test_apply_patch_identity(synth):
    """Applying a zero-hunk patch leaves source unchanged."""
    patch = synth.make_patch(IDENTICAL, IDENTICAL)
    result = synth.apply_patch(IDENTICAL, patch)
    assert result == IDENTICAL


def test_apply_patch_wrong_old_lines_raises_patch_error(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED, filename="greet.py")
    wrong_source = "completely different content\nthat won't match\n"
    with pytest.raises(PatchError):
        synth.apply_patch(wrong_source, patch)


# ---------------------------------------------------------------------------
# apply_str_patch
# ---------------------------------------------------------------------------


def test_apply_str_patch_valid_roundtrip(synth):
    patch = synth.make_patch(ORIGINAL, MODIFIED, filename="greet.py")
    patch_str = synth.patch_to_str(patch)
    recovered = synth.apply_str_patch(ORIGINAL, patch_str)
    assert recovered == MODIFIED


def test_apply_str_patch_malformed_raises_patch_error(synth):
    with pytest.raises(PatchError):
        synth.apply_str_patch(ORIGINAL, "this is not a diff at all")


def test_apply_str_patch_empty_raises_patch_error(synth):
    with pytest.raises(PatchError):
        synth.apply_str_patch(ORIGINAL, "")


# ---------------------------------------------------------------------------
# PatchError
# ---------------------------------------------------------------------------


def test_patch_error_is_exception_subclass():
    assert issubclass(PatchError, Exception)


def test_patch_error_can_be_raised_and_caught():
    with pytest.raises(PatchError, match="test error"):
        raise PatchError("test error")


# ---------------------------------------------------------------------------
# PATCH_REGISTRY
# ---------------------------------------------------------------------------


def test_patch_registry_contains_unified():
    assert "unified" in PATCH_REGISTRY


def test_patch_registry_unified_is_patch_synthesizer():
    assert PATCH_REGISTRY["unified"] is PatchSynthesizer
