"""Unit tests for src/cli/family_commands.py."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import replace

import pytest

from src.cli import family_commands as fc
from src.model.family import MODEL_VARIANT_REGISTRY, ModelVariant, get_family
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    fc.build_family_parser(sub)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_handler_registry_non_empty():
    assert fc.FAMILY_COMMAND_HANDLERS
    assert set(fc.FAMILY_COMMAND_HANDLERS.keys()) == {
        "list",
        "show",
        "variants",
        "manifest",
        "compare",
        "check",
    }


def test_all_handlers_callable():
    for name, handler in fc.FAMILY_COMMAND_HANDLERS.items():
        assert callable(handler), name


def test_build_family_parser_returns_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    family_p = fc.build_family_parser(sub)
    assert isinstance(family_p, argparse.ArgumentParser)


# ---------------------------------------------------------------------------
# Parsing tests per subcommand
# ---------------------------------------------------------------------------


def test_parse_family_list():
    ns = _parse(["family", "list"])
    assert ns.command == "family"
    assert ns.family_command == "list"


def test_parse_family_show_namespace_shape():
    ns = _parse(["family", "show", "aurelius"])
    assert ns.family_command == "show"
    assert ns.family_name == "aurelius"


def test_parse_family_variants():
    ns = _parse(["family", "variants", "aurelius"])
    assert ns.family_command == "variants"
    assert ns.family_name == "aurelius"


def test_parse_family_manifest():
    ns = _parse(["family", "manifest", "aurelius/base-1.395b"])
    assert ns.family_command == "manifest"
    assert ns.variant_id == "aurelius/base-1.395b"


def test_parse_family_compare():
    ns = _parse(
        [
            "family",
            "compare",
            "aurelius/base-1.395b",
            "aurelius/base-1.395b",
        ]
    )
    assert ns.family_command == "compare"
    assert ns.required_variant_id == "aurelius/base-1.395b"
    assert ns.candidate_variant_id == "aurelius/base-1.395b"


def test_parse_family_check():
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "dev",
        ]
    )
    assert ns.family_command == "check"
    assert ns.variant_id == "aurelius/base-1.395b"
    assert ns.policy == "dev"


def test_parse_family_check_invalid_policy_raises_system_exit():
    with pytest.raises(SystemExit):
        _parse(
            [
                "family",
                "check",
                "aurelius/base-1.395b",
                "--policy",
                "bogus",
            ]
        )


# ---------------------------------------------------------------------------
# Handler behaviour
# ---------------------------------------------------------------------------


def test_list_outputs_aurelius():
    buf = io.StringIO()
    rc = fc.handle_family_list(_parse(["family", "list"]), buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "aurelius" in payload["families"]


def test_show_unknown_family_returns_nonzero():
    buf = io.StringIO()
    rc = fc.handle_family_show(_parse(["family", "show", "nosuch"]), buf)
    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert "error" in payload


def test_show_known_family_returns_zero():
    buf = io.StringIO()
    rc = fc.handle_family_show(_parse(["family", "show", "aurelius"]), buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["family_name"] == "aurelius"
    assert any(v["name"] == "base-1.395b" for v in payload["variants"])


def test_variants_lists_base_1_395b():
    buf = io.StringIO()
    rc = fc.handle_family_variants(_parse(["family", "variants", "aurelius"]), buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "base-1.395b" in payload["variants"]


def test_variants_unknown_family_nonzero():
    buf = io.StringIO()
    rc = fc.handle_family_variants(_parse(["family", "variants", "nosuch"]), buf)
    assert rc == 1


def test_manifest_outputs_valid_json_roundtrip():
    buf = io.StringIO()
    rc = fc.handle_family_manifest(_parse(["family", "manifest", "aurelius/base-1.395b"]), buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    # JSON round-trip stability
    assert json.loads(json.dumps(payload)) == payload
    assert payload["family_name"] == "aurelius"
    assert payload["variant_name"] == "base-1.395b"


def test_manifest_unknown_variant_nonzero():
    buf = io.StringIO()
    rc = fc.handle_family_manifest(_parse(["family", "manifest", "aurelius/nope"]), buf)
    assert rc == 1


def test_compare_same_variant_exact():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "compare",
            "aurelius/base-1.395b",
            "aurelius/base-1.395b",
        ]
    )
    rc = fc.handle_family_compare(ns, buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["compatible"] is True
    assert payload["severity"] == "exact"
    assert payload["backend_contract_verdict"] == "exact"


def test_compare_backend_mismatch_reports_major_and_fails() -> None:
    variant_name = "cli-compare-jax"
    manifest = replace(
        AURELIUS_REFERENCE_MANIFEST,
        variant_name=variant_name,
        backend_name="jax",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    variant = ModelVariant(
        manifest=manifest,
        description="Temporary compare fixture",
    )
    family = get_family("aurelius")
    family.add_variant(variant_name, variant)
    MODEL_VARIANT_REGISTRY[manifest.registry_key] = variant
    try:
        buf = io.StringIO()
        ns = _parse(
            [
                "family",
                "compare",
                "aurelius/base-1.395b",
                manifest.registry_key,
            ]
        )
        rc = fc.handle_family_compare(ns, buf)
        assert rc == 1
        payload = json.loads(buf.getvalue())
        assert payload["compatible"] is False
        assert payload["severity"] == "major_break"
        assert payload["backend_contract_verdict"] == "major_break"
        assert any("backend contract" in r for r in payload["reasons"])
    finally:
        family.variants.pop(variant_name, None)
        MODEL_VARIANT_REGISTRY.pop(manifest.registry_key, None)


def test_check_production_policy_on_research_denies():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "production",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert payload["allowed"] is False
    assert payload["track"] == "research"


def test_check_dev_policy_allows_research():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "dev",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["allowed"] is True


def test_check_internal_policy_with_override_allows_research():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "internal",
            "--override",
            "allow_research",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["allowed"] is True


def test_check_internal_policy_bare_research_override():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "internal",
            "--override",
            "research",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 0


def test_check_internal_policy_without_override_denies_research():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "internal",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 1


def test_check_unknown_variant_nonzero():
    buf = io.StringIO()
    ns = _parse(
        [
            "family",
            "check",
            "aurelius/nope",
            "--policy",
            "dev",
        ]
    )
    rc = fc.handle_family_check(ns, buf)
    assert rc == 1


def test_list_output_is_deterministic():
    buf1 = io.StringIO()
    buf2 = io.StringIO()
    fc.handle_family_list(_parse(["family", "list"]), buf1)
    fc.handle_family_list(_parse(["family", "list"]), buf2)
    assert buf1.getvalue() == buf2.getvalue()


def test_dispatch_family_command_routes_list():
    buf = io.StringIO()
    rc = fc.dispatch_family_command(_parse(["family", "list"]), buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "families" in payload


def test_dispatch_family_command_missing_subcommand():
    ns = argparse.Namespace()  # no family_command attribute
    buf = io.StringIO()
    rc = fc.dispatch_family_command(ns, buf)
    assert rc == 1
