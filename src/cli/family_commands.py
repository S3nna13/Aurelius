"""CLI subcommands for model family + variant listing and selection.

Under Meta-Prompt v5 (Model Family Edition), the Aurelius CLI grows a
``family`` command group backed by the existing family registry and
release-track router:

    * ``family list``                       -- list registered families
    * ``family show <family_name>``         -- show a family + variants
    * ``family variants <family_name>``     -- list variants in a family
    * ``family manifest <family/variant>``  -- dump a manifest as JSON
    * ``family compare <required> <candidate>`` -- compare two manifests
    * ``family check <family/variant>       -- apply release-track router
        --policy {production|internal|dev}
        [--override FLAG ...]``

Handlers are pure functions of ``(argparse.Namespace, TextIO) -> int`` and
never call ``sys.exit`` themselves. Invalid ``--policy`` values are
rejected by argparse's own ``choices`` (which raises ``SystemExit`` in
the usual argparse way).

Pure stdlib (``argparse, json, sys, io, typing``). No foreign imports.
No new config flags. Additive within this file; ``src/cli/main.py`` is
touched only through an additive branch guarded by a subparser.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, TextIO

from src.model.family import MODEL_FAMILY_REGISTRY, MODEL_VARIANT_REGISTRY
from src.model.compatibility import check_manifest_compatibility
from src.model.manifest import dump_manifest
from src.model.manifest_v2 import compare_backend_contracts
from src.model.release_track_router import (
    POLICY_REGISTRY,
    ReleaseTrackRouter,
)


__all__ = [
    "FAMILY_COMMAND_HANDLERS",
    "build_family_parser",
    "handle_family_list",
    "handle_family_show",
    "handle_family_variants",
    "handle_family_manifest",
    "handle_family_compare",
    "handle_family_check",
    "dispatch_family_command",
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_family_list(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Emit a JSON array of registered family names (sorted)."""
    stream = out_stream if out_stream is not None else sys.stdout
    names = sorted(MODEL_FAMILY_REGISTRY.keys())
    stream.write(json.dumps({"families": names}, sort_keys=True))
    stream.write("\n")
    return 0


def handle_family_show(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Emit JSON describing one family + its variants."""
    stream = out_stream if out_stream is not None else sys.stdout
    name = getattr(args, "family_name", None)
    if name is None or name not in MODEL_FAMILY_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown family: {name!r}",
                    "known_families": sorted(MODEL_FAMILY_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1
    family = MODEL_FAMILY_REGISTRY[name]
    payload = {
        "family_name": family.family_name,
        "default_variant": family.default_variant,
        "variants": [
            {
                "name": vname,
                "description": variant.description,
                "release_notes": variant.release_notes,
                "release_track": variant.manifest.release_track,
            }
            for vname, variant in family.variants.items()
        ],
    }
    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")
    return 0


def handle_family_variants(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Emit a JSON array of variant names within a family."""
    stream = out_stream if out_stream is not None else sys.stdout
    name = getattr(args, "family_name", None)
    if name is None or name not in MODEL_FAMILY_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown family: {name!r}",
                    "known_families": sorted(MODEL_FAMILY_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1
    family = MODEL_FAMILY_REGISTRY[name]
    stream.write(
        json.dumps(
            {
                "family_name": family.family_name,
                "variants": list(family.list_variants()),
            },
            sort_keys=True,
        )
    )
    stream.write("\n")
    return 0


def handle_family_manifest(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Dump a variant's :class:`FamilyManifest` as JSON."""
    stream = out_stream if out_stream is not None else sys.stdout
    variant_id = getattr(args, "variant_id", None)
    if variant_id is None or variant_id not in MODEL_VARIANT_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown variant id: {variant_id!r}",
                    "known_variants": sorted(MODEL_VARIANT_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1
    variant = MODEL_VARIANT_REGISTRY[variant_id]
    stream.write(json.dumps(dump_manifest(variant.manifest), sort_keys=True))
    stream.write("\n")
    return 0


def handle_family_compare(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Compare two variant manifests and print compatibility as JSON."""
    stream = out_stream if out_stream is not None else sys.stdout
    required_id = getattr(args, "required_variant_id", None)
    candidate_id = getattr(args, "candidate_variant_id", None)

    if required_id is None or required_id not in MODEL_VARIANT_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown required variant id: {required_id!r}",
                    "known_variants": sorted(MODEL_VARIANT_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1

    if candidate_id is None or candidate_id not in MODEL_VARIANT_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown candidate variant id: {candidate_id!r}",
                    "known_variants": sorted(MODEL_VARIANT_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1

    required = MODEL_VARIANT_REGISTRY[required_id]
    candidate = MODEL_VARIANT_REGISTRY[candidate_id]
    verdict = check_manifest_compatibility(
        required.manifest, candidate.manifest
    )
    backend_verdict = compare_backend_contracts(
        required.manifest, candidate.manifest
    )
    payload = {
        "required_variant_id": required_id,
        "candidate_variant_id": candidate_id,
        "compatible": verdict.compatible,
        "severity": verdict.severity,
        "reasons": list(verdict.reasons),
        "backend_contract_verdict": backend_verdict,
    }
    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")
    return 0 if verdict.compatible else 1


def handle_family_check(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Route a variant under a named policy and print the decision as JSON."""
    stream = out_stream if out_stream is not None else sys.stdout
    variant_id = getattr(args, "variant_id", None)
    policy_name = getattr(args, "policy", None)

    if variant_id is None or variant_id not in MODEL_VARIANT_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown variant id: {variant_id!r}",
                    "known_variants": sorted(MODEL_VARIANT_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1

    if policy_name is None or policy_name not in POLICY_REGISTRY:
        stream.write(
            json.dumps(
                {
                    "error": f"unknown policy: {policy_name!r}",
                    "known_policies": sorted(POLICY_REGISTRY.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1

    override_flags = set(getattr(args, "override", None) or [])
    # Accept short forms like "allow_research" or bare "research".
    normalized: set[str] = set()
    for flag in override_flags:
        if flag.startswith("allow_"):
            normalized.add(flag)
        else:
            normalized.add(f"allow_{flag}")

    router = ReleaseTrackRouter(POLICY_REGISTRY[policy_name])
    decision = router.resolve_by_variant_id(variant_id, override_flags=normalized)
    payload = {
        "allowed": decision.allowed,
        "reason": decision.reason,
        "variant_id": decision.variant_id,
        "track": decision.track.value,
        "warnings": list(decision.warnings),
        "policy": policy_name,
    }
    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")
    return 0 if decision.allowed else 1


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

FAMILY_COMMAND_HANDLERS: dict[
    str, Callable[[argparse.Namespace, TextIO], int]
] = {
    "list": handle_family_list,
    "show": handle_family_show,
    "variants": handle_family_variants,
    "manifest": handle_family_manifest,
    "compare": handle_family_compare,
    "check": handle_family_check,
}


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def build_family_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``family`` command group to an existing subparsers action.

    Returns the ``family`` parser so callers can further customize it if
    they need to (tests do this to introspect the sub-subparser shape).
    """
    family_parser = subparsers.add_parser(
        "family",
        help="inspect model families and variants",
        description="Model family + variant listing and selection.",
    )
    family_sub = family_parser.add_subparsers(
        dest="family_command", metavar="family_command"
    )

    # family list
    family_sub.add_parser("list", help="list registered families")

    # family show <family_name>
    show_p = family_sub.add_parser("show", help="show a family and its variants")
    show_p.add_argument("family_name", help="family name (e.g. 'aurelius')")

    # family variants <family_name>
    variants_p = family_sub.add_parser(
        "variants", help="list variant names in a family"
    )
    variants_p.add_argument("family_name", help="family name")

    # family manifest <family/variant>
    manifest_p = family_sub.add_parser(
        "manifest", help="dump a variant manifest as JSON"
    )
    manifest_p.add_argument(
        "variant_id", help="variant id in the form 'family/variant'"
    )

    # family compare <required> <candidate>
    compare_p = family_sub.add_parser(
        "compare", help="compare two variant manifests"
    )
    compare_p.add_argument(
        "required_variant_id", help="required variant id (family/variant)"
    )
    compare_p.add_argument(
        "candidate_variant_id", help="candidate variant id (family/variant)"
    )

    # family check <family/variant> --policy ... [--override FLAG]
    check_p = family_sub.add_parser(
        "check", help="apply the release-track router to a variant"
    )
    check_p.add_argument(
        "variant_id", help="variant id in the form 'family/variant'"
    )
    check_p.add_argument(
        "--policy",
        required=True,
        choices=("production", "internal", "dev"),
        help="release-track policy to apply",
    )
    check_p.add_argument(
        "--override",
        action="append",
        default=None,
        metavar="FLAG",
        help=(
            "override flag (repeatable); 'allow_research' or bare "
            "'research' are both accepted"
        ),
    )

    return family_parser


def dispatch_family_command(
    args: argparse.Namespace, out_stream: TextIO | None = None
) -> int:
    """Route an ``argparse.Namespace`` from ``build_family_parser`` to a handler."""
    name = getattr(args, "family_command", None)
    if name is None:
        stream = out_stream if out_stream is not None else sys.stdout
        stream.write(
            json.dumps(
                {
                    "error": "no family subcommand given",
                    "known_subcommands": sorted(FAMILY_COMMAND_HANDLERS.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1
    handler = FAMILY_COMMAND_HANDLERS.get(name)
    if handler is None:
        stream = out_stream if out_stream is not None else sys.stdout
        stream.write(
            json.dumps(
                {
                    "error": f"unknown family subcommand: {name!r}",
                    "known_subcommands": sorted(FAMILY_COMMAND_HANDLERS.keys()),
                },
                sort_keys=True,
            )
        )
        stream.write("\n")
        return 1
    return handler(args, out_stream if out_stream is not None else sys.stdout)
