"""CycloneDX 1.5 SBOM generator for Aurelius.

Supply-chain hardening: CycloneDX 1.5 SBOM per v8 release gate.
Finding AUR-SEC-2026-0022; CWE-1104 (use of unmaintained third-party
components) defense.

This module parses ``uv.lock`` (TOML) and emits a CycloneDX 1.5 JSON
BOM with deterministic component ordering, stable purls, and a sha256
digest of the rendered file.  Standard library only: no external SBOM
dependencies are pulled in at runtime.

Usage::

    from src.security.sbom_generator import generate_sbom
    result = generate_sbom()             # reads uv.lock, writes .aurelius-sbom.json
    print(result.component_count, result.sha256_of_sbom)

Or invoke as a module::

    python -m src.security.sbom_generator [--lockfile uv.lock] [--output out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── constants ────────────────────────────────────────────────────────────────

SBOM_SPEC_VERSION = "1.5"
SBOM_BOM_FORMAT = "CycloneDX"
SBOM_TOOL_VENDOR = "Aurelius"
SBOM_TOOL_NAME = "aurelius-sbom-generator"
SBOM_TOOL_VERSION = "0.1.0"
SBOM_ROOT_COMPONENT_NAME = "aurelius"
SBOM_ROOT_COMPONENT_VERSION = "0.1.0"


# ── result dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SBOMResult:
    """Summary of an SBOM generation run."""

    component_count: int
    output_path: Path
    sha256_of_sbom: str


# ── helpers ──────────────────────────────────────────────────────────────────


def _iso_utc_now() -> str:
    """Return an ISO-8601 timestamp terminated with ``Z`` (UTC)."""
    # datetime.isoformat uses +00:00; swap to Z for CycloneDX style.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _purl(name: str, version: str) -> str:
    return f"pkg:pypi/{name}@{version}"


def _extract_hashes(pkg: dict[str, Any]) -> list[dict[str, str]]:
    """Collect distinct sha256 hashes advertised by the package.

    uv.lock packages may carry a ``sdist.hash`` and one or more
    ``wheels[].hash`` entries, each typically of the form ``sha256:<hex>``.
    We normalize to CycloneDX 1.5 hash records ``{"alg": "SHA-256", "content": hex}``
    and de-duplicate while preserving first-seen order for determinism.
    """
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def _add(raw: str | None) -> None:
        if not raw:
            return
        # Accept both "sha256:xxx" and bare hex strings.
        if ":" in raw:
            alg, _, content = raw.partition(":")
            if alg.lower() != "sha256":
                return
        else:
            content = raw
        content = content.strip().lower()
        if not content or content in seen:
            return
        seen.add(content)
        out.append({"alg": "SHA-256", "content": content})

    sdist = pkg.get("sdist")
    if isinstance(sdist, dict):
        _add(sdist.get("hash"))

    wheels = pkg.get("wheels")
    if isinstance(wheels, list):
        for w in wheels:
            if isinstance(w, dict):
                _add(w.get("hash"))

    # Some lockfiles put a top-level "hash" on url-sourced packages.
    _add(pkg.get("hash"))

    return out


def _validate_package(pkg: dict[str, Any]) -> tuple[str, str]:
    name = pkg.get("name")
    version = pkg.get("version")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Malformed lockfile: package missing name: {pkg!r}")
    if not isinstance(version, str) or not version:
        raise ValueError(
            f"Malformed lockfile: package {name!r} missing version"
        )
    return name, version


def _build_component(pkg: dict[str, Any]) -> dict[str, Any]:
    name, version = _validate_package(pkg)
    purl = _purl(name, version)
    return {
        "type": "library",
        "bom-ref": purl,
        "name": name,
        "version": version,
        "purl": purl,
        "hashes": _extract_hashes(pkg),
    }


# ── public API ───────────────────────────────────────────────────────────────


def generate_sbom(
    lockfile_path: Path = Path("uv.lock"),
    output_path: Path = Path(".aurelius-sbom.json"),
) -> SBOMResult:
    """Parse a uv lockfile and emit a CycloneDX 1.5 SBOM.

    Args:
        lockfile_path: Path to ``uv.lock``.
        output_path: Where to write the JSON SBOM.

    Returns:
        :class:`SBOMResult` with component count, output path, and the
        sha256 digest of the emitted bytes.

    Raises:
        FileNotFoundError: The lockfile does not exist.
        ValueError: The lockfile is not valid TOML or a package is
            missing required name/version fields.
    """
    lockfile_path = Path(lockfile_path)
    output_path = Path(output_path)

    if not lockfile_path.exists():
        raise FileNotFoundError(f"Lockfile not found: {lockfile_path}")

    try:
        with lockfile_path.open("rb") as fh:
            parsed = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Malformed lockfile (invalid TOML): {exc}") from exc

    packages_raw = parsed.get("package", [])
    if not isinstance(packages_raw, list):
        raise ValueError("Malformed lockfile: [[package]] must be a list")

    components = [_build_component(pkg) for pkg in packages_raw]
    # Deterministic alphabetical ordering by name, then version.
    components.sort(key=lambda c: (c["name"], c["version"]))

    bom = {
        "bomFormat": SBOM_BOM_FORMAT,
        "specVersion": SBOM_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": _iso_utc_now(),
            "tools": [
                {
                    "vendor": SBOM_TOOL_VENDOR,
                    "name": SBOM_TOOL_NAME,
                    "version": SBOM_TOOL_VERSION,
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": f"pkg:generic/{SBOM_ROOT_COMPONENT_NAME}@{SBOM_ROOT_COMPONENT_VERSION}",
                "name": SBOM_ROOT_COMPONENT_NAME,
                "version": SBOM_ROOT_COMPONENT_VERSION,
            },
        },
        "components": components,
    }

    # Serialize deterministically (sorted keys, stable separators) — the
    # only non-deterministic fields are serialNumber + timestamp by design.
    payload = json.dumps(bom, indent=2, sort_keys=True).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    return SBOMResult(
        component_count=len(components),
        output_path=output_path,
        sha256_of_sbom=digest,
    )


# ── CLI entrypoint ───────────────────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.security.sbom_generator",
        description="Generate a CycloneDX 1.5 SBOM from uv.lock.",
    )
    parser.add_argument(
        "--lockfile",
        default="uv.lock",
        help="Path to uv.lock (default: uv.lock)",
    )
    parser.add_argument(
        "--output",
        default=".aurelius-sbom.json",
        help="Path to write the SBOM JSON (default: .aurelius-sbom.json)",
    )
    args = parser.parse_args(argv)

    try:
        result = generate_sbom(
            lockfile_path=Path(args.lockfile),
            output_path=Path(args.output),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"CycloneDX {SBOM_SPEC_VERSION} SBOM written to {result.output_path} "
        f"({result.component_count} components, sha256={result.sha256_of_sbom})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(_cli())
