"""Integration test: SBOM generator against the real uv.lock.

Supply-chain hardening: CycloneDX 1.5 SBOM per v8 release gate.
Finding AUR-SEC-2026-0022; CWE-1104 (use of unmaintained third-party
components) defense.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.security.sbom_generator import SBOM_SPEC_VERSION, generate_sbom


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LOCKFILE = REPO_ROOT / "uv.lock"


@pytest.mark.skipif(not REAL_LOCKFILE.exists(), reason="uv.lock missing in repo")
def test_real_uv_lock_produces_valid_sbom(tmp_path: Path) -> None:
    out = tmp_path / ".aurelius-sbom.json"
    result = generate_sbom(lockfile_path=REAL_LOCKFILE, output_path=out)

    # File exists and parses as JSON.
    assert out.exists()
    data = json.loads(out.read_text())

    # CycloneDX 1.5 basic envelope.
    assert data["bomFormat"] == "CycloneDX"
    assert data["specVersion"] == SBOM_SPEC_VERSION
    assert data["metadata"]["component"]["name"].lower().startswith("aurelius")

    # > 50 components for the real lockfile.
    components = data["components"]
    assert result.component_count == len(components)
    assert len(components) > 50, f"expected >50 components, got {len(components)}"

    # Each component carries the required fields.
    required = {"type", "bom-ref", "name", "version", "purl", "hashes"}
    for comp in components:
        missing = required - set(comp)
        assert not missing, f"component {comp.get('name')!r} missing {missing}"
        assert comp["type"] == "library"
        assert comp["purl"].startswith("pkg:pypi/")
        assert comp["purl"].endswith(f"@{comp['version']}")

    # Deterministic ordering.
    names = [c["name"] for c in components]
    assert names == sorted(names)


@pytest.mark.skipif(not REAL_LOCKFILE.exists(), reason="uv.lock missing in repo")
def test_real_uv_lock_round_trip_is_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "one.json"
    out2 = tmp_path / "two.json"
    generate_sbom(lockfile_path=REAL_LOCKFILE, output_path=out1)
    generate_sbom(lockfile_path=REAL_LOCKFILE, output_path=out2)
    d1 = json.loads(out1.read_text())
    d2 = json.loads(out2.read_text())
    # Ignore fields that vary by design.
    for d in (d1, d2):
        d["serialNumber"] = "X"
        d["metadata"]["timestamp"] = "X"
    assert d1 == d2
