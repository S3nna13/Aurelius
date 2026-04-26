"""Tests for CycloneDX 1.5 SBOM generator.

Supply-chain hardening: CycloneDX 1.5 SBOM per v8 release gate.
Finding AUR-SEC-2026-0022; CWE-1104 (use of unmaintained third-party
components) defense.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.security.sbom_generator import (
    SBOM_SPEC_VERSION,
    SBOMResult,
    generate_sbom,
)

EMPTY_LOCK = """\
version = 1
revision = 3
requires-python = ">=3.12"
"""

SINGLE_PACKAGE_LOCK = """\
version = 1
revision = 3
requires-python = ">=3.12"

[[package]]
name = "absl-py"
version = "2.4.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://example/absl.tar.gz", hash = "sha256:deadbeef", size = 1 }
wheels = [
    { url = "https://example/absl.whl", hash = "sha256:cafebabe", size = 2 },
]
"""

GIT_SOURCE_LOCK = """\
version = 1
revision = 3

[[package]]
name = "mylib"
version = "0.1.0"
source = { git = "https://github.com/example/mylib.git?rev=abc123#abc123" }
"""

URL_SOURCE_LOCK = """\
version = 1
revision = 3

[[package]]
name = "weirdpkg"
version = "1.2.3"
source = { url = "https://example.com/weirdpkg-1.2.3.tar.gz" }
sdist = { url = "https://example.com/weirdpkg-1.2.3.tar.gz", hash = "sha256:feed0000", size = 9 }
"""

MULTI_PACKAGE_LOCK = """\
version = 1
revision = 3

[[package]]
name = "zeta"
version = "9.0.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://e/z.tgz", hash = "sha256:11", size = 1 }

[[package]]
name = "alpha"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://e/a.tgz", hash = "sha256:22", size = 1 }

[[package]]
name = "mid"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://e/m.tgz", hash = "sha256:33", size = 1 }
"""


@pytest.fixture
def tmp_lock(tmp_path: Path):
    def _write(content: str, name: str = "uv.lock") -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    return _write


def _run(tmp_lock_factory, content: str, tmp_path: Path) -> tuple[SBOMResult, dict]:
    lock = tmp_lock_factory(content)
    out = tmp_path / "out.json"
    result = generate_sbom(lockfile_path=lock, output_path=out)
    data = json.loads(out.read_text())
    return result, data


def test_empty_lockfile_produces_zero_components(tmp_lock, tmp_path):
    result, data = _run(tmp_lock, EMPTY_LOCK, tmp_path)
    assert result.component_count == 0
    assert data["components"] == []


def test_single_package_registry_source(tmp_lock, tmp_path):
    result, data = _run(tmp_lock, SINGLE_PACKAGE_LOCK, tmp_path)
    assert result.component_count == 1
    comp = data["components"][0]
    assert comp["name"] == "absl-py"
    assert comp["version"] == "2.4.0"
    assert comp["type"] == "library"
    assert comp["purl"] == "pkg:pypi/absl-py@2.4.0"
    # sha256 from sdist recorded
    assert any(h["alg"] == "SHA-256" and h["content"] == "deadbeef" for h in comp["hashes"])


def test_package_with_git_source(tmp_lock, tmp_path):
    result, data = _run(tmp_lock, GIT_SOURCE_LOCK, tmp_path)
    assert result.component_count == 1
    comp = data["components"][0]
    assert comp["name"] == "mylib"
    # git source still yields a purl, hashes may be empty
    assert comp["purl"] == "pkg:pypi/mylib@0.1.0"
    assert "hashes" in comp


def test_package_with_url_source(tmp_lock, tmp_path):
    result, data = _run(tmp_lock, URL_SOURCE_LOCK, tmp_path)
    assert result.component_count == 1
    comp = data["components"][0]
    assert comp["name"] == "weirdpkg"
    assert any(h["content"] == "feed0000" for h in comp["hashes"])


def test_deterministic_ordering_two_runs_same_hash(tmp_lock, tmp_path):
    lock = tmp_lock(MULTI_PACKAGE_LOCK)
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    r1 = generate_sbom(lockfile_path=lock, output_path=out1)
    r2 = generate_sbom(lockfile_path=lock, output_path=out2)
    # Strip serialNumber + timestamp (which vary) and compare the rest
    d1 = json.loads(out1.read_text())
    d2 = json.loads(out2.read_text())
    for d in (d1, d2):
        d["serialNumber"] = "X"
        d["metadata"]["timestamp"] = "X"
    assert d1 == d2
    # Components must be alphabetically sorted
    names = [c["name"] for c in d1["components"]]
    assert names == sorted(names)
    # Result sha256 reflects file contents
    assert r1.sha256_of_sbom == hashlib.sha256(out1.read_bytes()).hexdigest()
    assert r2.sha256_of_sbom == hashlib.sha256(out2.read_bytes()).hexdigest()


def test_cyclonedx_1_5_schema_compliance(tmp_lock, tmp_path):
    _, data = _run(tmp_lock, SINGLE_PACKAGE_LOCK, tmp_path)
    assert data["bomFormat"] == "CycloneDX"
    assert data["specVersion"] == SBOM_SPEC_VERSION == "1.5"
    assert data["version"] == 1
    assert "serialNumber" in data
    assert "metadata" in data
    assert "tools" in data["metadata"]
    assert "component" in data["metadata"]
    assert data["metadata"]["component"]["name"].lower().startswith("aurelius")


def test_purl_format_correctness(tmp_lock, tmp_path):
    _, data = _run(tmp_lock, MULTI_PACKAGE_LOCK, tmp_path)
    for comp in data["components"]:
        assert re.fullmatch(r"pkg:pypi/[A-Za-z0-9_.-]+@[A-Za-z0-9_.\-+]+", comp["purl"]), comp[
            "purl"
        ]


def test_serial_number_is_urn_uuid(tmp_lock, tmp_path):
    _, data = _run(tmp_lock, SINGLE_PACKAGE_LOCK, tmp_path)
    assert re.fullmatch(
        r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        data["serialNumber"],
    )


def test_timestamp_is_iso8601_z(tmp_lock, tmp_path):
    _, data = _run(tmp_lock, SINGLE_PACKAGE_LOCK, tmp_path)
    ts = data["metadata"]["timestamp"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z", ts), ts


def test_bom_ref_unique_per_component(tmp_lock, tmp_path):
    _, data = _run(tmp_lock, MULTI_PACKAGE_LOCK, tmp_path)
    refs = [c["bom-ref"] for c in data["components"]]
    assert len(refs) == len(set(refs))
    assert all(refs)


def test_malformed_toml_is_rejected(tmp_lock, tmp_path):
    lock = tmp_lock("this is not { valid toml at all = = =\n[[[")
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        generate_sbom(lockfile_path=lock, output_path=out)


def test_missing_lockfile_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_sbom(
            lockfile_path=tmp_path / "does_not_exist.lock",
            output_path=tmp_path / "out.json",
        )


def test_package_missing_name_or_version_is_rejected(tmp_lock, tmp_path):
    bad = """
[[package]]
version = "1.0"
source = { registry = "x" }
"""
    lock = tmp_lock(bad)
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        generate_sbom(lockfile_path=lock, output_path=out)


def test_sbom_result_fields(tmp_lock, tmp_path):
    lock = tmp_lock(SINGLE_PACKAGE_LOCK)
    out = tmp_path / "out.json"
    result = generate_sbom(lockfile_path=lock, output_path=out)
    assert isinstance(result, SBOMResult)
    assert result.output_path == out
    assert result.component_count == 1
    assert re.fullmatch(r"[0-9a-f]{64}", result.sha256_of_sbom)


def test_registered_in_security_registry():
    from src.security import SBOM_REGISTRY

    assert "default" in SBOM_REGISTRY
    assert callable(SBOM_REGISTRY["default"])
