"""Tests for src.deployment.container_builder.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.deployment.container_builder import (
    ARTIFACT_BUILDER_REGISTRY,
    BuildResult,
    ContainerSpec,
    DockerfileBuilder,
    SBOMGenerator,
)


# ---------------------------------------------------------------------------
# DockerfileBuilder tests
# ---------------------------------------------------------------------------


def test_dockerfile_builder_writes_file(tmp_path: Path) -> None:
    """DockerfileBuilder.build writes a Dockerfile to the output directory."""
    spec = ContainerSpec(image_name="aurelius-test")
    builder = DockerfileBuilder()
    result = builder.build(spec, tmp_path)

    assert result.success is True
    dockerfile = tmp_path / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile was not written"


def test_dockerfile_contains_base_image(tmp_path: Path) -> None:
    """Generated Dockerfile starts with the correct FROM line."""
    spec = ContainerSpec(image_name="aurelius-test", base_image="python:3.14-slim")
    builder = DockerfileBuilder()
    builder.build(spec, tmp_path)

    content = (tmp_path / "Dockerfile").read_text()
    assert "FROM python:3.14-slim" in content


def test_dockerfile_contains_expose(tmp_path: Path) -> None:
    """Generated Dockerfile exposes port 8080."""
    spec = ContainerSpec(image_name="aurelius-test", expose_ports=[8080])
    builder = DockerfileBuilder()
    builder.build(spec, tmp_path)

    content = (tmp_path / "Dockerfile").read_text()
    assert "EXPOSE 8080" in content


def test_dockerfile_contains_entrypoint(tmp_path: Path) -> None:
    """Generated Dockerfile sets the correct ENTRYPOINT."""
    spec = ContainerSpec(image_name="aurelius-test")
    builder = DockerfileBuilder()
    builder.build(spec, tmp_path)

    content = (tmp_path / "Dockerfile").read_text()
    assert "src.serving" in content


def test_dockerfile_build_result_image_tag(tmp_path: Path) -> None:
    """BuildResult.image_tag is set to '<image_name>:latest'."""
    spec = ContainerSpec(image_name="my-image")
    builder = DockerfileBuilder()
    result = builder.build(spec, tmp_path)

    assert result.image_tag == "my-image:latest"


# ---------------------------------------------------------------------------
# SBOMGenerator tests
# ---------------------------------------------------------------------------


def test_sbom_generator_writes_file(tmp_path: Path) -> None:
    """SBOMGenerator.generate writes sbom.json to output_dir."""
    gen = SBOMGenerator()
    sbom_path = gen.generate(tmp_path)

    assert sbom_path.exists()
    assert sbom_path.name == "sbom.json"


def test_sbom_format_field(tmp_path: Path) -> None:
    """sbom.json contains 'format': 'spdx-json'."""
    gen = SBOMGenerator()
    sbom_path = gen.generate(tmp_path)

    data = json.loads(sbom_path.read_text())
    assert data["format"] == "spdx-json"


def test_sbom_spdx_version(tmp_path: Path) -> None:
    """sbom.json contains the correct spdxVersion."""
    gen = SBOMGenerator()
    sbom_path = gen.generate(tmp_path)

    data = json.loads(sbom_path.read_text())
    assert data["spdxVersion"] == "SPDX-2.3"


def test_sbom_packages_list(tmp_path: Path) -> None:
    """sbom.json packages field is a non-empty list."""
    gen = SBOMGenerator()
    sbom_path = gen.generate(tmp_path)

    data = json.loads(sbom_path.read_text())
    assert isinstance(data["packages"], list)
    assert len(data["packages"]) >= 1


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_artifact_builder_registry_contains_dockerfile() -> None:
    """ARTIFACT_BUILDER_REGISTRY has a 'dockerfile' entry."""
    assert "dockerfile" in ARTIFACT_BUILDER_REGISTRY


def test_artifact_builder_registry_value_is_class() -> None:
    """ARTIFACT_BUILDER_REGISTRY['dockerfile'] is a ContainerBuilder subclass."""
    from src.deployment.container_builder import ContainerBuilder

    cls = ARTIFACT_BUILDER_REGISTRY["dockerfile"]
    assert issubclass(cls, ContainerBuilder)
