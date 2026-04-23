"""Integration tests for src.deployment.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.deployment as deployment_module
from src.deployment import ARTIFACT_BUILDER_REGISTRY, DEPLOY_TARGET_REGISTRY
from src.deployment.container_builder import (
    ContainerSpec,
    DockerfileBuilder,
    SBOMGenerator,
)


# ---------------------------------------------------------------------------
# Registry smoke tests
# ---------------------------------------------------------------------------


def test_deploy_target_registry_is_populated() -> None:
    """DEPLOY_TARGET_REGISTRY imported from src.deployment is a non-empty dict."""
    assert isinstance(DEPLOY_TARGET_REGISTRY, dict)
    assert len(DEPLOY_TARGET_REGISTRY) >= 1


def test_artifact_builder_registry_is_populated() -> None:
    """ARTIFACT_BUILDER_REGISTRY imported from src.deployment is a non-empty dict."""
    assert isinstance(ARTIFACT_BUILDER_REGISTRY, dict)
    assert len(ARTIFACT_BUILDER_REGISTRY) >= 1


def test_deploy_target_registry_required_keys() -> None:
    """DEPLOY_TARGET_REGISTRY contains docker, k8s, compose keys."""
    for key in ("docker", "k8s", "compose"):
        assert key in DEPLOY_TARGET_REGISTRY, f"Missing key: {key}"


def test_artifact_builder_registry_has_dockerfile() -> None:
    """ARTIFACT_BUILDER_REGISTRY contains 'dockerfile' key."""
    assert "dockerfile" in ARTIFACT_BUILDER_REGISTRY


# ---------------------------------------------------------------------------
# DockerfileBuilder full workflow
# ---------------------------------------------------------------------------


def test_dockerfile_builder_full_workflow(tmp_path: Path) -> None:
    """DockerfileBuilder generates a valid Dockerfile in a temp directory."""
    spec = ContainerSpec(
        image_name="aurelius-integration",
        base_image="python:3.14-slim",
        expose_ports=[8080],
        labels={"maintainer": "aurelius-ci"},
    )
    builder = DockerfileBuilder()
    result = builder.build(spec, tmp_path)

    assert result.success is True
    dockerfile = tmp_path / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not created"

    content = dockerfile.read_text()
    assert "FROM python:3.14-slim" in content
    assert "EXPOSE 8080" in content
    assert "src.serving" in content


# ---------------------------------------------------------------------------
# SBOMGenerator full workflow
# ---------------------------------------------------------------------------


def test_sbom_generator_full_workflow(tmp_path: Path) -> None:
    """SBOMGenerator produces a valid spdx-json SBOM in a temp directory."""
    gen = SBOMGenerator()
    sbom_path = gen.generate(tmp_path)

    assert sbom_path.exists(), "sbom.json not created"
    data = json.loads(sbom_path.read_text())
    assert data["format"] == "spdx-json"
    assert data["spdxVersion"] == "SPDX-2.3"
    assert isinstance(data["packages"], list)
    assert len(data["packages"]) >= 1


# ---------------------------------------------------------------------------
# Combined workflow: Dockerfile + SBOM
# ---------------------------------------------------------------------------


def test_dockerfile_and_sbom_both_exist(tmp_path: Path) -> None:
    """Both Dockerfile and sbom.json are produced in the same output directory."""
    spec = ContainerSpec(image_name="aurelius-combined", expose_ports=[8080])
    builder = DockerfileBuilder()
    gen = SBOMGenerator()

    builder.build(spec, tmp_path)
    gen.generate(tmp_path)

    assert (tmp_path / "Dockerfile").exists()
    assert (tmp_path / "sbom.json").exists()
