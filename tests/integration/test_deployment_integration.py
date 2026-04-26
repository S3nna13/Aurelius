"""Integration tests for src.deployment.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.

Includes integration tests for GitHub Actions CI YAML generator (additive).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.deployment import ARTIFACT_BUILDER_REGISTRY, DEPLOY_TARGET_REGISTRY
from src.deployment.container_builder import (
    ContainerSpec,
    DockerfileBuilder,
    SBOMGenerator,
)
from src.deployment.helm_chart import HelmChart, HelmChartGenerator, HelmChartValues
from src.deployment.otel_instrumentation import Tracer

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


# ---------------------------------------------------------------------------
# Helm chart integration
# ---------------------------------------------------------------------------


def test_helm_chart_write(tmp_path: Path) -> None:
    """generate_chart_yaml + write_chart produce Chart.yaml and values.yaml on disk."""
    chart = HelmChart(
        name="aurelius-integration",
        version="0.1.0",
        app_version="0.1.0",
        description="Integration test chart",
        values=HelmChartValues(),
    )
    gen = HelmChartGenerator()
    written = gen.write_chart(chart, tmp_path)

    chart_yaml = written["Chart.yaml"]
    values_yaml = written["values.yaml"]

    assert chart_yaml.exists(), "Chart.yaml not written"
    assert values_yaml.exists(), "values.yaml not written"

    chart_content = chart_yaml.read_text()
    assert "apiVersion: v2" in chart_content
    assert "aurelius-integration" in chart_content

    values_content = values_yaml.read_text()
    assert "image:" in values_content


# ---------------------------------------------------------------------------
# OTel tracer lifecycle integration
# ---------------------------------------------------------------------------


def test_otel_tracer_lifecycle() -> None:
    """Start a span, add an attribute, end it, verify it appears in get_spans()."""
    tracer = Tracer()

    span = tracer.start_span("inference-request", attributes={"model": "aurelius-1.3b"})
    span.attributes["tokens_generated"] = 256
    tracer.end_span(span)

    spans = tracer.get_spans()
    assert len(spans) == 1

    recorded = spans[0]
    assert recorded.name == "inference-request"
    assert recorded.attributes["model"] == "aurelius-1.3b"
    assert recorded.attributes["tokens_generated"] == 256
    assert recorded.end_ns is not None
    assert recorded.end_ns >= recorded.start_ns


# ---------------------------------------------------------------------------
# GitHub Actions CI generator integration
# ---------------------------------------------------------------------------


def test_github_actions_generate(tmp_path: Path) -> None:
    """Generate a CI workflow, write to tempdir, verify .github/workflows/ path."""
    from src.deployment.github_actions import GHActionsGenerator, GHActionsWorkflow

    gen = GHActionsGenerator()
    workflow: GHActionsWorkflow = gen.default_ci_workflow(python_version="3.14")

    out_path = gen.write_workflow(workflow, tmp_path)

    # Verify it landed under .github/workflows/
    assert out_path.parent.name == "workflows"
    assert out_path.parent.parent.name == ".github"
    assert out_path.suffix == ".yml"
    assert out_path.exists()

    content = out_path.read_text()
    assert "actions/checkout" in content
    assert "actions/setup-python" in content
    assert "pytest" in content
    assert "3.14" in content
