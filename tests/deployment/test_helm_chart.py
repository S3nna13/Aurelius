"""Tests for src.deployment.helm_chart.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deployment.helm_chart import (
    HELM_CHART_REGISTRY,
    HelmChart,
    HelmChartError,
    HelmChartGenerator,
    HelmChartValues,
)


# ---------------------------------------------------------------------------
# HelmChartValues defaults
# ---------------------------------------------------------------------------


def test_helm_chart_values_default_replicas() -> None:
    v = HelmChartValues()
    assert v.replicas == 1


def test_helm_chart_values_default_port() -> None:
    v = HelmChartValues()
    assert v.port == 8080


def test_helm_chart_values_default_service_type() -> None:
    v = HelmChartValues()
    assert v.service_type == "ClusterIP"


def test_helm_chart_values_default_image_repository() -> None:
    v = HelmChartValues()
    assert v.image_repository == "ghcr.io/s3nna13/aurelius"


def test_helm_chart_values_default_image_tag() -> None:
    v = HelmChartValues()
    assert v.image_tag == "latest"


def test_helm_chart_values_default_gpu_disabled() -> None:
    v = HelmChartValues()
    assert v.gpu_enabled is False


def test_helm_chart_values_default_ingress_disabled() -> None:
    v = HelmChartValues()
    assert v.ingress_enabled is False


def test_helm_chart_values_default_resources_requests() -> None:
    v = HelmChartValues()
    assert v.resources_requests == {"cpu": "2", "memory": "8Gi"}


def test_helm_chart_values_default_resources_limits() -> None:
    v = HelmChartValues()
    assert v.resources_limits == {"cpu": "4", "memory": "16Gi"}


# ---------------------------------------------------------------------------
# HelmChartGenerator — chart.yaml
# ---------------------------------------------------------------------------


def _sample_chart() -> HelmChart:
    return HelmChart(
        name="aurelius",
        version="1.2.3",
        app_version="0.9.0",
        description="Test chart",
        values=HelmChartValues(),
    )


def test_generate_chart_yaml_has_api_version() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_chart_yaml(_sample_chart())
    assert "apiVersion: v2" in out


def test_generate_chart_yaml_has_name() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_chart_yaml(_sample_chart())
    assert "aurelius" in out


def test_generate_chart_yaml_has_version() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_chart_yaml(_sample_chart())
    assert "1.2.3" in out


# ---------------------------------------------------------------------------
# HelmChartGenerator — values.yaml
# ---------------------------------------------------------------------------


def test_generate_values_yaml_has_image_key() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_values_yaml(_sample_chart())
    assert "image:" in out


def test_generate_values_yaml_has_replicas() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_values_yaml(_sample_chart())
    assert "replicas" in out or "replicaCount" in out


# ---------------------------------------------------------------------------
# HelmChartGenerator — deployment.yaml
# ---------------------------------------------------------------------------


def test_generate_deployment_yaml_has_kind() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_deployment_yaml(_sample_chart())
    assert "kind: Deployment" in out


def test_generate_deployment_yaml_has_chart_name() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_deployment_yaml(_sample_chart())
    assert "aurelius" in out


# ---------------------------------------------------------------------------
# HelmChartGenerator — service.yaml
# ---------------------------------------------------------------------------


def test_generate_service_yaml_has_kind() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_service_yaml(_sample_chart())
    assert "kind: Service" in out


def test_generate_service_yaml_has_chart_name() -> None:
    gen = HelmChartGenerator()
    out = gen.generate_service_yaml(_sample_chart())
    assert "aurelius" in out


# ---------------------------------------------------------------------------
# write_chart
# ---------------------------------------------------------------------------


def test_write_chart_creates_all_files(tmp_path: Path) -> None:
    gen = HelmChartGenerator()
    chart = _sample_chart()
    written = gen.write_chart(chart, tmp_path)

    assert "Chart.yaml" in written
    assert "values.yaml" in written
    assert "templates/deployment.yaml" in written
    assert "templates/service.yaml" in written


def test_write_chart_files_exist_on_disk(tmp_path: Path) -> None:
    gen = HelmChartGenerator()
    written = gen.write_chart(_sample_chart(), tmp_path)
    for path in written.values():
        assert path.exists(), f"Missing file: {path}"


def test_write_chart_chart_yaml_content(tmp_path: Path) -> None:
    gen = HelmChartGenerator()
    written = gen.write_chart(_sample_chart(), tmp_path)
    content = written["Chart.yaml"].read_text()
    assert "apiVersion: v2" in content


def test_write_chart_values_yaml_content(tmp_path: Path) -> None:
    gen = HelmChartGenerator()
    written = gen.write_chart(_sample_chart(), tmp_path)
    content = written["values.yaml"].read_text()
    assert "image:" in content


# ---------------------------------------------------------------------------
# Registry and error class
# ---------------------------------------------------------------------------


def test_helm_chart_registry_is_dict() -> None:
    assert isinstance(HELM_CHART_REGISTRY, dict)


def test_helm_chart_registry_contains_aurelius() -> None:
    assert "aurelius" in HELM_CHART_REGISTRY


def test_helm_chart_error_is_exception_subclass() -> None:
    assert issubclass(HelmChartError, Exception)


def test_helm_chart_error_can_be_raised() -> None:
    with pytest.raises(HelmChartError):
        raise HelmChartError("boom")


def test_helm_chart_values_gpu_defaults_false() -> None:
    chart = _sample_chart()
    assert chart.values.gpu_enabled is False
