"""Helm chart generator for Aurelius deployment.

Inspired by Helm (Apache-2.0, helm.sh), OpenTelemetry SDK (Apache-2.0, opentelemetry.io),
Prometheus text format (Apache-2.0), clean-room Aurelius implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HelmChartError(Exception):
    """Raised when Helm chart generation fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HelmChartValues:
    """Values for a Helm chart (mirrors values.yaml)."""

    image_repository: str = "ghcr.io/s3nna13/aurelius"
    image_tag: str = "latest"
    replicas: int = 1
    port: int = 8080
    resources_requests: dict = field(default_factory=lambda: {"cpu": "2", "memory": "8Gi"})
    resources_limits: dict = field(default_factory=lambda: {"cpu": "4", "memory": "16Gi"})
    env_vars: dict = field(default_factory=dict)
    service_type: str = "ClusterIP"
    ingress_enabled: bool = False
    gpu_enabled: bool = False


@dataclass
class HelmChart:
    """Metadata + values for a Helm chart."""

    name: str
    version: str
    app_version: str
    description: str
    values: HelmChartValues


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class HelmChartGenerator:
    """Generates Helm chart YAML files as strings — no helm CLI required."""

    def generate_chart_yaml(self, chart: HelmChart) -> str:
        """Return the content of Chart.yaml as a string."""
        return (
            "apiVersion: v2\n"
            f"name: {chart.name}\n"
            f"description: {chart.description}\n"
            "type: application\n"
            f"version: {chart.version}\n"
            f'appVersion: "{chart.app_version}"\n'
        )

    def generate_values_yaml(self, chart: HelmChart) -> str:
        """Return the content of values.yaml as a string."""
        v = chart.values

        # Build env_vars block
        env_lines: list[str] = []
        if v.env_vars:
            for k, val in v.env_vars.items():
                env_lines.append(f'  {k}: "{val}"')
        env_block = "\n".join(env_lines) if env_lines else "  {}"

        # Build resources blocks
        req_lines = "\n".join(f'    {k}: "{val}"' for k, val in v.resources_requests.items())
        lim_lines = "\n".join(f'    {k}: "{val}"' for k, val in v.resources_limits.items())

        return (
            f"replicaCount: {v.replicas}\n"
            "\n"
            "image:\n"
            f"  repository: {v.image_repository}\n"
            f'  tag: "{v.image_tag}"\n'
            "  pullPolicy: IfNotPresent\n"
            "\n"
            f"service:\n"
            f"  type: {v.service_type}\n"
            f"  port: {v.port}\n"
            "\n"
            "resources:\n"
            "  requests:\n"
            f"{req_lines}\n"
            "  limits:\n"
            f"{lim_lines}\n"
            "\n"
            "ingress:\n"
            f"  enabled: {'true' if v.ingress_enabled else 'false'}\n"
            "\n"
            "gpu:\n"
            f"  enabled: {'true' if v.gpu_enabled else 'false'}\n"
            "\n"
            "env:\n"
            f"{env_block}\n"
        )

    def generate_deployment_yaml(self, chart: HelmChart) -> str:
        """Return the content of templates/deployment.yaml as a string."""
        v = chart.values
        gpu_block = ""
        if v.gpu_enabled:
            gpu_block = (
                "          resources:\n            limits:\n              nvidia.com/gpu: 1\n"
            )

        return (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            f"  name: {chart.name}\n"
            "  labels:\n"
            f"    app: {chart.name}\n"
            "spec:\n"
            f"  replicas: {v.replicas}\n"
            "  selector:\n"
            "    matchLabels:\n"
            f"      app: {chart.name}\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            f"        app: {chart.name}\n"
            "    spec:\n"
            "      containers:\n"
            f"        - name: {chart.name}\n"
            f"          image: {v.image_repository}:{v.image_tag}\n"
            "          ports:\n"
            f"            - containerPort: {v.port}\n"
            "          resources:\n"
            "            requests:\n"
            + "\n".join(f"              {k}: {val}" for k, val in v.resources_requests.items())
            + "\n"
            "            limits:\n"
            + "\n".join(f"              {k}: {val}" for k, val in v.resources_limits.items())
            + "\n"
            + gpu_block
        )

    def generate_service_yaml(self, chart: HelmChart) -> str:
        """Return the content of templates/service.yaml as a string."""
        v = chart.values
        return (
            "apiVersion: v1\n"
            "kind: Service\n"
            "metadata:\n"
            f"  name: {chart.name}\n"
            "spec:\n"
            f"  type: {v.service_type}\n"
            "  selector:\n"
            f"    app: {chart.name}\n"
            "  ports:\n"
            f"    - port: {v.port}\n"
            f"      targetPort: {v.port}\n"
            "      protocol: TCP\n"
        )

    def write_chart(self, chart: HelmChart, output_dir: Path) -> dict[str, Path]:
        """Write all Helm chart files to output_dir/<chart.name>/.

        Creates:
          Chart.yaml
          values.yaml
          templates/deployment.yaml
          templates/service.yaml

        Returns:
            Mapping of filename (relative within chart dir) to absolute Path.

        Raises:
            HelmChartError: If writing any file fails.
        """
        output_dir = Path(output_dir)
        chart_dir = output_dir / chart.name
        templates_dir = chart_dir / "templates"

        try:
            chart_dir.mkdir(parents=True, exist_ok=True)
            templates_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HelmChartError(f"Cannot create chart directory: {exc}") from exc

        files: dict[str, Path] = {}

        try:
            p = chart_dir / "Chart.yaml"
            p.write_text(self.generate_chart_yaml(chart), encoding="utf-8")
            files["Chart.yaml"] = p

            p = chart_dir / "values.yaml"
            p.write_text(self.generate_values_yaml(chart), encoding="utf-8")
            files["values.yaml"] = p

            p = templates_dir / "deployment.yaml"
            p.write_text(self.generate_deployment_yaml(chart), encoding="utf-8")
            files["templates/deployment.yaml"] = p

            p = templates_dir / "service.yaml"
            p.write_text(self.generate_service_yaml(chart), encoding="utf-8")
            files["templates/service.yaml"] = p
        except OSError as exc:
            raise HelmChartError(f"Failed to write chart file: {exc}") from exc

        return files


# ---------------------------------------------------------------------------
# Registry & well-known chart
# ---------------------------------------------------------------------------

_GENERATOR = HelmChartGenerator()

_AURELIUS_CHART = HelmChart(
    name="aurelius",
    version="0.1.0",
    app_version="0.1.0",
    description="Aurelius LLM inference service",
    values=HelmChartValues(),
)

HELM_CHART_REGISTRY: dict[str, HelmChart] = {
    "aurelius": _AURELIUS_CHART,
}

# ---------------------------------------------------------------------------
# Register into ARTIFACT_BUILDER_REGISTRY
# ---------------------------------------------------------------------------

from src.deployment.container_builder import (  # noqa: E402
    ARTIFACT_BUILDER_REGISTRY,
    BuildResult,
    ContainerBuilder,
    ContainerSpec,
)


class _HelmArtifactBuilder(ContainerBuilder):
    """Adapter: registers Helm chart generation as an ARTIFACT_BUILDER_REGISTRY entry."""

    def build(self, spec: ContainerSpec, output_dir: Path) -> BuildResult:  # type: ignore[override]
        chart = HELM_CHART_REGISTRY.get("aurelius", _AURELIUS_CHART)
        # Override image name from spec
        chart_copy = HelmChart(
            name=chart.name,
            version=chart.version,
            app_version=chart.app_version,
            description=chart.description,
            values=HelmChartValues(image_repository=spec.image_name),
        )
        written = _GENERATOR.write_chart(chart_copy, Path(output_dir))
        return BuildResult(
            success=True,
            image_tag=f"{spec.image_name}:latest",
            sbom_path=None,
            log_lines=[f"Wrote Helm chart files: {list(written.keys())}"],
        )


ARTIFACT_BUILDER_REGISTRY["helm"] = _HelmArtifactBuilder
