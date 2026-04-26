"""Container image build abstractions for Aurelius deployment.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ContainerSpec:
    """Specification for building a container image."""

    image_name: str
    base_image: str = "python:3.14-slim"
    python_version: str = "3.14"
    extra_packages: list[str] = field(default_factory=list)
    expose_ports: list[int] = field(default_factory=lambda: [8080])
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class BuildResult:
    """Result of a container build operation."""

    success: bool
    image_tag: str
    sbom_path: str | None
    log_lines: list[str] = field(default_factory=list)


class ContainerBuilder(ABC):
    """Abstract base for container image builders."""

    @abstractmethod
    def build(self, spec: ContainerSpec, output_dir: Path) -> BuildResult:
        """Build (or generate) a container image from the given spec.

        Args:
            spec: The container specification.
            output_dir: Directory where build artifacts are written.

        Returns:
            BuildResult describing the outcome.
        """


def _render_dockerfile(spec: ContainerSpec) -> str:
    """Render a Dockerfile string from a ContainerSpec."""
    lines: list[str] = [
        f"FROM {spec.base_image}",
        "WORKDIR /app",
        "RUN pip install uv",
    ]

    if spec.extra_packages:
        pkgs = " ".join(spec.extra_packages)
        lines.append(f"RUN pip install {pkgs}")

    lines += [
        "COPY pyproject.toml .",
        "COPY src/ src/",
        "RUN uv sync --no-dev",
    ]

    for port in spec.expose_ports:
        lines.append(f"EXPOSE {port}")

    for key, value in spec.labels.items():
        lines.append(f'LABEL {key}="{value}"')

    lines.append('ENTRYPOINT ["python", "-m", "src.serving"]')
    return "\n".join(lines) + "\n"


class DockerfileBuilder(ContainerBuilder):
    """Generates a Dockerfile from a ContainerSpec without invoking docker CLI."""

    def build(self, spec: ContainerSpec, output_dir: Path) -> BuildResult:
        """Write a Dockerfile to output_dir and return a BuildResult.

        Does NOT invoke docker CLI — safe for test environments.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dockerfile_content = _render_dockerfile(spec)
        dockerfile_path = output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

        log_lines = [
            f"Rendered Dockerfile to {dockerfile_path}",
            f"Base image: {spec.base_image}",
            f"Ports: {spec.expose_ports}",
        ]

        return BuildResult(
            success=True,
            image_tag=f"{spec.image_name}:latest",
            sbom_path=None,
            log_lines=log_lines,
        )


class SBOMGenerator:
    """Generates a minimal SPDX-JSON SBOM from pyproject.toml if present."""

    def generate(self, output_dir: Path) -> Path:
        """Write sbom.json to output_dir and return its path.

        Reads pyproject.toml from the output_dir (or its parents) if present;
        otherwise returns a stub SBOM.

        Raises:
            OSError: If sbom.json cannot be written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        packages = self._collect_packages(output_dir)

        sbom = {
            "format": "spdx-json",
            "spdxVersion": "SPDX-2.3",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "aurelius-sbom",
            "dataLicense": "CC0-1.0",
            "documentNamespace": "https://github.com/aurelius/sbom",
            "packages": packages,
        }

        sbom_path = output_dir / "sbom.json"
        sbom_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
        return sbom_path

    def _collect_packages(self, output_dir: Path) -> list[dict]:
        """Attempt to read pyproject.toml to populate package list."""
        # Search output_dir and up to 3 parents for pyproject.toml
        search_dirs = [output_dir] + list(output_dir.parents)[:3]
        pyproject_path: Path | None = None
        for d in search_dirs:
            candidate = d / "pyproject.toml"
            if candidate.exists():
                pyproject_path = candidate
                break

        if pyproject_path is None:
            return [
                {
                    "SPDXID": "SPDXRef-Package-aurelius",
                    "name": "aurelius",
                    "versionInfo": "0.1.0",
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                }
            ]

        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                # Cannot parse toml; return stub
                return [
                    {
                        "SPDXID": "SPDXRef-Package-aurelius",
                        "name": "aurelius",
                        "versionInfo": "unknown",
                        "downloadLocation": "NOASSERTION",
                        "filesAnalyzed": False,
                    }
                ]

        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        name = project.get("name", "aurelius")
        version = project.get("version", "0.1.0")
        deps = project.get("dependencies", [])

        packages = [
            {
                "SPDXID": f"SPDXRef-Package-{name}",
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
            }
        ]

        for dep in deps:
            # dep is like "torch>=2.3.0" — extract name only
            dep_name = dep.split(">=")[0].split("==")[0].split("!=")[0].split("<")[0].strip()
            packages.append(
                {
                    "SPDXID": f"SPDXRef-Package-{dep_name}",
                    "name": dep_name,
                    "versionInfo": "NOASSERTION",
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                }
            )

        return packages


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ARTIFACT_BUILDER_REGISTRY: dict[str, type[ContainerBuilder]] = {
    "dockerfile": DockerfileBuilder,
}
