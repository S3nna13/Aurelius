"""
deployment_manifest.py — Deployment manifest builder and validator.
Aurelius LLM Project — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResourceSpec:
    cpu_millicores: int
    memory_mb: int
    gpu_count: int = 0


@dataclass
class ContainerSpec:
    image: str
    tag: str
    resources: ResourceSpec
    env_vars: dict = field(default_factory=dict)
    ports: list = field(default_factory=list)


class DeploymentManifest:
    """Builds and validates a deployment manifest."""

    def __init__(self, name: str, namespace: str = "default", replicas: int = 1) -> None:
        self.name = name
        self.namespace = namespace
        self.replicas = replicas
        self._containers: list[ContainerSpec] = []
        self._labels: dict[str, str] = {}

    def add_container(self, spec: ContainerSpec) -> None:
        """Append a container specification."""
        self._containers.append(spec)

    def set_label(self, key: str, value: str) -> None:
        """Set or overwrite a manifest label."""
        self._labels[key] = value

    def validate(self) -> list:
        """
        Validate the manifest. Returns a (possibly empty) list of error strings.

        Checks:
        - name not empty
        - replicas >= 1
        - at least one container
        - no container image empty
        - cpu_millicores > 0 for every container
        - memory_mb > 0 for every container
        """
        errors: list[str] = []

        if not self.name or not self.name.strip():
            errors.append("Manifest name must not be empty.")

        if self.replicas < 1:
            errors.append("replicas must be >= 1.")

        if not self._containers:
            errors.append("At least one container must be specified.")

        for i, c in enumerate(self._containers):
            if not c.image or not c.image.strip():
                errors.append(f"Container[{i}] image must not be empty.")
            if c.resources.cpu_millicores <= 0:
                errors.append(
                    f"Container[{i}] cpu_millicores must be > 0 (got {c.resources.cpu_millicores})."
                )
            if c.resources.memory_mb <= 0:
                errors.append(
                    f"Container[{i}] memory_mb must be > 0 (got {c.resources.memory_mb})."
                )

        return errors

    def to_dict(self) -> dict:
        """Serialise the manifest to a plain dict."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "labels": dict(self._labels),
            "containers": [
                {
                    "image": c.image,
                    "tag": c.tag,
                    "resources": {
                        "cpu_millicores": c.resources.cpu_millicores,
                        "memory_mb": c.resources.memory_mb,
                        "gpu_count": c.resources.gpu_count,
                    },
                    "env_vars": dict(c.env_vars),
                    "ports": list(c.ports),
                }
                for c in self._containers
            ],
        }

    def to_yaml_stub(self) -> str:
        """
        Return a minimal YAML-like text representation.
        Uses only stdlib string formatting — no external deps.
        """
        lines: list[str] = [
            f"name: {self.name}",
            f"namespace: {self.namespace}",
            f"replicas: {self.replicas}",
        ]

        if self._labels:
            lines.append("labels:")
            for k, v in sorted(self._labels.items()):
                lines.append(f"  {k}: {v}")

        if self._containers:
            lines.append("containers:")
            for c in self._containers:
                lines.append(f"  - image: {c.image}:{c.tag}")
                lines.append(f"    cpu_millicores: {c.resources.cpu_millicores}")
                lines.append(f"    memory_mb: {c.resources.memory_mb}")
                if c.resources.gpu_count:
                    lines.append(f"    gpu_count: {c.resources.gpu_count}")
                if c.env_vars:
                    lines.append("    env_vars:")
                    for ek, ev in sorted(c.env_vars.items()):
                        lines.append(f"      {ek}: {ev}")
                if c.ports:
                    lines.append("    ports:")
                    for p in c.ports:
                        lines.append(f"      - {p}")

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------
DEPLOYMENT_MANIFEST_REGISTRY: dict = {"default": DeploymentManifest}

REGISTRY = DEPLOYMENT_MANIFEST_REGISTRY
