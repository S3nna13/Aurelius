"""Aurelius deployment surface — container build, health endpoints, SBOM.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

from src.deployment.container_builder import ARTIFACT_BUILDER_REGISTRY
from src.deployment.healthz import DEPLOY_TARGET_REGISTRY
from src.deployment.helm_chart import (
    HELM_CHART_REGISTRY,
    HelmChart,
    HelmChartError,
    HelmChartGenerator,
)
from src.deployment.otel_instrumentation import (
    DEFAULT_METRICS,
    DEFAULT_TRACER,
    OTEL_TRACER_REGISTRY,
    PrometheusMetrics,
    Tracer,
)

__all__ = [
    "ARTIFACT_BUILDER_REGISTRY",
    "DEFAULT_METRICS",
    "DEFAULT_TRACER",
    "DEPLOY_TARGET_REGISTRY",
    "HELM_CHART_REGISTRY",
    "HelmChart",
    "HelmChartError",
    "HelmChartGenerator",
    "OTEL_TRACER_REGISTRY",
    "PrometheusMetrics",
    "Tracer",
]

# --- Additive registration for GitHub Actions CI YAML generator --------------
# Additive only; reuses ARTIFACT_BUILDER_REGISTRY above and leaves all prior
# entries untouched.
from src.deployment.github_actions import (
    GHActionsGenerator as _GHActionsGenerator,
    GHActionsWorkflow as _GHActionsWorkflow,
    GHActionsError as _GHActionsError,
    GHActionsTrigger as _GHActionsTrigger,
    GHActionsJob as _GHActionsJob,
)

GHActionsGenerator = _GHActionsGenerator
GHActionsWorkflow = _GHActionsWorkflow
GHActionsError = _GHActionsError
GHActionsTrigger = _GHActionsTrigger
GHActionsJob = _GHActionsJob

ARTIFACT_BUILDER_REGISTRY.setdefault("github_actions", _GHActionsGenerator)

__all__ = list(__all__) + [
    "GHActionsGenerator",
    "GHActionsWorkflow",
    "GHActionsError",
    "GHActionsTrigger",
    "GHActionsJob",
]
