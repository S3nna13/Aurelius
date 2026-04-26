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
    GHActionsError as _GHActionsError,
)
from src.deployment.github_actions import (
    GHActionsGenerator as _GHActionsGenerator,
)
from src.deployment.github_actions import (
    GHActionsJob as _GHActionsJob,
)
from src.deployment.github_actions import (
    GHActionsTrigger as _GHActionsTrigger,
)
from src.deployment.github_actions import (
    GHActionsWorkflow as _GHActionsWorkflow,
)

GHActionsGenerator = _GHActionsGenerator
GHActionsWorkflow = _GHActionsWorkflow
GHActionsError = _GHActionsError
GHActionsTrigger = _GHActionsTrigger
GHActionsJob = _GHActionsJob

ARTIFACT_BUILDER_REGISTRY.setdefault("github_actions", _GHActionsGenerator)  # type: ignore[arg-type]

__all__ = list(__all__) + [
    "GHActionsGenerator",
    "GHActionsWorkflow",
    "GHActionsError",
    "GHActionsTrigger",
    "GHActionsJob",
]

# --- Rollout manager ----------------------------------------------------------
from src.deployment.rollout_manager import (
    CANARY_3_STAGE as _CANARY_3_STAGE,
)
from src.deployment.rollout_manager import (
    ROLLOUT_REGISTRY as _ROLLOUT_REGISTRY,
)
from src.deployment.rollout_manager import (
    RolloutManager as _RolloutManager,
)
from src.deployment.rollout_manager import (
    RolloutPlan as _RolloutPlan,
)
from src.deployment.rollout_manager import (
    RolloutStage as _RolloutStage,
)
from src.deployment.rollout_manager import (
    RolloutStrategy as _RolloutStrategy,
)

CANARY_3_STAGE = _CANARY_3_STAGE
ROLLOUT_REGISTRY = _ROLLOUT_REGISTRY
RolloutManager = _RolloutManager
RolloutPlan = _RolloutPlan
RolloutStage = _RolloutStage
RolloutStrategy = _RolloutStrategy

__all__ = list(__all__) + [
    "CANARY_3_STAGE",
    "ROLLOUT_REGISTRY",
    "RolloutManager",
    "RolloutPlan",
    "RolloutStage",
    "RolloutStrategy",
]

# --- Secret provider ----------------------------------------------------------
from src.deployment.secret_provider import (
    SECRET_PROVIDER_REGISTRY as _SECRET_PROVIDER_REGISTRY,
)
from src.deployment.secret_provider import (
    SecretBackend as _SecretBackend,
)
from src.deployment.secret_provider import (
    SecretProvider as _SecretProvider,
)
from src.deployment.secret_provider import (
    SecretValue as _SecretValue,
)

SECRET_PROVIDER_REGISTRY = _SECRET_PROVIDER_REGISTRY
SecretBackend = _SecretBackend
SecretProvider = _SecretProvider
SecretValue = _SecretValue

__all__ = list(__all__) + [
    "SECRET_PROVIDER_REGISTRY",
    "SecretBackend",
    "SecretProvider",
    "SecretValue",
]

# --- Kubernetes operator (additive, cycle-146) ---------------------------------
# --- A/B test router ----------------------------------------------------------
from src.deployment.ab_test_router import (  # noqa: E402,F401
    AB_TEST_ROUTER_REGISTRY,
    ABTestRouter,
    Assignment,
    Variant,
)

# --- Canary controller --------------------------------------------------------
from src.deployment.canary_controller import (  # noqa: E402,F401
    CANARY_CONTROLLER_REGISTRY,
    DEFAULT_STAGES,
    CanaryController,
    CanaryStage,
    CanaryState,
)

# --- Config drift detector ----------------------------------------------------
from src.deployment.config_drift_detector import (  # noqa: E402,F401
    CONFIG_DRIFT_DETECTOR_REGISTRY,
    ConfigDriftDetector,
    ConfigField,
    DriftReport,
)
from src.deployment.kubernetes_operator import (  # noqa: E402,F401
    K8S_OPERATOR_REGISTRY,
    KubernetesOperator,
    ModelDeploymentSpec,
)

# --- Model packager (additive, cycle-146) --------------------------------------
from src.deployment.model_packager import (  # noqa: E402,F401
    MODEL_PACKAGER_REGISTRY,
    ModelPackager,
    PackageManifest,
)

# --- Serve config (additive, cycle-146) ----------------------------------------
from src.deployment.serve_config import (  # noqa: E402,F401
    SERVE_CONFIG_REGISTRY,
    ServeConfigBuilder,
    ServeDeploymentConfig,
)
