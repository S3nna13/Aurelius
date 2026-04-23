"""Aurelius deployment surface — container build, health endpoints, SBOM.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

from src.deployment.container_builder import ARTIFACT_BUILDER_REGISTRY
from src.deployment.healthz import DEPLOY_TARGET_REGISTRY

__all__ = [
    "DEPLOY_TARGET_REGISTRY",
    "ARTIFACT_BUILDER_REGISTRY",
]
