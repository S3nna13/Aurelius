"""SLSA Level 3 provenance attestation generator.

Produces in-toto attestation statements recording builder identity,
build configuration, dependency materials (with SHA-256 digests), and
timestamps.  Pure stdlib — no third-party dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any


class ProvenanceError(Exception):
    """Base exception for provenance operations."""


class AttestationVerificationError(ProvenanceError):
    """Raised when an attestation signature fails verification."""


@dataclass(frozen=True)
class Material:
    """A dependency material identified by URI and SHA-256 digest."""

    uri: str
    sha256: str


@dataclass
class ProvenanceAttestation:
    """An in-toto-style provenance attestation statement.

    https://slsa.dev/spec/v1.0/provenance
    """

    builder_id: str
    build_config: dict[str, Any]
    materials: list[Material] = field(default_factory=list)
    recipe: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str = ""


class ProvenanceAttestationBuilder:
    """Build and verify SLSA Level 3 provenance attestations."""

    def __init__(self, builder_id: str, shared_key: bytes = b"") -> None:
        self._builder_id = builder_id
        self._shared_key = shared_key

    def build(
        self,
        build_config: dict[str, Any] | None = None,
        materials: list[Material] | None = None,
        recipe: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceAttestation:
        now = time.time()
        meta: dict[str, Any] = {
            "generated_at": now,
            "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        if metadata:
            meta.update(metadata)

        att = ProvenanceAttestation(
            builder_id=self._builder_id,
            build_config=build_config or {},
            materials=materials or [],
            recipe=recipe or {},
            metadata=meta,
        )
        att.signature = self._sign(self._payload(att))
        return att

    def verify(self, attestation: ProvenanceAttestation) -> bool:
        if not self._shared_key:
            msg = "no shared key configured — cannot verify"
            raise AttestationVerificationError(msg)
        expected = self._sign(self._payload(attestation))
        if not hmac.compare_digest(attestation.signature, expected):
            raise AttestationVerificationError("signature mismatch")
        return True

    @staticmethod
    def _payload(attestation: ProvenanceAttestation) -> bytes:
        return json.dumps(
            {
                "builder_id": attestation.builder_id,
                "build_config": attestation.build_config,
                "materials": [{"uri": m.uri, "sha256": m.sha256} for m in attestation.materials],
                "recipe": attestation.recipe,
                "metadata": attestation.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _sign(self, payload: bytes) -> str:
        if not self._shared_key:
            return ""
        return hmac.new(self._shared_key, payload, hashlib.sha256).hexdigest()


def build_file_attestation(
    file_path: str,
    builder_id: str,
    shared_key: bytes = b"",
    extra_labels: dict[str, Any] | None = None,
) -> ProvenanceAttestation:
    with open(file_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    builder = ProvenanceAttestationBuilder(builder_id=builder_id, shared_key=shared_key)
    return builder.build(
        build_config={"artifact_path": file_path},
        materials=[Material(uri=f"file:{file_path}", sha256=digest)],
        metadata={"labels": extra_labels or {}},
    )


PROVENANCE_ATTESTATION_REGISTRY: dict[str, ProvenanceAttestationBuilder] = {}
DEFAULT_PROVENANCE_BUILDER = ProvenanceAttestationBuilder(
    builder_id="https://aurelius.local/builder/v1",
)
PROVENANCE_ATTESTATION_REGISTRY["default"] = DEFAULT_PROVENANCE_BUILDER
