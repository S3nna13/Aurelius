"""Data ingest pipeline with provenance/license enforcement.

Adapted from the Heavens_Gate data_builder for Aurelius. Enforces a moderate
policy on training samples: only whitelisted source types and license tiers
are accepted, and restricted-license material is rejected unless the operator
explicitly opts in.

Stdlib only (no pydantic).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestPolicy:
    """Policy controlling which training samples may be accepted."""

    allowed_source_types: frozenset = frozenset(
        {"internal_logs", "approved_corpus", "synthetic_trace"}
    )
    allowed_license_tiers: frozenset = frozenset({"open", "commercial"})
    allow_restricted: bool = False


@dataclass(frozen=True)
class TrainingSample:
    """A single training sample with attached provenance metadata."""

    prompt: str
    response: str
    source_type: str
    license_tier: str
    origin_model: str
    task_type: str
    split: str = "train"
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class IngestResult:
    """Result of an ingest batch: accepted samples plus rejection reasons."""

    accepted: list[TrainingSample]
    rejected: list[tuple[TrainingSample, str]]
    total: int
    accept_rate: float


class DataIngestPipeline:
    """Validate and partition raw training samples by ingest policy."""

    def __init__(self, policy: IngestPolicy | None = None) -> None:
        self.policy = policy if policy is not None else IngestPolicy()

    def validate(self, sample: TrainingSample) -> tuple[bool, str]:
        """Return ``(ok, reason)`` for a single sample under the current policy."""
        if sample.source_type not in self.policy.allowed_source_types:
            return False, f"source_type '{sample.source_type}' not allowed"

        license_ok = sample.license_tier in self.policy.allowed_license_tiers
        if not license_ok:
            if self.policy.allow_restricted and sample.license_tier == "restricted":
                pass
            else:
                return False, f"license_tier '{sample.license_tier}' not allowed"

        if len(sample.prompt) < 1:
            return False, "prompt is empty"
        if len(sample.response) < 1:
            return False, "response is empty"
        return True, "ok"

    def ingest(self, samples: list[TrainingSample]) -> IngestResult:
        """Validate every sample and return an :class:`IngestResult`."""
        accepted: list[TrainingSample] = []
        rejected: list[tuple[TrainingSample, str]] = []
        for s in samples:
            ok, reason = self.validate(s)
            if ok:
                accepted.append(s)
            else:
                rejected.append((s, reason))
        total = len(samples)
        rate = (len(accepted) / total) if total else 0.0
        return IngestResult(
            accepted=accepted,
            rejected=rejected,
            total=total,
            accept_rate=rate,
        )

    def export_jsonl(self, result: IngestResult, path: str) -> int:
        """Write accepted samples from *result* as JSONL to *path*."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for s in result.accepted:
                record = {
                    "sample_id": s.sample_id,
                    "prompt": s.prompt,
                    "response": s.response,
                    "source_type": s.source_type,
                    "license_tier": s.license_tier,
                    "origin_model": s.origin_model,
                    "task_type": s.task_type,
                    "split": s.split,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(result.accepted)

    def filter_by_split(
        self, result: IngestResult, split: str
    ) -> list[TrainingSample]:
        """Return accepted samples whose ``split`` matches *split*."""
        return [s for s in result.accepted if s.split == split]


DATA_INGEST_PIPELINE_REGISTRY = {"default": DataIngestPipeline}
