"""Tests for src.data.data_ingest_pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.data_ingest_pipeline import (
    DATA_INGEST_PIPELINE_REGISTRY,
    DataIngestPipeline,
    IngestPolicy,
    IngestResult,
    TrainingSample,
)


def _sample(**overrides) -> TrainingSample:
    defaults = {
        "prompt": "hello",
        "response": "world",
        "source_type": "approved_corpus",
        "license_tier": "open",
        "origin_model": "test-model",
        "task_type": "qa",
        "split": "train",
    }
    defaults.update(overrides)
    return TrainingSample(**defaults)


def test_registry_default():
    assert DATA_INGEST_PIPELINE_REGISTRY["default"] is DataIngestPipeline


def test_policy_defaults():
    p = IngestPolicy()
    assert "internal_logs" in p.allowed_source_types
    assert "approved_corpus" in p.allowed_source_types
    assert "synthetic_trace" in p.allowed_source_types
    assert "open" in p.allowed_license_tiers
    assert "commercial" in p.allowed_license_tiers
    assert p.allow_restricted is False


def test_pipeline_default_policy():
    p = DataIngestPipeline()
    assert isinstance(p.policy, IngestPolicy)


def test_validate_accepts_clean_sample():
    p = DataIngestPipeline()
    ok, _ = p.validate(_sample())
    assert ok is True


def test_validate_rejects_bad_source_type():
    p = DataIngestPipeline()
    ok, reason = p.validate(_sample(source_type="scraped_web"))
    assert ok is False
    assert "source_type" in reason


def test_validate_rejects_bad_license():
    p = DataIngestPipeline()
    ok, reason = p.validate(_sample(license_tier="restricted"))
    assert ok is False
    assert "license_tier" in reason


def test_validate_rejects_unknown_license():
    p = DataIngestPipeline()
    ok, _ = p.validate(_sample(license_tier="proprietary"))
    assert ok is False


def test_validate_rejects_empty_prompt():
    p = DataIngestPipeline()
    ok, reason = p.validate(_sample(prompt=""))
    assert ok is False
    assert "prompt" in reason


def test_validate_rejects_empty_response():
    p = DataIngestPipeline()
    ok, reason = p.validate(_sample(response=""))
    assert ok is False
    assert "response" in reason


def test_validate_allow_restricted_accepts_restricted_license():
    policy = IngestPolicy(allow_restricted=True)
    p = DataIngestPipeline(policy)
    ok, _ = p.validate(_sample(license_tier="restricted"))
    assert ok is True


def test_validate_allow_restricted_still_rejects_unknown_license():
    policy = IngestPolicy(allow_restricted=True)
    p = DataIngestPipeline(policy)
    ok, _ = p.validate(_sample(license_tier="mystery"))
    assert ok is False


def test_validate_accepts_commercial():
    p = DataIngestPipeline()
    ok, _ = p.validate(_sample(license_tier="commercial"))
    assert ok is True


def test_validate_accepts_internal_logs():
    p = DataIngestPipeline()
    ok, _ = p.validate(_sample(source_type="internal_logs"))
    assert ok is True


def test_validate_accepts_synthetic_trace():
    p = DataIngestPipeline()
    ok, _ = p.validate(_sample(source_type="synthetic_trace"))
    assert ok is True


def test_ingest_accept_rate_full():
    p = DataIngestPipeline()
    samples = [_sample(), _sample()]
    result = p.ingest(samples)
    assert result.accept_rate == 1.0
    assert result.total == 2


def test_ingest_accept_rate_partial():
    p = DataIngestPipeline()
    samples = [_sample(), _sample(prompt=""), _sample(license_tier="restricted")]
    result = p.ingest(samples)
    assert result.total == 3
    assert len(result.accepted) == 1
    assert len(result.rejected) == 2
    assert result.accept_rate == pytest.approx(1 / 3)


def test_ingest_empty_samples():
    p = DataIngestPipeline()
    result = p.ingest([])
    assert result.total == 0
    assert result.accept_rate == 0.0


def test_ingest_returns_ingest_result():
    p = DataIngestPipeline()
    result = p.ingest([_sample()])
    assert isinstance(result, IngestResult)


def test_rejected_contains_reason():
    p = DataIngestPipeline()
    bad = _sample(source_type="nope")
    result = p.ingest([bad])
    assert result.rejected[0][0] is bad
    assert "source_type" in result.rejected[0][1]


def test_sample_id_auto_generated():
    s = _sample()
    assert isinstance(s.sample_id, str)
    assert len(s.sample_id) == 12


def test_sample_ids_unique():
    a = _sample()
    b = _sample()
    assert a.sample_id != b.sample_id


def test_training_sample_frozen():
    s = _sample()
    with pytest.raises(Exception):
        s.prompt = "new"  # type: ignore[misc]


def test_export_jsonl_writes_accepted(tmp_path: Path):
    p = DataIngestPipeline()
    result = p.ingest([_sample(), _sample(prompt="")])
    path = tmp_path / "out.jsonl"
    n = p.export_jsonl(result, str(path))
    assert n == 1
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


def test_export_jsonl_records_have_fields(tmp_path: Path):
    p = DataIngestPipeline()
    result = p.ingest([_sample()])
    path = tmp_path / "out.jsonl"
    p.export_jsonl(result, str(path))
    record = json.loads(path.read_text().splitlines()[0])
    for k in (
        "sample_id",
        "prompt",
        "response",
        "source_type",
        "license_tier",
        "origin_model",
        "task_type",
        "split",
    ):
        assert k in record


def test_export_jsonl_creates_parent(tmp_path: Path):
    p = DataIngestPipeline()
    result = p.ingest([_sample()])
    path = tmp_path / "a" / "b" / "out.jsonl"
    p.export_jsonl(result, str(path))
    assert path.exists()


def test_filter_by_split_train():
    p = DataIngestPipeline()
    result = p.ingest([_sample(split="train"), _sample(split="validation")])
    out = p.filter_by_split(result, "train")
    assert len(out) == 1
    assert out[0].split == "train"


def test_filter_by_split_validation():
    p = DataIngestPipeline()
    result = p.ingest(
        [_sample(split="train"), _sample(split="validation"), _sample(split="validation")]
    )
    assert len(p.filter_by_split(result, "validation")) == 2


def test_filter_by_split_missing_returns_empty():
    p = DataIngestPipeline()
    result = p.ingest([_sample(split="train")])
    assert p.filter_by_split(result, "test") == []


def test_custom_policy_tightens_source_types():
    policy = IngestPolicy(allowed_source_types=frozenset({"internal_logs"}))
    p = DataIngestPipeline(policy)
    ok, _ = p.validate(_sample(source_type="approved_corpus"))
    assert ok is False


def test_custom_policy_allowed_license_tier_only_commercial():
    policy = IngestPolicy(allowed_license_tiers=frozenset({"commercial"}))
    p = DataIngestPipeline(policy)
    ok, _ = p.validate(_sample(license_tier="open"))
    assert ok is False


def test_accept_rate_zero_when_all_rejected():
    p = DataIngestPipeline()
    result = p.ingest([_sample(prompt=""), _sample(response="")])
    assert result.accept_rate == 0.0
    assert len(result.accepted) == 0
