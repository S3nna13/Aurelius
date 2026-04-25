"""Tests for src.training.data_provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.data_provenance import (
    DATA_PROVENANCE_REGISTRY,
    ProvenanceLedger,
    ProvenanceRecord,
    ProvenanceValidator,
)


def _rec(**overrides) -> ProvenanceRecord:
    defaults = dict(
        source_type="approved_corpus",
        license_tier="open",
        origin_model="aurelius-base",
        task_type="reasoning",
        split="train",
    )
    defaults.update(overrides)
    return ProvenanceRecord(**defaults)


def test_record_auto_sample_id():
    r = _rec()
    assert isinstance(r.sample_id, str) and len(r.sample_id) == 32


def test_record_auto_timestamp():
    from datetime import datetime

    r = _rec()
    assert isinstance(r.timestamp, datetime)


def test_record_is_frozen():
    r = _rec()
    with pytest.raises(Exception):
        r.origin_model = "other"  # type: ignore[misc]


def test_record_notes_default_empty():
    r = _rec()
    assert r.notes == ""


def test_record_to_dict_contains_all_fields():
    r = _rec(notes="hi")
    d = r.to_dict()
    for key in (
        "sample_id",
        "source_type",
        "license_tier",
        "origin_model",
        "timestamp",
        "task_type",
        "split",
        "notes",
    ):
        assert key in d


def test_record_to_dict_iso_timestamp():
    d = _rec().to_dict()
    assert "T" in d["timestamp"]


def test_unique_auto_sample_ids():
    ids = {_rec().sample_id for _ in range(50)}
    assert len(ids) == 50


def test_validator_accepts_valid():
    v = ProvenanceValidator()
    assert v.validate(_rec()) is True


def test_validator_rejects_unknown_license():
    v = ProvenanceValidator()
    assert v.validate(_rec(license_tier="unknown")) is False


def test_validator_rejects_restricted_by_default():
    v = ProvenanceValidator()
    assert v.validate(_rec(license_tier="restricted")) is False


def test_validator_accepts_restricted_when_allowed():
    v = ProvenanceValidator(allow_restricted=True)
    assert v.validate(_rec(license_tier="restricted")) is True


def test_validator_rejects_empty_origin_model():
    v = ProvenanceValidator()
    assert v.validate(_rec(origin_model="")) is False


def test_validator_rejects_invalid_source():
    v = ProvenanceValidator()
    # bypass Literal typing
    bad = ProvenanceRecord.__new__(ProvenanceRecord)
    object.__setattr__(bad, "source_type", "garbage")
    object.__setattr__(bad, "license_tier", "open")
    object.__setattr__(bad, "origin_model", "m")
    object.__setattr__(bad, "task_type", "reasoning")
    object.__setattr__(bad, "split", "train")
    object.__setattr__(bad, "sample_id", "abc")
    from datetime import datetime, timezone

    object.__setattr__(bad, "timestamp", datetime.now(timezone.utc))
    object.__setattr__(bad, "notes", "")
    assert ProvenanceValidator().validate(bad) is False


def test_validate_batch_splits():
    v = ProvenanceValidator()
    good = _rec()
    bad = _rec(license_tier="unknown")
    valid, errors = v.validate_batch([good, bad])
    assert len(valid) == 1 and len(errors) == 1
    assert valid[0].sample_id == good.sample_id


def test_validate_batch_all_valid():
    v = ProvenanceValidator()
    valid, errors = v.validate_batch([_rec(), _rec()])
    assert len(valid) == 2 and errors == []


def test_validate_batch_all_invalid():
    v = ProvenanceValidator()
    valid, errors = v.validate_batch([_rec(origin_model=""), _rec(license_tier="unknown")])
    assert valid == [] and len(errors) == 2


def test_ledger_add_and_get():
    ledger = ProvenanceLedger()
    r = _rec()
    ledger.add(r)
    assert ledger.get(r.sample_id) is r


def test_ledger_get_missing_returns_none():
    assert ProvenanceLedger().get("missing") is None


def test_ledger_add_duplicate_raises():
    ledger = ProvenanceLedger()
    r = _rec()
    ledger.add(r)
    with pytest.raises(ValueError):
        ledger.add(r)


def test_ledger_len_and_contains():
    ledger = ProvenanceLedger()
    assert len(ledger) == 0
    r = _rec()
    ledger.add(r)
    assert len(ledger) == 1
    assert r.sample_id in ledger


def test_ledger_all_records():
    ledger = ProvenanceLedger()
    records = [_rec() for _ in range(4)]
    for r in records:
        ledger.add(r)
    assert len(ledger.all_records()) == 4


def test_ledger_filter_by_source():
    ledger = ProvenanceLedger()
    ledger.add(_rec(source_type="approved_corpus"))
    ledger.add(_rec(source_type="synthetic_trace"))
    ledger.add(_rec(source_type="synthetic_trace"))
    assert len(ledger.filter_by(source_type="synthetic_trace")) == 2


def test_ledger_filter_by_split():
    ledger = ProvenanceLedger()
    ledger.add(_rec(split="train"))
    ledger.add(_rec(split="eval"))
    assert len(ledger.filter_by(split="eval")) == 1


def test_ledger_filter_by_task():
    ledger = ProvenanceLedger()
    ledger.add(_rec(task_type="code"))
    ledger.add(_rec(task_type="reasoning"))
    assert len(ledger.filter_by(task_type="code")) == 1


def test_ledger_filter_combined():
    ledger = ProvenanceLedger()
    ledger.add(_rec(task_type="code", split="train"))
    ledger.add(_rec(task_type="code", split="eval"))
    ledger.add(_rec(task_type="reasoning", split="train"))
    matched = ledger.filter_by(task_type="code", split="train")
    assert len(matched) == 1


def test_ledger_filter_no_filters_returns_all():
    ledger = ProvenanceLedger()
    for _ in range(3):
        ledger.add(_rec())
    assert len(ledger.filter_by()) == 3


def test_ledger_stats_counts():
    ledger = ProvenanceLedger()
    ledger.add(_rec(source_type="approved_corpus", split="train", task_type="code"))
    ledger.add(_rec(source_type="approved_corpus", split="eval", task_type="code"))
    ledger.add(_rec(source_type="synthetic_trace", split="train", task_type="reasoning"))
    stats = ledger.stats()
    assert stats["source_type"]["approved_corpus"] == 2
    assert stats["split"]["train"] == 2
    assert stats["task_type"]["code"] == 2
    assert stats["total"]["count"] == 3


def test_ledger_stats_empty():
    s = ProvenanceLedger().stats()
    assert s["total"]["count"] == 0


def test_ledger_export_jsonl_roundtrip(tmp_path: Path):
    ledger = ProvenanceLedger()
    rs = [_rec() for _ in range(3)]
    for r in rs:
        ledger.add(r)
    out = tmp_path / "out.jsonl"
    n = ledger.export_jsonl(str(out))
    assert n == 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert {p["sample_id"] for p in parsed} == {r.sample_id for r in rs}


def test_ledger_export_creates_parent(tmp_path: Path):
    ledger = ProvenanceLedger()
    ledger.add(_rec())
    out = tmp_path / "nested" / "dir" / "o.jsonl"
    ledger.export_jsonl(str(out))
    assert out.exists()


def test_ledger_clone_with_notes():
    ledger = ProvenanceLedger()
    r = _rec()
    ledger.add(r)
    updated = ledger.clone_with_notes(r.sample_id, "annotated")
    assert updated.notes == "annotated"
    assert ledger.get(r.sample_id).notes == "annotated"


def test_registry_default():
    assert "default" in DATA_PROVENANCE_REGISTRY
    assert DATA_PROVENANCE_REGISTRY["default"] is ProvenanceLedger
