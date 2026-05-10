"""Tests for verifiable_store.py — MemMachine-inspired ground-truth preservation."""

from __future__ import annotations

import time
import uuid

import pytest

from plugins.memory.verifiable_store import (
    VerificationRecord,
    VerifiableFact,
    VerifiableMemoryStore,
)


class TestVerifiableFact:
    """VerifiableFact dataclass."""

    def test_fact_creation(self):
        fact = VerifiableFact(
            fact_id="test-123",
            source="direct_observation",
            content="The model runs at 50 tok/s",
            timestamp=time.time(),
            confidence=0.85,
            metadata={"run_id": "run-1"},
            verifications=[],
        )
        assert fact.fact_id == "test-123"
        assert fact.content == "The model runs at 50 tok/s"
        assert fact.confidence == 0.85
        assert fact.compressed_version is None
        assert fact.is_active is True

    def test_fact_defaults(self):
        fact = VerifiableFact(
            fact_id="x",
            source="user_provided",
            content="Some content",
            timestamp=0.0,
            confidence=0.5,
            metadata={},
            verifications=[],
        )
        assert fact.compressed_version is None
        assert fact.is_active is True


class TestVerificationRecord:
    """VerificationRecord dataclass."""

    def test_record_creation(self):
        ts = time.time()
        record = VerificationRecord(
            verified_by="agent-1",
            timestamp=ts,
            method="re-read context",
            confidence=0.9,
        )
        assert record.verified_by == "agent-1"
        assert record.timestamp == ts
        assert record.method == "re-read context"
        assert record.confidence == 0.9


class TestVerifiableMemoryStore:
    """Ground-truth preserving memory store."""

    def test_store_and_retrieve(self):
        store = VerifiableMemoryStore()
        fact = VerifiableFact(
            fact_id="f1",
            source="direct_observation",
            content="CPU usage is 72%",
            timestamp=time.time(),
            confidence=0.8,
            metadata={},
            verifications=[],
        )
        fid = store.store_fact(fact)
        assert fid == "f1"
        retrieved = store.get_fact("f1")
        assert retrieved is not None
        assert retrieved.content == "CPU usage is 72%"
        assert retrieved.confidence == 0.8

    def test_get_fact_not_found(self):
        store = VerifiableMemoryStore()
        result = store.get_fact("nonexistent")
        assert result is None

    def test_ground_truth_preserved_under_compression(self):
        store = VerifiableMemoryStore()
        fact = VerifiableFact(
            fact_id="f2",
            source="inference",
            content="User prefers dark mode",
            timestamp=time.time(),
            confidence=0.6,
            metadata={},
            verifications=[],
        )
        store.store_fact(fact)
        store.store_compressed("f2", "user=dm")
        retrieved = store.get_fact("f2")
        assert retrieved.content == "User prefers dark mode"
        assert retrieved.compressed_version == "user=dm"
        gt = store.get_ground_truth("f2")
        assert gt == "User prefers dark mode"

    def test_verify_fact_updates_confidence(self):
        store = VerifiableMemoryStore()
        fact = VerifiableFact(
            fact_id="f3",
            source="inference",
            content="Model is hallucinating",
            timestamp=time.time(),
            confidence=0.9,
            metadata={},
            verifications=[],
        )
        store.store_fact(fact)
        ts = time.time()
        record = VerificationRecord(
            verified_by="agent-2",
            timestamp=ts,
            method="checked against ground truth",
            confidence=0.3,
        )
        store.verify_fact("f3", record)
        updated = store.get_fact("f3")
        assert len(updated.verifications) == 1
        assert updated.verifications[0].verified_by == "agent-2"
        assert updated.confidence == 0.3  # min of 0.9 and 0.3

    def test_verify_fact_preserves_higher_confidence(self):
        store = VerifiableMemoryStore()
        fact = VerifiableFact(
            fact_id="f4",
            source="direct_observation",
            content="Stable fact",
            timestamp=time.time(),
            confidence=0.5,
            metadata={},
            verifications=[],
        )
        store.store_fact(fact)
        ts = time.time()
        record = VerificationRecord(
            verified_by="agent-1",
            timestamp=ts,
            method="re-read",
            confidence=0.8,
        )
        store.verify_fact("f4", record)
        updated = store.get_fact("f4")
        assert updated.confidence == 0.5  # min(0.5, 0.8) = 0.5

    def test_get_unverified(self):
        store = VerifiableMemoryStore()
        for i in range(5):
            fact = VerifiableFact(
                fact_id=f"f{i}",
                source="direct_observation",
                content=f"Fact {i}",
                timestamp=time.time(),
                confidence=0.3 if i < 2 else 0.9,
                metadata={},
                verifications=[],
            )
            store.store_fact(fact)
        # min_confidence=0.5 returns facts with confidence < 0.5
        unverified = store.get_unverified(min_confidence=0.5)
        assert len(unverified) == 2  # f0 and f1 have confidence 0.3

    def test_search(self):
        store = VerifiableMemoryStore()
        for content in [
            "The model achieves 72% on MMLU",
            "CPU usage peaks at 95%",
            "The MMLU score is 72%",
            "Memory usage is stable",
        ]:
            fact = VerifiableFact(
                fact_id=str(uuid.uuid4()),
                source="direct_observation",
                content=content,
                timestamp=time.time(),
                confidence=0.8,
                metadata={},
                verifications=[],
            )
            store.store_fact(fact)
        results = store.search("MMLU", top_k=2)
        assert 1 <= len(results) <= 2
        for r in results:
            assert "MMLU" in r.content

    def test_multiple_verifications(self):
        store = VerifiableMemoryStore()
        fact = VerifiableFact(
            fact_id="f5",
            source="inference",
            content="Can be verified multiple ways",
            timestamp=time.time(),
            confidence=0.7,
            metadata={},
            verifications=[],
        )
        store.store_fact(fact)
        for i in range(3):
            ts = time.time()
            record = VerificationRecord(
                verified_by=f"agent-{i}",
                timestamp=ts,
                method=f"method-{i}",
                confidence=0.5 + i * 0.1,
            )
            store.verify_fact("f5", record)
        updated = store.get_fact("f5")
        assert len(updated.verifications) == 3
        assert updated.confidence == 0.5  # min of all

    def test_get_ground_truth_not_found_raises(self):
        store = VerifiableMemoryStore()
        with pytest.raises(KeyError):
            store.get_ground_truth("nonexistent")

    def test_count_and_clear(self):
        store = VerifiableMemoryStore()
        for i in range(3):
            fact = VerifiableFact(
                fact_id=f"f{i}",
                source="user_provided",
                content=f"Content {i}",
                timestamp=time.time(),
                confidence=0.8,
                metadata={},
                verifications=[],
            )
            store.store_fact(fact)
        assert store.count() == 3
        store.clear()
        assert store.count() == 0