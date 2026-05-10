from __future__ import annotations

from src.vector import VectorStore


def test_vector_store_upsert_and_search() -> None:
    store = VectorStore(collection="test", dim=4)
    store.upsert("1", [1.0, 0.1, 0.0, 0.0], {"text": "first"})
    store.upsert("2", [0.9, 1.0, 0.0, 0.0], {"text": "second"})

    results = store.search([1.0, 0.2, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].score > 0.9


def test_vector_store_search_empty() -> None:
    store = VectorStore(collection="empty_test", dim=4)
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_vector_store_delete() -> None:
    store = VectorStore(collection="delete_test", dim=4)
    store.upsert("1", [1.0, 0.0, 0.0, 0.0])
    assert store.count() == 1
    store.delete("1")
    assert store.count() == 0


def test_vector_store_count() -> None:
    store = VectorStore(collection="count_test", dim=4)
    assert store.count() == 0
    store.upsert("a", [0.1, 0.2, 0.3, 0.4])
    store.upsert("b", [0.5, 0.6, 0.7, 0.8])
    assert store.count() == 2


def test_vector_store_payload_roundtrip() -> None:
    store = VectorStore(collection="payload_test", dim=4)
    store.upsert("1", [1.0, 0.0, 0.0, 0.0], {"key": "value", "num": 42})
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].payload["key"] == "value"
    assert results[0].payload["num"] == 42
