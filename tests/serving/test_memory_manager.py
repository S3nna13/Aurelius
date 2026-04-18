"""Tests for MemoryStore and SemanticMemory."""

import torch
import pytest

from src.serving.memory_manager import MemoryEntry, MemoryStore, SemanticMemory


EMBED_DIM = 64
VOCAB = 256


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "memory.json")
    return MemoryStore(storage_path=path)


@pytest.fixture
def sem_mem(store):
    return SemanticMemory(store=store, embed_dim=EMBED_DIM)


@pytest.fixture
def embed_weight():
    torch.manual_seed(42)
    return torch.randn(VOCAB, EMBED_DIM)


# 1. MemoryStore instantiates with tmp path
def test_memory_store_instantiates(tmp_path):
    path = str(tmp_path / "memory.json")
    ms = MemoryStore(storage_path=path)
    assert isinstance(ms, MemoryStore)


# 2. add returns a string id
def test_add_returns_string_id(store):
    entry_id = store.add("Hello world")
    assert isinstance(entry_id, str)
    assert len(entry_id) > 0


# 3. get retrieves added entry
def test_get_retrieves_entry(store):
    entry_id = store.add("Remember this")
    entry = store.get(entry_id)
    assert entry is not None
    assert isinstance(entry, MemoryEntry)
    assert entry.content == "Remember this"
    assert entry.id == entry_id


# 4. get returns None for missing id
def test_get_returns_none_for_missing(store):
    result = store.get("nonexistent-id")
    assert result is None


# 5. delete returns True for existing entry
def test_delete_returns_true_for_existing(store):
    entry_id = store.add("To be deleted")
    result = store.delete(entry_id)
    assert result is True
    assert store.get(entry_id) is None


# 6. delete returns False for missing entry
def test_delete_returns_false_for_missing(store):
    result = store.delete("no-such-id")
    assert result is False


# 7. list_all returns list containing added entries
def test_list_all_contains_added_entries(store):
    id1 = store.add("Entry one")
    id2 = store.add("Entry two")
    all_entries = store.list_all()
    ids = [e.id for e in all_entries]
    assert id1 in ids
    assert id2 in ids


# 8. search_by_tag finds entries by tag
def test_search_by_tag(store):
    store.add("Untagged entry")
    tagged_id = store.add("Tagged entry", tags=["important"])
    store.add("Other tag", tags=["misc"])
    results = store.search_by_tag("important")
    assert len(results) == 1
    assert results[0].id == tagged_id


# 9. SemanticMemory instantiates
def test_semantic_memory_instantiates(store):
    sm = SemanticMemory(store=store, embed_dim=EMBED_DIM)
    assert isinstance(sm, SemanticMemory)


# 10. remember stores entry and returns id
def test_remember_stores_entry(sem_mem, store, embed_weight):
    tokens = [1, 2, 3, 4]
    entry_id = sem_mem.remember("User prefers short answers", tokens, embed_weight, tags=["pref"])
    assert isinstance(entry_id, str)
    entry = store.get(entry_id)
    assert entry is not None
    assert entry.content == "User prefers short answers"
    assert entry.embedding is not None
    assert len(entry.embedding) == EMBED_DIM


# 11. recall returns list (may be empty with random embeddings, no error)
def test_recall_returns_list(sem_mem, embed_weight):
    tokens_store = [5, 10, 15]
    sem_mem.remember("Fact one", tokens_store, embed_weight)
    sem_mem.remember("Fact two", [20, 25], embed_weight)
    query_tokens = [5, 10]
    results = sem_mem.recall(query_tokens, embed_weight, top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


# 12. cosine_similarity of identical vectors = 1.0
def test_cosine_similarity_identical(sem_mem):
    vec = [1.0, 0.0, 0.5, -0.3]
    sim = sem_mem.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6
