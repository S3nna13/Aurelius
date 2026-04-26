import pytest
import torch

from src.inference.rag import RAGRetriever, RetrievalResult, VectorStore
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def _make_store():
    store = VectorStore()
    torch.manual_seed(42)
    for i in range(5):
        emb = torch.nn.functional.normalize(torch.randn(64), dim=-1)
        store.add(emb, f"document {i}")
    return store


def test_vector_store_add_and_len():
    store = VectorStore()
    emb = torch.randn(64)
    store.add(emb, "hello")
    assert len(store) == 1


def test_vector_store_search_shape():
    store = _make_store()
    query = torch.nn.functional.normalize(torch.randn(64), dim=-1)
    result = store.search(query, top_k=3)
    assert isinstance(result, RetrievalResult)
    assert len(result.texts) == 3
    assert len(result.scores) == 3
    assert len(result.indices) == 3


def test_vector_store_search_sorted():
    store = _make_store()
    query = torch.nn.functional.normalize(torch.randn(64), dim=-1)
    result = store.search(query, top_k=5)
    assert result.scores[0] >= result.scores[-1]


def test_vector_store_search_empty():
    store = VectorStore()
    query = torch.randn(64)
    result = store.search(query, top_k=3)
    assert result.texts == []
    assert result.scores == []


def test_vector_store_top_k_capped():
    store = _make_store()  # 5 items
    query = torch.nn.functional.normalize(torch.randn(64), dim=-1)
    result = store.search(query, top_k=100)  # more than stored
    assert len(result.texts) == 5


def test_vector_store_best_match():
    """The most similar embedding should be ranked first."""
    store = VectorStore()
    target = torch.nn.functional.normalize(torch.randn(64), dim=-1)
    store.add(target, "match")
    store.add(torch.nn.functional.normalize(torch.randn(64), dim=-1), "noise1")
    store.add(torch.nn.functional.normalize(torch.randn(64), dim=-1), "noise2")
    result = store.search(target, top_k=1)
    assert result.texts[0] == "match"


def test_rag_retriever_retrieve(small_model):
    store = _make_store()
    retriever = RAGRetriever(small_model, store)
    input_ids = torch.randint(0, 256, (1, 8))
    result = retriever.retrieve(input_ids, top_k=3)
    assert len(result.texts) == 3


def test_rag_retriever_index_and_retrieve(small_model):
    store = VectorStore()
    retriever = RAGRetriever(small_model, store)
    input_ids = torch.randint(0, 256, (3, 8))
    texts = ["doc A", "doc B", "doc C"]
    retriever.index(input_ids, texts)
    assert len(store) == 3
    query = torch.randint(0, 256, (1, 8))
    result = retriever.retrieve(query, top_k=2)
    assert len(result.texts) == 2
