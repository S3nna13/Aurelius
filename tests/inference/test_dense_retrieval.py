"""Tests for src/inference/dense_retrieval.py — 13 tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.dense_retrieval import (
    RetrieverConfig,
    BiEncoder,
    FlatIndex,
    CrossEncoderReranker,
    build_index,
    retrieve_and_rerank,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
PROJ_DIM = 32
SEQ_LEN = 16


@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def backbone(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def bi_encoder(backbone: AureliusTransformer) -> BiEncoder:
    torch.manual_seed(0)
    enc = BiEncoder(backbone, d_model=VOCAB_SIZE, proj_dim=PROJ_DIM)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def reranker(backbone: AureliusTransformer) -> CrossEncoderReranker:
    torch.manual_seed(1)
    r = CrossEncoderReranker(backbone)
    r.eval()
    return r


@pytest.fixture(scope="module")
def default_config() -> RetrieverConfig:
    return RetrieverConfig(top_k=3, use_reranker=False)


# ---------------------------------------------------------------------------
# Helper: generate random token ids
# ---------------------------------------------------------------------------

def rand_ids(batch: int, seq: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq))


# ---------------------------------------------------------------------------
# 1. RetrieverConfig defaults
# ---------------------------------------------------------------------------

def test_retriever_config_defaults():
    cfg = RetrieverConfig()
    assert cfg.d_model == 512
    assert cfg.index_type == "flat"
    assert cfg.top_k == 5
    assert cfg.use_reranker is False
    assert cfg.normalize_embeddings is True
    assert cfg.batch_size == 32


# ---------------------------------------------------------------------------
# 2. BiEncoder.encode output shape (B, proj_dim)
# ---------------------------------------------------------------------------

def test_bi_encoder_encode_shape(bi_encoder: BiEncoder):
    ids = rand_ids(4)
    embs = bi_encoder.encode(ids, normalize=False)
    assert embs.shape == (4, PROJ_DIM)


# ---------------------------------------------------------------------------
# 3. BiEncoder.encode L2-normalized (norms approx 1)
# ---------------------------------------------------------------------------

def test_bi_encoder_encode_normalized(bi_encoder: BiEncoder):
    ids = rand_ids(3)
    embs = bi_encoder.encode(ids, normalize=True)
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. BiEncoder.forward returns two tensors of same shape
# ---------------------------------------------------------------------------

def test_bi_encoder_forward_shapes(bi_encoder: BiEncoder):
    q_ids = rand_ids(2)
    p_ids = rand_ids(2)
    q_emb, p_emb = bi_encoder(q_ids, p_ids)
    assert q_emb.shape == p_emb.shape == (2, PROJ_DIM)


# ---------------------------------------------------------------------------
# 5. FlatIndex.add increases size correctly
# ---------------------------------------------------------------------------

def test_flat_index_add_size():
    idx = FlatIndex(dim=PROJ_DIM)
    assert idx.size() == 0

    embs1 = torch.randn(5, PROJ_DIM)
    idx.add(embs1)
    assert idx.size() == 5

    embs2 = torch.randn(3, PROJ_DIM)
    idx.add(embs2)
    assert idx.size() == 8


# ---------------------------------------------------------------------------
# 6. FlatIndex.search returns correct shapes (B, top_k) for scores and ids
# ---------------------------------------------------------------------------

def test_flat_index_search_shapes():
    idx = FlatIndex(dim=PROJ_DIM)
    idx.add(torch.randn(10, PROJ_DIM))

    query = torch.randn(2, PROJ_DIM)
    scores, ids = idx.search(query, top_k=3)

    assert scores.shape == (2, 3)
    assert ids.shape == (2, 3)


# ---------------------------------------------------------------------------
# 7. FlatIndex.search — top result is self when query is in index
# ---------------------------------------------------------------------------

def test_flat_index_search_self_match():
    idx = FlatIndex(dim=PROJ_DIM)
    # Normalize so dot-product == cosine
    corpus = F.normalize(torch.randn(8, PROJ_DIM), dim=-1)
    idx.add(corpus)

    # Query is exactly the 3rd vector
    query = corpus[3].unsqueeze(0)  # (1, PROJ_DIM)
    scores, ids = idx.search(query, top_k=1)

    assert ids[0, 0].item() == 3


# ---------------------------------------------------------------------------
# 8. FlatIndex.reset clears the index
# ---------------------------------------------------------------------------

def test_flat_index_reset():
    idx = FlatIndex(dim=PROJ_DIM)
    idx.add(torch.randn(5, PROJ_DIM))
    assert idx.size() == 5

    idx.reset()
    assert idx.size() == 0


# ---------------------------------------------------------------------------
# 9. CrossEncoderReranker.score output shape (B,)
# ---------------------------------------------------------------------------

def test_cross_encoder_score_shape(reranker: CrossEncoderReranker):
    ids = rand_ids(4, seq=SEQ_LEN)
    scores = reranker.score(ids)
    assert scores.shape == (4,)


# ---------------------------------------------------------------------------
# 10. build_index returns FlatIndex with correct size
# ---------------------------------------------------------------------------

def test_build_index_size(bi_encoder: BiEncoder):
    passages = [rand_ids(1, seq=SEQ_LEN)[0] for _ in range(7)]
    cfg = RetrieverConfig(batch_size=3)
    idx = build_index(passages, bi_encoder, cfg)
    assert isinstance(idx, FlatIndex)
    assert idx.size() == 7


# ---------------------------------------------------------------------------
# 11. retrieve_and_rerank returns list of ints
# ---------------------------------------------------------------------------

def test_retrieve_and_rerank_returns_list(bi_encoder: BiEncoder):
    passages = [rand_ids(1, seq=SEQ_LEN)[0] for _ in range(6)]
    cfg = RetrieverConfig(top_k=3, use_reranker=False)
    idx = build_index(passages, bi_encoder, cfg)

    query_ids = rand_ids(1, seq=SEQ_LEN)
    result = retrieve_and_rerank(query_ids, idx, bi_encoder, None, passages, cfg)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(i, int) for i in result)


# ---------------------------------------------------------------------------
# 12. retrieve_and_rerank — top result is most similar passage
# ---------------------------------------------------------------------------

def test_retrieve_and_rerank_top_is_most_similar(bi_encoder: BiEncoder):
    """Construct passages where passage 0 is a near-copy of the query."""
    torch.manual_seed(7)

    # Use a fixed query
    query_ids = rand_ids(1, seq=SEQ_LEN)  # (1, T)
    # Passage 0 = same as query; passages 1..5 = random (very different vocab region)
    similar_passage = query_ids[0].clone()  # same tokens
    other_passages = [
        # tokens drawn from high range (likely far in embedding space for this model)
        torch.full((SEQ_LEN,), fill_value=max(1, VOCAB_SIZE - 1 - i), dtype=torch.long)
        for i in range(5)
    ]
    passages = [similar_passage] + other_passages

    cfg = RetrieverConfig(top_k=3, use_reranker=False)
    idx = build_index(passages, bi_encoder, cfg)

    result = retrieve_and_rerank(query_ids, idx, bi_encoder, None, passages, cfg)

    # Passage 0 (identical to query) should be the top result
    assert result[0] == 0, f"Expected passage 0 first, got {result}"


# ---------------------------------------------------------------------------
# 13. retrieve_and_rerank with reranker=None still works
# ---------------------------------------------------------------------------

def test_retrieve_and_rerank_no_reranker(bi_encoder: BiEncoder):
    passages = [rand_ids(1, seq=SEQ_LEN)[0] for _ in range(5)]
    cfg = RetrieverConfig(top_k=2, use_reranker=False)
    idx = build_index(passages, bi_encoder, cfg)

    query_ids = rand_ids(1, seq=SEQ_LEN)
    result = retrieve_and_rerank(
        query_ids, idx, bi_encoder, reranker=None, passage_ids_list=passages, config=cfg
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(0 <= i < 5 for i in result)
