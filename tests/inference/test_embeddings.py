import pytest
import torch

from src.inference.embeddings import (
    EmbeddingConfig,
    EmbeddingExtractor,
    cosine_similarity_matrix,
    deduplicate_by_similarity,
    find_nearest_neighbors,
)
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


def test_encode_shape(small_model):
    extractor = EmbeddingExtractor(small_model)
    input_ids = torch.randint(0, 256, (2, 8))
    embs = extractor.encode(input_ids)
    assert embs.shape == (2, 64)  # (B, d_model)


def test_encode_normalized(small_model):
    extractor = EmbeddingExtractor(small_model, EmbeddingConfig(normalize=True))
    input_ids = torch.randint(0, 256, (3, 8))
    embs = extractor.encode(input_ids)
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_encode_last_pooling(small_model):
    extractor = EmbeddingExtractor(small_model, EmbeddingConfig(pooling="last", normalize=False))
    input_ids = torch.randint(0, 256, (2, 8))
    embs = extractor.encode(input_ids)
    assert embs.shape == (2, 64)


def test_encode_attention_mask(small_model):
    extractor = EmbeddingExtractor(small_model, EmbeddingConfig(normalize=False))
    input_ids = torch.randint(0, 256, (1, 8))
    full_mask = torch.ones(1, 8, dtype=torch.bool)
    half_mask = torch.zeros(1, 8, dtype=torch.bool)
    half_mask[0, :4] = True
    emb_full = extractor.encode(input_ids, full_mask)
    emb_half = extractor.encode(input_ids, half_mask)
    # Different masks -> different embeddings
    assert not torch.allclose(emb_full, emb_half)


def test_cosine_similarity_matrix_shape():
    embs = torch.randn(5, 64)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    sim = cosine_similarity_matrix(embs)
    assert sim.shape == (5, 5)


def test_cosine_similarity_self_is_one():
    embs = torch.randn(4, 32)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    sim = cosine_similarity_matrix(embs)
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones(4), atol=1e-5)


def test_find_nearest_neighbors():
    query = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    corpus = torch.nn.functional.normalize(torch.randn(10, 16), dim=-1)
    scores, indices = find_nearest_neighbors(query, corpus, top_k=3)
    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (scores <= 1.0 + 1e-5).all()


def test_deduplicate_by_similarity():
    # 3 similar + 2 different
    base = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
    similar = base.expand(3, -1) + torch.randn(3, 16) * 0.01
    different = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    embs = torch.cat([similar, different], dim=0)
    kept = deduplicate_by_similarity(embs, threshold=0.95)
    assert len(kept) < 5  # some duplicates removed
    assert 0 in kept  # first always kept
