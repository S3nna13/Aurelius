"""Tests for hierarchical_summarizer module."""

import pytest
import torch

from src.inference.hierarchical_summarizer import (
    HierarchicalSummarizer,
    SummarizerConfig,
    TextChunk,
    chunk_document,
    encode_chunk,
    greedy_summarize,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def _encode(text):
    """Toy tokenizer: encode string -> list of byte values."""
    return list(text.encode("utf-8", errors="replace"))


def _decode(ids):
    """Toy tokenizer: decode list of ints -> string."""
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def summarizer(small_model):
    cfg = SummarizerConfig(
        chunk_size=16,
        overlap=4,
        max_summary_tokens=4,
        n_levels=2,
        compression_ratio=0.3,
    )
    return HierarchicalSummarizer(small_model, cfg, _encode, _decode)


# ---------------------------------------------------------------------------
# 1. SummarizerConfig defaults
# ---------------------------------------------------------------------------


def test_summarizer_config_defaults():
    cfg = SummarizerConfig()
    assert cfg.chunk_size == 256
    assert cfg.overlap == 32
    assert cfg.max_summary_tokens == 64
    assert cfg.n_levels == 2
    assert cfg.compression_ratio == 0.3


# ---------------------------------------------------------------------------
# 2. TextChunk fields
# ---------------------------------------------------------------------------


def test_text_chunk_fields():
    chunk = TextChunk(text="hello", token_ids=[1, 2, 3], start_pos=10, level=1)
    assert chunk.text == "hello"
    assert chunk.token_ids == [1, 2, 3]
    assert chunk.start_pos == 10
    assert chunk.level == 1


def test_text_chunk_default_level():
    chunk = TextChunk(text="x", token_ids=[0], start_pos=0)
    assert chunk.level == 0


# ---------------------------------------------------------------------------
# 3. chunk_document: correct number of chunks
# ---------------------------------------------------------------------------


def test_chunk_document_correct_n_chunks():
    ids = list(range(50))
    chunks = chunk_document(ids, chunk_size=20, overlap=5)
    # stride = 15; positions: 0, 15, 30 -- chunk at 30 covers [30..49] = end
    # so 3 chunks total
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# 4. chunk_document: each chunk correct size (except last)
# ---------------------------------------------------------------------------


def test_chunk_document_chunk_sizes():
    ids = list(range(50))
    chunks = chunk_document(ids, chunk_size=20, overlap=5)
    for c in chunks[:-1]:
        assert len(c.token_ids) == 20
    assert len(chunks[-1].token_ids) <= 20


# ---------------------------------------------------------------------------
# 5. chunk_document: overlap -- consecutive chunks share tokens
# ---------------------------------------------------------------------------


def test_chunk_document_overlap():
    ids = list(range(40))
    chunks = chunk_document(ids, chunk_size=16, overlap=4)
    if len(chunks) >= 2:
        end_of_first = set(chunks[0].token_ids[-4:])
        start_of_second = set(chunks[1].token_ids[:4])
        assert len(end_of_first & start_of_second) > 0


# ---------------------------------------------------------------------------
# 6. chunk_document: empty input returns empty list
# ---------------------------------------------------------------------------


def test_chunk_document_empty():
    chunks = chunk_document([], chunk_size=16, overlap=4)
    assert chunks == []


# ---------------------------------------------------------------------------
# 7. encode_chunk: returns tensor of shape (d_model,)
# ---------------------------------------------------------------------------


def test_encode_chunk_shape(small_model):
    chunk = TextChunk(text="hello", token_ids=[1, 2, 3, 4, 5], start_pos=0)
    emb = encode_chunk(small_model, chunk)
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (small_model.config.d_model,)


# ---------------------------------------------------------------------------
# 8. greedy_summarize: returns string
# ---------------------------------------------------------------------------


def test_greedy_summarize_returns_string(small_model):
    input_ids = [10, 20, 30, 40]
    result = greedy_summarize(small_model, input_ids, max_new_tokens=3, tokenizer_decode=_decode)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 9. HierarchicalSummarizer.summarize_chunk returns string
# ---------------------------------------------------------------------------


def test_summarize_chunk_returns_string(summarizer):
    chunk = TextChunk(text="test sentence", token_ids=_encode("test sentence"), start_pos=0)
    result = summarizer.summarize_chunk(chunk)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 10. HierarchicalSummarizer.summarize_level returns list of TextChunk
# ---------------------------------------------------------------------------


def test_summarize_level_returns_list(summarizer):
    chunks = [
        TextChunk(text="a", token_ids=_encode("hello world"), start_pos=0, level=0),
        TextChunk(text="b", token_ids=_encode("foo bar baz"), start_pos=12, level=0),
    ]
    result = summarizer.summarize_level(chunks)
    assert isinstance(result, list)
    assert all(isinstance(c, TextChunk) for c in result)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 11. HierarchicalSummarizer.summarize_level level incremented
# ---------------------------------------------------------------------------


def test_summarize_level_increments_level(summarizer):
    chunks = [
        TextChunk(text="a", token_ids=_encode("hello"), start_pos=0, level=0),
    ]
    result = summarizer.summarize_level(chunks)
    assert result[0].level == 1


def test_summarize_level_increments_level_from_higher(summarizer):
    chunks = [
        TextChunk(text="a", token_ids=_encode("hello"), start_pos=0, level=2),
    ]
    result = summarizer.summarize_level(chunks)
    assert result[0].level == 3


# ---------------------------------------------------------------------------
# 12. HierarchicalSummarizer.summarize returns dict with required keys
# ---------------------------------------------------------------------------


def test_summarize_returns_dict_keys(summarizer):
    text = "hello world " * 10
    result = summarizer.summarize(text)
    assert isinstance(result, dict)
    assert "summary" in result
    assert "n_chunks_level0" in result
    assert "n_levels" in result
    assert "compression" in result


# ---------------------------------------------------------------------------
# 13. HierarchicalSummarizer.summarize compression > 1 for long input
# ---------------------------------------------------------------------------


def test_summarize_compression_long_input(summarizer):
    # A long-ish text (more tokens than max_summary_tokens)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    result = summarizer.summarize(text)
    assert result["compression"] > 1.0


# ---------------------------------------------------------------------------
# 14. HierarchicalSummarizer.extractive_summary returns string
# ---------------------------------------------------------------------------


def test_extractive_summary_returns_string(summarizer):
    text = "The cat sat on the mat. The dog ran in the park. Birds fly in the sky."
    result = summarizer.extractive_summary(text, n_sentences=2)
    assert isinstance(result, str)
    assert len(result) > 0
