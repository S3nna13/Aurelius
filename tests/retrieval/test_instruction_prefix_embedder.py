"""Unit tests for :mod:`src.retrieval.instruction_prefix_embedder`."""

from __future__ import annotations

import pytest
import torch

from src.retrieval.dense_embedding_trainer import DenseEmbedder, EmbedderConfig
from src.retrieval.instruction_prefix_embedder import (
    INSTRUCTION_PREFIXES,
    InstructionPrefixEmbedder,
)


def _tiny_config() -> EmbedderConfig:
    return EmbedderConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=32,
        dropout=0.0,
        pad_token_id=0,
        embed_dim=16,
    )


def _build_embedder(seed: int = 0) -> DenseEmbedder:
    torch.manual_seed(seed)
    embedder = DenseEmbedder(_tiny_config())
    embedder.train(False)
    return embedder


class _ByteTokenizer:
    """Deterministic word-level tokenizer emitting ids in [1, vocab).

    Hashes each whitespace-separated token to a small id. This keeps
    tokenized lengths well below the tiny test ``max_seq_len``.
    """

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[int]:
        self.calls.append(text)
        words = text.split()
        return [
            ((sum(ord(c) for c in w) % (self.vocab_size - 1)) + 1) for w in words
        ]


def test_instruction_prefixes_has_all_six_documented_tasks() -> None:
    expected = {
        "query",
        "passage",
        "code_search_query",
        "code_search_passage",
        "question_answering",
        "classification",
    }
    assert set(INSTRUCTION_PREFIXES) == expected
    for prefix in INSTRUCTION_PREFIXES.values():
        assert isinstance(prefix, str) and len(prefix) > 0


def _make_wrapper(seed: int = 0) -> tuple[InstructionPrefixEmbedder, _ByteTokenizer]:
    emb = _build_embedder(seed)
    tok = _ByteTokenizer(vocab_size=emb.config.vocab_size)
    return InstructionPrefixEmbedder(emb, tok), tok


def test_encode_query_returns_embed_dim_vector() -> None:
    wrapper, _ = _make_wrapper()
    out = wrapper.encode("hi", task="query")
    assert out.shape == (16,)
    assert torch.isfinite(out).all()


def test_encode_passage_differs_from_query_for_same_input() -> None:
    wrapper, _ = _make_wrapper()
    q = wrapper.encode("hello", task="query")
    p = wrapper.encode("hello", task="passage")
    assert not torch.allclose(q, p, atol=1e-6)


def test_encode_batch_returns_b_by_embed_dim() -> None:
    wrapper, _ = _make_wrapper()
    out = wrapper.encode_batch(["a", "bb", "ccc"], task="passage")
    assert out.shape == (3, 16)


def test_similarity_returns_list_of_floats() -> None:
    wrapper, _ = _make_wrapper()
    sims = wrapper.similarity("foo", ["bar", "baz"])
    assert isinstance(sims, list)
    assert len(sims) == 2
    for s in sims:
        assert isinstance(s, float)


def test_similarity_of_identical_text_same_task_is_approximately_one() -> None:
    wrapper, _ = _make_wrapper()
    sims = wrapper.similarity(
        "abc", ["abc"], query_task="passage", passage_task="passage"
    )
    assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_different_task_yields_different_embedding_via_similarity() -> None:
    wrapper, _ = _make_wrapper()
    same_task = wrapper.similarity(
        "abc", ["abc"], query_task="query", passage_task="query"
    )[0]
    cross_task = wrapper.similarity(
        "abc", ["abc"], query_task="query", passage_task="passage"
    )[0]
    assert same_task == pytest.approx(1.0, abs=1e-5)
    assert cross_task < 0.9999


def test_unknown_task_raises() -> None:
    wrapper, _ = _make_wrapper()
    with pytest.raises(KeyError):
        wrapper.encode("hi", task="not_a_real_task")
    with pytest.raises(KeyError):
        wrapper.encode_batch(["hi"], task="bogus")
    with pytest.raises(KeyError):
        wrapper.similarity("hi", ["ho"], query_task="nope")


def test_tokenizer_receives_prefix_plus_text() -> None:
    wrapper, tok = _make_wrapper()
    wrapper.encode("world", task="query")
    assert len(tok.calls) == 1
    expected = INSTRUCTION_PREFIXES["query"] + "world"
    assert tok.calls[0] == expected


def test_text_exceeding_max_seq_len_raises() -> None:
    wrapper, _ = _make_wrapper()
    # prefix itself is already dozens of chars; anything non-trivial will
    # blow past the 32-token cap.
    # Prefix already contains ~7 words; appending many more pushes past
    # the 32-token cap of the tiny config.
    long_text = " ".join(f"w{i}" for i in range(200))
    with pytest.raises(ValueError):
        wrapper.encode(long_text, task="query")


def test_determinism() -> None:
    wrapper_a, _ = _make_wrapper(seed=123)
    wrapper_b, _ = _make_wrapper(seed=123)
    a = wrapper_a.encode("stable", task="passage")
    b = wrapper_b.encode("stable", task="passage")
    assert torch.allclose(a, b, atol=0.0)

    # Same wrapper, repeated call: identical output.
    c = wrapper_a.encode("stable", task="passage")
    assert torch.allclose(a, c, atol=0.0)


def test_empty_text_encodes_prefix_alone() -> None:
    wrapper, tok = _make_wrapper()
    out = wrapper.encode("", task="classification")
    assert out.shape == (16,)
    assert torch.isfinite(out).all()
    # tokenizer still saw the prefix string (empty text appended).
    assert tok.calls[-1] == INSTRUCTION_PREFIXES["classification"]


def test_code_search_tasks_differ_from_text_tasks() -> None:
    wrapper, _ = _make_wrapper()
    code_q = wrapper.encode("foo", task="code_search_query")
    text_q = wrapper.encode("foo", task="query")
    assert not torch.allclose(code_q, text_q, atol=1e-6)


def test_encode_batch_rejects_empty_list() -> None:
    wrapper, _ = _make_wrapper()
    with pytest.raises(ValueError):
        wrapper.encode_batch([], task="query")


def test_constructor_type_checks() -> None:
    emb = _build_embedder()
    with pytest.raises(TypeError):
        InstructionPrefixEmbedder("not an embedder", lambda s: [1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InstructionPrefixEmbedder(emb, "not callable")  # type: ignore[arg-type]


def test_encode_rejects_non_string() -> None:
    wrapper, _ = _make_wrapper()
    with pytest.raises(TypeError):
        wrapper.encode(123, task="query")  # type: ignore[arg-type]
