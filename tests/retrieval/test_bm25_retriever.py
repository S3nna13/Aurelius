"""Unit tests for the BM25 sparse retriever."""

from __future__ import annotations

import math
import time

import pytest

from src.retrieval.bm25_retriever import BM25Retriever, default_tokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "never gonna give you up never gonna let you down",
    "to be or not to be that is the question",
    "a fox is a small mammal of the dog family",
    "the lazy dog sleeps all day in the sun",
]


def _fresh() -> BM25Retriever:
    r = BM25Retriever()
    r.add_documents(CORPUS)
    return r


# --- 1. top-1 exact match ------------------------------------------------- #
def test_top1_exact_match():
    r = _fresh()
    top = r.query("brown fox jumps", k=1)
    assert len(top) == 1
    assert top[0][0] == 0  # "the quick brown fox jumps over the lazy dog"


# --- 2. top-k ordering ---------------------------------------------------- #
def test_topk_ordering_sane():
    r = _fresh()
    results = r.query("lazy dog", k=5)
    assert len(results) > 0
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # Both docs 0 and 4 mention "lazy dog" and should beat the others.
    top_ids = {doc for doc, _ in results[:2]}
    assert top_ids == {0, 4}


# --- 3. k larger than corpus --------------------------------------------- #
def test_k_larger_than_corpus():
    r = _fresh()
    # "the" appears in several docs; use a generic query that hits most.
    results = r.query("the dog fox question", k=1000)
    assert len(results) <= len(CORPUS)
    # All returned ids must be valid and unique.
    ids = [d for d, _ in results]
    assert len(set(ids)) == len(ids)
    assert all(0 <= d < len(CORPUS) for d in ids)


# --- 4. OOV-only query returns [] ---------------------------------------- #
def test_oov_query_empty():
    r = _fresh()
    assert r.query("zyxwvut qqqqq flibbertigibbet", k=5) == []


# --- 5. empty query returns [] ------------------------------------------- #
def test_empty_query():
    r = _fresh()
    assert r.query("", k=5) == []
    assert r.query("     ", k=5) == []  # whitespace tokenizes to nothing


# --- 6. determinism ------------------------------------------------------- #
def test_determinism():
    r1 = _fresh()
    r2 = _fresh()
    a = r1.query("fox dog", k=5)
    b = r2.query("fox dog", k=5)
    assert a == b
    # And scores are stable across repeated calls on the same instance.
    assert r1.query("fox dog", k=5) == a


# --- 7. batch matches individual ----------------------------------------- #
def test_batch_matches_individual():
    r = _fresh()
    qs = ["brown fox", "lazy dog", "question", "never gonna"]
    batch = r.query_batch(qs, k=3)
    single = [r.query(q, k=3) for q in qs]
    assert batch == single


# --- 8. custom tokenizer honored ----------------------------------------- #
def test_custom_tokenizer_honored():
    calls = []

    def tok(s: str) -> list[str]:
        calls.append(s)
        return s.split("|")

    r = BM25Retriever(tokenizer=tok)
    r.add_documents(["foo|bar|baz", "bar|qux", "baz|foo"])
    # Default regex tokenizer would split on word-chars, producing
    # "foo|bar|baz" -> ["foo", "bar", "baz"] too, but we verify the
    # pipe-splitter was actually called and that "qux" is a single
    # token (which the default tokenizer would also give us, so test
    # a separator-sensitive case):
    r2 = BM25Retriever(tokenizer=lambda s: [s])  # one token per doc
    r2.add_documents(["alpha beta", "gamma delta", "alpha beta"])
    # Query "alpha beta" must be the one token; docs 0 and 2 match.
    res = r2.query("alpha beta", k=3)
    ids = {d for d, _ in res}
    assert ids == {0, 2}
    # And the original custom tokenizer saw all docs.
    assert len(calls) == 3


# --- 9. duplicate documents get same score ------------------------------- #
def test_duplicate_documents_same_score():
    r = BM25Retriever()
    r.add_documents(["cat dog bird", "cat dog bird", "completely different text"])
    res = r.query("cat dog", k=3)
    # Docs 0 and 1 are identical, must have identical scores.
    by_id = dict(res)
    assert math.isclose(by_id[0], by_id[1], rel_tol=0, abs_tol=1e-12)


# --- 10. IDF matches hand-computed value --------------------------------- #
def test_idf_hand_computed():
    # 3-doc corpus with known term frequencies.
    docs = [
        "apple banana",  # doc 0: apple(1), banana(1)
        "apple cherry",  # doc 1: apple(1), cherry(1)
        "banana cherry date",  # doc 2: banana(1), cherry(1), date(1)
    ]
    r = BM25Retriever()
    r.add_documents(docs)
    n = 3
    # df: apple=2, banana=2, cherry=2, date=1
    expected = {
        "apple": math.log((n - 2 + 0.5) / (2 + 0.5) + 1.0),
        "banana": math.log((n - 2 + 0.5) / (2 + 0.5) + 1.0),
        "cherry": math.log((n - 2 + 0.5) / (2 + 0.5) + 1.0),
        "date": math.log((n - 1 + 0.5) / (1 + 0.5) + 1.0),
    }
    for tok, v in expected.items():
        assert math.isclose(r.idf(tok), v, abs_tol=1e-6), tok


# --- 11. scores are non-negative and finite ------------------------------ #
def test_scores_non_negative_finite():
    r = _fresh()
    for q in ["fox", "dog lazy", "the the the", "quick brown"]:
        for _, s in r.query(q, k=10):
            assert math.isfinite(s)
            assert s >= 0.0


# --- 12. adversarial: huge doc + huge query ------------------------------ #
def test_adversarial_scale():
    # 10K repeated tokens in one doc should not explode IDF.
    big = " ".join(["zap"] * 10_000)
    normal = "hello world"
    r = BM25Retriever()
    r.add_documents([big, normal, "another unrelated document"])
    # IDF of "zap" must be finite and bounded.
    idf_zap = r.idf("zap")
    assert math.isfinite(idf_zap)
    # df=1, N=3 -> log((3-1+0.5)/(1+0.5) + 1) = log(2.6666... + 1) ~ log(3.666)
    assert 0.0 < idf_zap < 10.0

    # 1K-token query must not hang.
    huge_query = " ".join(["zap"] * 1000)
    t0 = time.perf_counter()
    res = r.query(huge_query, k=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"query too slow: {elapsed:.3f}s"
    assert res[0][0] == 0  # the zap-heavy doc wins


# --- Extra correctness / edge-case tests --------------------------------- #
def test_unicode_and_case_folding():
    r = BM25Retriever()
    r.add_documents(["Café crème", "CAFÉ CRÈME", "hello world"])
    res = r.query("café", k=3)
    ids = {d for d, _ in res}
    # Case-folding + Unicode: docs 0 and 1 both match.
    assert ids == {0, 1}


def test_very_short_documents():
    r = BM25Retriever()
    r.add_documents(["a", "b", "a b", "c"])
    res = r.query("a", k=4)
    ids = [d for d, _ in res]
    assert ids[0] in (0, 2)  # both contain "a"
    assert all(math.isfinite(s) and s >= 0 for _, s in res)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        BM25Retriever(k1=0)
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1.0)
    with pytest.raises(ValueError):
        BM25Retriever(b=-0.1)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)
    with pytest.raises(TypeError):
        BM25Retriever(tokenizer=42)  # type: ignore[arg-type]


def test_empty_corpus_rejected():
    r = BM25Retriever()
    with pytest.raises(ValueError):
        r.add_documents([])


def test_query_before_index_raises():
    r = BM25Retriever()
    with pytest.raises(RuntimeError):
        r.query("hello", k=1)


def test_reindex_rejected():
    r = _fresh()
    with pytest.raises(RuntimeError):
        r.add_documents(["new doc"])


def test_invalid_k_raises():
    r = _fresh()
    with pytest.raises(ValueError):
        r.query("fox", k=0)
    with pytest.raises(ValueError):
        r.query("fox", k=-3)


def test_default_tokenizer_basic():
    assert default_tokenizer("Hello, WORLD!") == ["hello", "world"]
    assert default_tokenizer("") == []
