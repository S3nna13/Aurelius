"""
15 tests for src/eval/nlg_metrics.py
Pure stdlib + torch; fast string-based inputs.
"""

import math

import pytest
import torch
from aurelius.eval.nlg_metrics import (
    BLEUScore,
    METEORScore,
    NGramCounter,
    ROUGEScore,
    SemanticSimilarityScore,
)


# ---------------------------------------------------------------------------
# 1. NGramCounter.count — correct n-gram counts for known sequence
# ---------------------------------------------------------------------------
def test_ngram_counter_count_bigrams():
    counter = NGramCounter(2)
    tokens = ["a", "b", "a", "c"]
    counts = counter.count(tokens)
    assert counts[("a", "b")] == 1
    assert counts[("b", "a")] == 1
    assert counts[("a", "c")] == 1
    assert len(counts) == 3


# ---------------------------------------------------------------------------
# 2. NGramCounter.clip_count — min between hyp and ref counts
# ---------------------------------------------------------------------------
def test_ngram_counter_clip_count():
    counter = NGramCounter(1)
    hyp_counts = counter.count(["a", "a", "b", "c"])
    ref_counts = counter.count(["a", "b", "b", "d"])
    clipped = counter.clip_count(hyp_counts, ref_counts)
    # "a": min(2,1)=1, "b": min(1,2)=1, "c": min(1,0)=0 => total=2
    assert clipped == 2


# ---------------------------------------------------------------------------
# 3. BLEUScore.sentence_bleu — 1.0 for identical hypothesis and reference
# ---------------------------------------------------------------------------
def test_bleu_sentence_identical():
    bleu = BLEUScore(max_n=4, smooth=True)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    score = bleu.sentence_bleu(tokens, tokens)
    assert abs(score - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. BLEUScore.sentence_bleu — 0.0 for completely disjoint sequences
# ---------------------------------------------------------------------------
def test_bleu_sentence_disjoint():
    bleu = BLEUScore(max_n=4, smooth=False)
    hyp = ["a", "b", "c", "d"]
    ref = ["x", "y", "z", "w"]
    score = bleu.sentence_bleu(hyp, ref)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 5. BLEUScore.bleu_n — individual n=1 unigram precision
# ---------------------------------------------------------------------------
def test_bleu_n_unigram():
    bleu = BLEUScore()
    hyp = ["the", "cat", "sat"]
    ref = ["the", "dog", "sat", "there"]
    p1 = bleu.bleu_n(hyp, ref, n=1)
    # "the" clips to 1, "sat" clips to 1, total clipped=2, total hyp=3
    assert abs(p1 - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# 6. BLEUScore.bleu_n — individual n=2 bigram precision
# ---------------------------------------------------------------------------
def test_bleu_n_bigram():
    bleu = BLEUScore()
    hyp = ["a", "b", "c"]
    ref = ["a", "b", "d"]
    p2 = bleu.bleu_n(hyp, ref, n=2)
    # bigrams hyp: (a,b),(b,c); ref: (a,b),(b,d); overlap: (a,b)=1; total=2
    assert abs(p2 - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 7. BLEUScore.corpus_bleu — higher than worst sentence BLEU
# ---------------------------------------------------------------------------
def test_bleu_corpus_vs_sentence():
    bleu = BLEUScore(max_n=2, smooth=True)
    hyps = [
        ["the", "cat", "sat"],
        ["hello", "world"],
    ]
    refs = [
        ["the", "cat", "sat"],  # perfect match
        ["goodbye", "earth"],  # zero match
    ]
    corpus = bleu.corpus_bleu(hyps, refs)
    # worst sentence BLEU (smooth) > 0; corpus should be > second sentence
    sentence_bad = bleu.sentence_bleu(hyps[1], refs[1])
    assert corpus > sentence_bad


# ---------------------------------------------------------------------------
# 8. BLEUScore brevity_penalty — < 1.0 for short hypothesis
# ---------------------------------------------------------------------------
def test_brevity_penalty_short():
    bleu = BLEUScore()
    bp = bleu._brevity_penalty(hyp_len=3, ref_len=5)
    assert bp < 1.0
    assert bp == pytest.approx(math.exp(1 - 5 / 3), rel=1e-6)


# ---------------------------------------------------------------------------
# 9. BLEUScore brevity_penalty — = 1.0 when hyp >= ref
# ---------------------------------------------------------------------------
def test_brevity_penalty_no_penalty():
    bleu = BLEUScore()
    assert bleu._brevity_penalty(5, 5) == 1.0
    assert bleu._brevity_penalty(6, 4) == 1.0


# ---------------------------------------------------------------------------
# 10. ROUGEScore.rouge_n — f1=1.0 for identical, f1=0.0 for disjoint
# ---------------------------------------------------------------------------
def test_rouge_n_identical_and_disjoint():
    rouge = ROUGEScore()
    tokens = ["a", "b", "c", "d"]
    assert rouge.rouge_n(tokens, tokens, n=1)["f1"] == pytest.approx(1.0)
    disjoint = ["x", "y", "z"]
    assert rouge.rouge_n(tokens, disjoint, n=1)["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. ROUGEScore.rouge_l — f1 ≤ rouge_1 f1 (LCS ≤ unigram overlap)
# ---------------------------------------------------------------------------
def test_rouge_l_leq_rouge_1():
    rouge = ROUGEScore()
    hyp = ["the", "cat", "on", "the", "mat"]
    ref = ["the", "mat", "on", "the", "cat"]
    r1 = rouge.rouge_n(hyp, ref, n=1)["f1"]
    rl = rouge.rouge_l(hyp, ref)["f1"]
    assert rl <= r1 + 1e-9


# ---------------------------------------------------------------------------
# 12. ROUGEScore.rouge_l — LCS correctly finds longest common subsequence
# ---------------------------------------------------------------------------
def test_rouge_l_lcs_value():
    rouge = ROUGEScore()
    hyp = ["a", "b", "c", "d"]
    ref = ["a", "x", "c", "d"]
    # LCS = ["a","c","d"] length 3
    rl = rouge.rouge_l(hyp, ref)
    assert rl["precision"] == pytest.approx(3 / 4)
    assert rl["recall"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# 13. METEORScore.score — in [0, 1] for any inputs
# ---------------------------------------------------------------------------
def test_meteor_range():
    meteor = METEORScore()
    hyp = ["the", "cat", "sat", "on", "a", "mat"]
    ref = ["the", "dog", "sat", "by", "the", "door"]
    s = meteor.score(hyp, ref)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 14. METEORScore.score — 1.0 for identical sequences
# ---------------------------------------------------------------------------
def test_meteor_identical():
    meteor = METEORScore()
    tokens = ["the", "cat", "sat"]
    s = meteor.score(tokens, tokens)
    assert abs(s - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 15. SemanticSimilarityScore.embed — unit norm, correct shape
# ---------------------------------------------------------------------------
def test_semantic_embed_shape_and_norm():
    sem = SemanticSimilarityScore(embed_dim=16)
    tokens = ["hello", "world", "foo"]
    emb = sem.embed(tokens)
    assert emb.shape == (3, 16)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


# ---------------------------------------------------------------------------
# 16 (counts as test 15 set): SemanticSimilarityScore.score — f1 in [0,1], =1 for identical
# ---------------------------------------------------------------------------
def test_semantic_score_identical_and_range():
    sem = SemanticSimilarityScore(embed_dim=16)
    tokens = ["the", "cat", "sat"]
    result = sem.score(tokens, tokens)
    assert 0.0 <= result["f1"] <= 1.0 + 1e-6
    assert result["f1"] == pytest.approx(1.0, abs=1e-5)

    # range check on partial overlap
    hyp = ["the", "cat"]
    ref = ["dog", "runs"]
    result2 = sem.score(hyp, ref)
    assert 0.0 <= result2["f1"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 17 (extra): All metrics handle empty hypothesis gracefully — return 0 not crash
# ---------------------------------------------------------------------------
def test_empty_hypothesis_all_metrics():
    ref = ["the", "cat", "sat"]

    bleu = BLEUScore()
    assert bleu.sentence_bleu([], ref) == 0.0
    assert bleu.corpus_bleu([[]], [ref]) == 0.0

    rouge = ROUGEScore()
    assert rouge.rouge_n([], ref)["f1"] == 0.0
    assert rouge.rouge_l([], ref)["f1"] == 0.0
    assert rouge.rouge_w([], ref)["f1"] == 0.0

    meteor = METEORScore()
    assert meteor.score([], ref) == 0.0

    sem = SemanticSimilarityScore()
    result = sem.score([], ref)
    assert result["f1"] == 0.0
