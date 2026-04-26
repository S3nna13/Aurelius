"""Comprehensive text generation evaluation metrics.

Implements diversity, faithfulness, and statistical metrics for generated text:
- Distinct-n (Li et al. 2016)
- Repetition rate
- Self-BLEU
- Coverage and Density (Grusky et al. 2018)
- Compression ratio
- Generation statistics
- Vocabulary coverage
- Average token entropy
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.eval.text_metrics import bleu as text_metrics_bleu

# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Whitespace-split tokenizer."""
    return text.split()


def _tokenize_lower(text: str) -> list[str]:
    """Whitespace-split + lowercase tokenizer."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# N-gram helper
# ---------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return list of n-gram tuples from token list."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Distinct-n
# ---------------------------------------------------------------------------


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct-n (Li et al. 2016): measures lexical diversity.

    distinct_n = len(unique_n_grams) / len(all_n_grams)

    Tokenizes all texts together (pools all tokens), counts global
    unique/total n-grams.

    Returns float in [0, 1] (1 = all n-grams unique, 0 = all identical).
    Returns 0.0 if no n-grams exist.
    """
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = _tokenize(text)
        all_ngrams.extend(_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    unique_count = len(set(all_ngrams))
    total_count = len(all_ngrams)
    return unique_count / total_count


# ---------------------------------------------------------------------------
# Repetition rate
# ---------------------------------------------------------------------------


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that appear more than once.

    rep_n = 1 - distinct_n([text], n)

    High repetition -> model is looping.
    """
    return 1.0 - distinct_n([text], n)


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------


def _ngram_precision(hypothesis: list[str], reference_tokens: list[str], n: int) -> float:
    """Compute clipped n-gram precision of hypothesis against reference."""
    hyp_ngrams = Counter(_ngrams(hypothesis, n))
    ref_ngrams = Counter(_ngrams(reference_tokens, n))

    if not hyp_ngrams:
        return 0.0

    clipped = 0
    for gram, count in hyp_ngrams.items():
        clipped += min(count, ref_ngrams.get(gram, 0))

    total = sum(hyp_ngrams.values())
    return min(clipped / total, 1.0)


def self_bleu(texts: list[str], n_gram: int = 4) -> float:
    """Self-BLEU: measures similarity between multiple generated texts.

    For each text, compute BLEU against all other texts as references.
    Average over all texts.

    High self-BLEU -> low diversity (texts are similar to each other).
    Low self-BLEU -> high diversity.

    Uses geometric mean of 1-gram through n_gram precision (clipped to 1.0).

    Returns 0.0 if only 1 text (no diversity to measure).
    Returns float in [0, 1].
    """
    if len(texts) <= 1:
        return 0.0

    scores: list[float] = []
    tokenized = [_tokenize(t) for t in texts]

    for i, hyp_tokens in enumerate(tokenized):
        # Union of all other texts as reference
        ref_tokens: list[str] = []
        for j, tok in enumerate(tokenized):
            if j != i:
                ref_tokens.extend(tok)

        if not hyp_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        log_avg = 0.0
        active = 0
        for n in range(1, n_gram + 1):
            hyp_ngrams = _ngrams(hyp_tokens, n)
            if not hyp_ngrams:
                continue
            p = _ngram_precision(hyp_tokens, ref_tokens, n)
            active += 1
            if p == 0.0:
                # Avoid log(0): use tiny epsilon
                log_avg += math.log(1e-10)
            else:
                log_avg += math.log(p)

        if active == 0:
            scores.append(0.0)
        else:
            scores.append(math.exp(log_avg / active))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def coverage(source: str, summary: str) -> float:
    """Coverage (Grusky et al. 2018): fraction of summary tokens in source.

    coverage = |tokens(summary) ∩ tokens(source)| / |tokens(summary)|

    Tokenizes by whitespace split + lowercase.
    Returns 0 if summary is empty.
    """
    summary_tokens = _tokenize_lower(summary)
    if not summary_tokens:
        return 0.0

    source_token_set = set(_tokenize_lower(source))
    overlap = sum(1 for t in summary_tokens if t in source_token_set)
    return overlap / len(summary_tokens)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def density(source: str, summary: str) -> float:
    """Density: average length of extractive fragments.

    Fragment: maximal contiguous subsequence of summary tokens that appears
    in source. Uses greedy scan — for each position in summary, find the
    longest prefix that appears as a contiguous substring in source tokens.

    density = sum(len(fragment)^2) / len(summary_tokens)

    High density -> summary is more extractive (copied from source).
    """
    source_tokens = _tokenize_lower(source)
    summary_tokens = _tokenize_lower(summary)

    if not summary_tokens:
        return 0.0

    # Build a mapping from token to positions in source for fast lookup
    source_index: dict[str, list[int]] = {}
    for idx, tok in enumerate(source_tokens):
        source_index.setdefault(tok, []).append(idx)

    fragment_lengths: list[int] = []
    i = 0
    while i < len(summary_tokens):
        # Find the longest contiguous match starting at summary position i
        best_len = 0
        start_tok = summary_tokens[i]
        for src_pos in source_index.get(start_tok, []):
            length = 0
            while (
                i + length < len(summary_tokens)
                and src_pos + length < len(source_tokens)
                and summary_tokens[i + length] == source_tokens[src_pos + length]
            ):
                length += 1
            if length > best_len:
                best_len = length

        if best_len > 0:
            fragment_lengths.append(best_len)
            i += best_len
        else:
            i += 1

    return sum(fl**2 for fl in fragment_lengths) / len(summary_tokens)


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


def compression_ratio(source: str, summary: str) -> float:
    """len(source_tokens) / len(summary_tokens). Returns inf if summary empty."""
    summary_tokens = _tokenize(summary)
    if not summary_tokens:
        return float("inf")
    source_tokens = _tokenize(source)
    return len(source_tokens) / len(summary_tokens)


# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    mean_length: float
    std_length: float
    min_length: int
    max_length: int
    distinct_1: float
    distinct_2: float
    repetition_4gram: float
    self_bleu_4: float


def generation_statistics(texts: list[str]) -> GenerationStats:
    """Compute all generation statistics for a list of generated texts.

    Lengths are measured in whitespace-tokenized words.
    """
    if not texts:
        return GenerationStats(
            mean_length=0.0,
            std_length=0.0,
            min_length=0,
            max_length=0,
            distinct_1=0.0,
            distinct_2=0.0,
            repetition_4gram=0.0,
            self_bleu_4=0.0,
        )

    lengths = [len(_tokenize(t)) for t in texts]
    n = len(lengths)
    mean_len = sum(lengths) / n
    variance = sum((line - mean_len) ** 2 for line in lengths) / n
    std_len = math.sqrt(variance)

    d1 = distinct_n(texts, n=1)
    d2 = distinct_n(texts, n=2)
    rep4 = repetition_rate(" ".join(texts), n=4)
    sb4 = self_bleu(texts, n_gram=4)

    return GenerationStats(
        mean_length=mean_len,
        std_length=std_len,
        min_length=min(lengths),
        max_length=max(lengths),
        distinct_1=d1,
        distinct_2=d2,
        repetition_4gram=rep4,
        self_bleu_4=sb4,
    )


# ---------------------------------------------------------------------------
# Vocabulary coverage
# ---------------------------------------------------------------------------


def vocabulary_coverage(
    generated_texts: list[str],
    reference_vocabulary: set[str],
) -> dict:
    """Measure how much of the reference vocabulary is covered by generated texts.

    Returns:
        coverage_rate: fraction of ref vocab seen in generated
        unknown_word_rate: fraction of generated words not in ref vocab
        generated_vocab_size: number of unique words in generated texts
    """
    generated_tokens: list[str] = []
    for text in generated_texts:
        generated_tokens.extend(_tokenize_lower(text))

    generated_vocab = set(generated_tokens)
    generated_vocab_size = len(generated_vocab)

    if reference_vocabulary:
        covered = len(generated_vocab & reference_vocabulary)
        coverage_rate = covered / len(reference_vocabulary)
    else:
        coverage_rate = 0.0

    if generated_tokens:
        unknown_count = sum(1 for t in generated_tokens if t not in reference_vocabulary)
        unknown_word_rate = unknown_count / len(generated_tokens)
    else:
        unknown_word_rate = 0.0

    return {
        "coverage_rate": coverage_rate,
        "unknown_word_rate": unknown_word_rate,
        "generated_vocab_size": generated_vocab_size,
    }


# ---------------------------------------------------------------------------
# Average token entropy
# ---------------------------------------------------------------------------


def average_token_entropy(texts: list[str]) -> float:
    """Compute average unigram entropy over all generated texts.

    H = -sum(p(w) * log2(p(w))) where p(w) is word frequency across all texts.
    Returns float >= 0.
    """
    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(_tokenize(text))

    if not all_tokens:
        return 0.0

    counts = Counter(all_tokens)
    total = len(all_tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------


class GenerationEvaluator:
    """Comprehensive evaluator that computes all metrics at once."""

    def evaluate(
        self,
        generated: list[str],
        references: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> dict:
        """Compute available metrics.

        Always computed:
            distinct_1, distinct_2, repetition_rate, self_bleu, and all
            fields from generation_statistics.

        If references provided:
            bleu_4 (corpus-level average), coverage_vs_references.

        If sources provided:
            coverage, density, compression_ratio (averaged over pairs).

        Returns flat dict of all computed metrics.
        """
        result: dict = {}

        # Always-on metrics
        result["distinct_1"] = distinct_n(generated, n=1)
        result["distinct_2"] = distinct_n(generated, n=2)
        result["repetition_rate"] = repetition_rate(" ".join(generated), n=4)
        result["self_bleu"] = self_bleu(generated, n_gram=4)

        stats = generation_statistics(generated)
        result["mean_length"] = stats.mean_length
        result["std_length"] = stats.std_length
        result["min_length"] = stats.min_length
        result["max_length"] = stats.max_length

        # Reference-based metrics
        if references is not None and len(references) == len(generated):
            bleu_scores = [text_metrics_bleu(gen, [ref]) for gen, ref in zip(generated, references)]
            result["bleu_4"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

            cov_scores = [coverage(ref, gen) for gen, ref in zip(generated, references)]
            result["coverage_vs_references"] = (
                sum(cov_scores) / len(cov_scores) if cov_scores else 0.0
            )

        # Source-based metrics
        if sources is not None and len(sources) == len(generated):
            cov_scores = [coverage(src, gen) for gen, src in zip(generated, sources)]
            den_scores = [density(src, gen) for gen, src in zip(generated, sources)]
            comp_scores = [compression_ratio(src, gen) for gen, src in zip(generated, sources)]
            result["coverage"] = sum(cov_scores) / len(cov_scores) if cov_scores else 0.0
            result["density"] = sum(den_scores) / len(den_scores) if den_scores else 0.0
            result["compression_ratio"] = (
                sum(c for c in comp_scores if not math.isinf(c)) / len(comp_scores)
                if comp_scores
                else 0.0
            )

        return result
