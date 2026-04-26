"""
Minimum Bayes Risk (MBR) Decoding
===================================
Select the hypothesis that maximizes expected utility under the model's
distribution, estimated via Monte Carlo sampling from the model.

Pure stdlib + torch only — no third-party dependencies.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SequenceSimilarity
# ---------------------------------------------------------------------------


class SequenceSimilarity:
    """Compute pairwise similarity between candidate integer sequences."""

    VALID_MODES = ("ngram_f1", "exact_match", "token_overlap")

    def __init__(self, mode: str = "ngram_f1") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode

    # ------------------------------------------------------------------
    # Core similarity measures
    # ------------------------------------------------------------------

    def ngram_f1(self, hyp: list[int], ref: list[int], n: int = 2) -> float:
        """F1 of n-gram overlap between *hyp* and *ref* (∈ [0, 1])."""

        def _ngrams(seq: list[int], n: int) -> Counter:
            return Counter(tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1)))

        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)

        hyp_total = sum(hyp_ng.values())
        ref_total = sum(ref_ng.values())

        if hyp_total == 0 and ref_total == 0:
            return 1.0
        if hyp_total == 0 or ref_total == 0:
            return 0.0

        # Overlap = sum of min counts for shared n-grams
        overlap = sum((hyp_ng & ref_ng).values())

        precision = overlap / hyp_total
        recall = overlap / ref_total

        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def exact_match(self, hyp: list[int], ref: list[int]) -> float:
        """Return 1.0 if sequences are identical, else 0.0."""
        return 1.0 if hyp == ref else 0.0

    def token_overlap(self, hyp: list[int], ref: list[int]) -> float:
        """Jaccard token overlap: |intersection| / |union| over token sets."""
        set_hyp = set(hyp)
        set_ref = set(ref)
        union = set_hyp | set_ref
        if not union:
            return 1.0
        intersection = set_hyp & set_ref
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, hyp: list[int], ref: list[int]) -> float:
        """Dispatch to the selected similarity mode."""
        if self.mode == "ngram_f1":
            return self.ngram_f1(hyp, ref)
        if self.mode == "exact_match":
            return self.exact_match(hyp, ref)
        if self.mode == "token_overlap":
            return self.token_overlap(hyp, ref)
        raise ValueError(f"Unknown mode: {self.mode!r}")


# ---------------------------------------------------------------------------
# HypothesisPool
# ---------------------------------------------------------------------------


class HypothesisPool:
    """Manage a pool of sampled hypotheses with associated log-probabilities."""

    def __init__(
        self,
        sequences: list[list[int]],
        log_probs: list[float],
    ) -> None:
        if len(sequences) != len(log_probs):
            raise ValueError(
                f"sequences and log_probs must have equal length, "
                f"got {len(sequences)} vs {len(log_probs)}"
            )
        if not sequences:
            raise ValueError("sequences must be non-empty")
        self.sequences: list[list[int]] = [list(s) for s in sequences]
        self.log_probs: list[float] = list(log_probs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def normalized_probs(self) -> list[float]:
        """Softmax over log_probs → probability weights that sum to 1."""
        lp = torch.tensor(self.log_probs, dtype=torch.float64)
        probs = F.softmax(lp, dim=0)
        return probs.tolist()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, with_replacement: bool = True) -> list[list[int]]:
        """Return *n* sequences sampled weighted by normalized_probs."""
        probs = torch.tensor(self.normalized_probs, dtype=torch.float64)
        indices = torch.multinomial(
            probs.float(), num_samples=n, replacement=with_replacement
        ).tolist()
        return [list(self.sequences[i]) for i in indices]

    # ------------------------------------------------------------------
    # Pool manipulation
    # ------------------------------------------------------------------

    def deduplicate(self) -> HypothesisPool:
        """Return a new HypothesisPool with unique sequences.

        For duplicate sequences the log-probabilities are combined via
        log-sum-exp (equivalent to summing the raw probabilities then
        taking the log again).
        """
        seen: dict[tuple[int, ...], float] = {}
        for seq, lp in zip(self.sequences, self.log_probs):
            key = tuple(seq)
            if key in seen:
                # log-sum-exp to combine probabilities
                a = seen[key]
                b = lp
                seen[key] = max(a, b) + math.log1p(math.exp(-abs(a - b)))
            else:
                seen[key] = lp

        new_seqs = [list(k) for k in seen]
        new_lps = list(seen.values())
        return HypothesisPool(new_seqs, new_lps)

    def top_k(self, k: int) -> HypothesisPool:
        """Return a new HypothesisPool with the *k* highest log_prob sequences."""
        k = min(k, len(self.sequences))
        indices = sorted(range(len(self.log_probs)), key=lambda i: self.log_probs[i], reverse=True)[
            :k
        ]
        return HypothesisPool(
            [self.sequences[i] for i in indices],
            [self.log_probs[i] for i in indices],
        )

    def __len__(self) -> int:
        return len(self.sequences)


# ---------------------------------------------------------------------------
# MBRScorer
# ---------------------------------------------------------------------------


class MBRScorer:
    """Score each hypothesis against a reference pool via expected similarity."""

    def __init__(
        self,
        similarity_fn: SequenceSimilarity,
        n_references: int = 10,
    ) -> None:
        self.similarity_fn = similarity_fn
        self.n_references = n_references

    def score_hypothesis(self, hyp: list[int], pool: HypothesisPool) -> float:
        """Sample n_references from pool and compute probability-weighted mean similarity."""
        probs = pool.normalized_probs  # length == len(pool)
        n_refs = min(self.n_references, len(pool))

        # Sample reference indices (with replacement)
        prob_tensor = torch.tensor(probs, dtype=torch.float32)
        ref_indices = torch.multinomial(prob_tensor, num_samples=n_refs, replacement=True).tolist()

        total_sim = 0.0
        total_weight = 0.0
        for idx in ref_indices:
            w = probs[idx]
            sim = self.similarity_fn(hyp, pool.sequences[idx])
            total_sim += w * sim
            total_weight += w

        if total_weight == 0.0:
            return 0.0
        return total_sim / total_weight

    def score_all(self, pool: HypothesisPool) -> list[float]:
        """Return MBR scores for every hypothesis in the pool."""
        return [self.score_hypothesis(seq, pool) for seq in pool.sequences]


# ---------------------------------------------------------------------------
# MBRDecoder
# ---------------------------------------------------------------------------


class MBRDecoder:
    """Full MBR decoding pipeline: sample → score → select best."""

    def __init__(
        self,
        model,
        similarity_fn: SequenceSimilarity,
        n_samples: int = 10,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.similarity_fn = similarity_fn
        self.n_samples = n_samples
        self.temperature = max(temperature, 1e-8)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> tuple[list[int], float]:
        """Autoregressively sample one sequence; return (token_ids, log_prob).

        Args:
            input_ids: LongTensor of shape (1, T) — the prompt.
            max_new_tokens: number of new tokens to generate.

        Returns:
            sequence: list of generated token ids (length == max_new_tokens).
            log_prob: sum of log-probabilities of each sampled token.
        """
        generated: list[int] = []
        cumulative_log_prob = 0.0

        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.model(current_ids)  # (1, T, V)
            # Take last position logits
            next_logits = logits[0, -1, :]  # (V,)

            # Apply temperature
            scaled_logits = next_logits / self.temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = log_probs.exp()

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            token_log_prob = log_probs[next_token].item()

            generated.append(int(next_token))
            cumulative_log_prob += token_log_prob

            # Append to context
            next_token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=current_ids.device
            )
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

        return generated, cumulative_log_prob

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> tuple[list[int], list[float]]:
        """Run MBR decoding.

        Args:
            input_ids: LongTensor of shape (1, T).
            max_new_tokens: tokens to generate per hypothesis.

        Returns:
            best_sequence: token list with highest MBR score.
            mbr_scores: list of MBR scores for all n_samples hypotheses.
        """
        # 1. Collect hypotheses
        all_seqs: list[list[int]] = []
        all_lps: list[float] = []

        for _ in range(self.n_samples):
            seq, lp = self._sample_sequence(input_ids, max_new_tokens)
            all_seqs.append(seq)
            all_lps.append(lp)

        pool = HypothesisPool(all_seqs, all_lps)

        # 2. Score all hypotheses with MBR
        scorer = MBRScorer(self.similarity_fn, n_references=max(1, self.n_samples))
        mbr_scores = scorer.score_all(pool)

        # 3. Select best
        best_idx = max(range(len(mbr_scores)), key=lambda i: mbr_scores[i])
        best_sequence = pool.sequences[best_idx]

        return best_sequence, mbr_scores


# ---------------------------------------------------------------------------
# MBRAnalytics
# ---------------------------------------------------------------------------


class MBRAnalytics:
    """Post-hoc analysis utilities for an MBR hypothesis pool."""

    def __init__(self) -> None:
        pass

    def diversity(self, pool: HypothesisPool) -> float:
        """Mean pairwise dissimilarity (1 - similarity) over the pool.

        Uses ngram_f1 internally.  Returns a float in [0, 1].
        """
        sim_fn = SequenceSimilarity(mode="ngram_f1")
        n = len(pool)
        if n < 2:
            return 0.0

        total_dissim = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_fn(pool.sequences[i], pool.sequences[j])
                total_dissim += 1.0 - sim
                count += 1

        return total_dissim / count if count > 0 else 0.0

    def confidence(self, scores: list[float]) -> float:
        """Margin = max(scores) - mean(scores).

        Measures how dominant the best candidate is.  Always ≥ 0.
        """
        if not scores:
            return 0.0
        best = max(scores)
        mean = sum(scores) / len(scores)
        return best - mean

    def length_stats(self, pool: HypothesisPool) -> dict:
        """Return basic length statistics over the hypothesis pool.

        Keys: mean_len, std_len, min_len, max_len.
        """
        lengths = [len(s) for s in pool.sequences]
        n = len(lengths)
        mean_len = sum(lengths) / n
        variance = sum((line - mean_len) ** 2 for line in lengths) / n
        std_len = math.sqrt(variance)
        return {
            "mean_len": mean_len,
            "std_len": std_len,
            "min_len": min(lengths),
            "max_len": max(lengths),
        }
