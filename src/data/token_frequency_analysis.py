"""Token frequency analysis for dataset quality and preprocessing.

Analyzes token distributions: frequency counts, Zipf law fit,
rare token detection, and distribution shift between datasets.

Useful for: tokenizer analysis, dataset balancing, OOV detection,
curriculum learning order, and identifying domain-specific tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# TokenFrequencyCounter
# ---------------------------------------------------------------------------


class TokenFrequencyCounter:
    """Accumulates token frequency counts over one or more batches."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.counts: Tensor = torch.zeros(vocab_size, dtype=torch.long)
        self.total_tokens: int = 0

    def update(self, token_ids: Tensor) -> None:
        """Accumulate counts for a batch of token ids (any shape LongTensor)."""
        flat = token_ids.reshape(-1).long()
        self.counts += torch.bincount(flat, minlength=self.vocab_size)
        self.total_tokens += flat.numel()

    def frequency(self) -> Tensor:
        """Return per-token relative frequency as a (vocab_size,) float tensor."""
        return self.counts.float() / self.total_tokens

    def top_k_tokens(self, k: int) -> tuple[Tensor, Tensor]:
        """Return (token_ids, counts) for the k most frequent tokens, descending."""
        topk = torch.topk(self.counts, k=min(k, self.vocab_size))
        return topk.indices, topk.values

    def rare_tokens(self, min_count: int = 5) -> Tensor:
        """Return token ids whose count is strictly less than min_count."""
        return torch.where(self.counts < min_count)[0]


# ---------------------------------------------------------------------------
# ZipfAnalyzer
# ---------------------------------------------------------------------------


class ZipfAnalyzer:
    """Tests whether a token distribution follows Zipf's law."""

    def __init__(self) -> None:
        pass  # stateless

    def fit_zipf(self, counts: Tensor) -> dict[str, float]:
        """Fit log-log linear regression to rank-frequency data.

        Returns {'slope': float, 'intercept': float, 'r_squared': float}.
        Zipf's law predicts slope ≈ -1.
        """
        sorted_counts, _ = torch.sort(counts.float(), descending=True)
        # Only keep non-zero counts to avoid log(0)
        sorted_counts = sorted_counts[sorted_counts > 0]
        n = sorted_counts.numel()
        if n == 0:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

        ranks = torch.arange(1, n + 1, dtype=torch.float32)
        log_ranks = torch.log(ranks)
        log_counts = torch.log(sorted_counts)

        # Build design matrix [1, log_rank] for lstsq
        # Solve: log_count = intercept + slope * log_rank
        A = torch.stack([torch.ones(n), log_ranks], dim=1)  # (n, 2)
        # torch.linalg.lstsq returns solution in .solution
        result = torch.linalg.lstsq(A, log_counts.unsqueeze(1))
        intercept = result.solution[0, 0].item()
        slope = result.solution[1, 0].item()

        # R-squared
        predicted = intercept + slope * log_ranks
        ss_res = torch.sum((log_counts - predicted) ** 2).item()
        ss_tot = torch.sum((log_counts - log_counts.mean()) ** 2).item()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        }

    def zipf_divergence(self, counts: Tensor) -> float:
        """Return |slope + 1|, measuring deviation from ideal Zipf (slope=-1)."""
        result = self.fit_zipf(counts)
        return abs(result["slope"] + 1.0)


# ---------------------------------------------------------------------------
# DistributionShiftDetector
# ---------------------------------------------------------------------------


class DistributionShiftDetector:
    """Detects vocabulary distribution shift between two datasets."""

    def __init__(self) -> None:
        pass  # stateless

    def kl_divergence(self, freq_a: Tensor, freq_b: Tensor, eps: float = 1e-8) -> float:
        """KL(A || B) with eps smoothing.  Returns a non-negative scalar."""
        a = freq_a.float() + eps
        b = freq_b.float() + eps
        # Normalise so they are proper distributions
        a = a / a.sum()
        b = b / b.sum()
        kl = torch.sum(a * torch.log(a / b)).item()
        return float(kl)

    def top_k_shift(self, freq_a: Tensor, freq_b: Tensor, k: int = 20) -> dict[str, Tensor]:
        """Return tokens that entered ('gained') or left ('lost') the top-k between A and B."""
        _, top_a_ids = torch.topk(freq_a, k=min(k, freq_a.numel()))
        _, top_b_ids = torch.topk(freq_b, k=min(k, freq_b.numel()))

        set_a = set(top_a_ids.tolist())
        set_b = set(top_b_ids.tolist())

        gained = torch.tensor(sorted(set_b - set_a), dtype=torch.long)
        lost = torch.tensor(sorted(set_a - set_b), dtype=torch.long)
        return {"gained": gained, "lost": lost}

    def coverage(self, freq_a: Tensor, freq_b: Tensor, threshold: float = 1e-6) -> float:
        """Fraction of B-vocabulary tokens (freq_b > threshold) that also appear in A."""
        in_b = freq_b > threshold
        in_a = freq_a > threshold
        b_count = in_b.sum().item()
        if b_count == 0:
            return 1.0
        overlap = (in_b & in_a).sum().item()
        return float(overlap / b_count)


# ---------------------------------------------------------------------------
# VocabularyStats
# ---------------------------------------------------------------------------


class VocabularyStats:
    """Summary statistics about token vocabulary usage."""

    def __init__(self) -> None:
        pass  # stateless

    def compute(self, counter: TokenFrequencyCounter) -> dict[str, float]:
        """Compute summary stats from a TokenFrequencyCounter.

        Returns keys: vocab_coverage, hapax_legomena, mean_freq,
                      median_freq, max_freq, entropy.
        """
        counts = counter.counts
        vocab_size = counter.vocab_size

        vocab_coverage = float((counts > 0).sum().item()) / vocab_size

        hapax_legomena = int((counts == 1).sum().item())

        freq = counter.frequency()
        nonzero_freq = freq[freq > 0]

        mean_freq = float(freq.mean().item())
        # torch.median on the full distribution (including zeros)
        median_freq = float(torch.median(freq).item())
        max_freq = float(freq.max().item())

        # Shannon entropy: -sum(p * log(p)) over non-zero probabilities
        if nonzero_freq.numel() > 0:
            entropy = float(-torch.sum(nonzero_freq * torch.log(nonzero_freq)).item())
        else:
            entropy = 0.0

        return {
            "vocab_coverage": vocab_coverage,
            "hapax_legomena": hapax_legomena,
            "mean_freq": mean_freq,
            "median_freq": median_freq,
            "max_freq": max_freq,
            "entropy": entropy,
        }
