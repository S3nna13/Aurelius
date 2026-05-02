"""Lloyd-Max optimal scalar quantizer for the Beta(2,2) distribution.

Used by PolarQuant (TurboQuant Stage 1) to quantize normalized KV cache vectors.
Beta(2,2) is the distribution of values after per-vector min-max normalization
to [0, 1] (approximately), motivated by the empirical KV cache distribution.

Closed-form centroids — no scipy required:
  PDF:  p(t) = 6t(1-t),   t in [0, 1]
  CDF:  F(t) = 3t^2 - 2t^3
  Antiderivative of t*p(t):  2t^3 - (3/2)t^4
  Antiderivative of p(t):    3t^2 - 2t^3
"""

from __future__ import annotations

import torch

# Module-level cache: n_codes -> codebook tensor
_CODEBOOK_CACHE: dict[int, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# Closed-form Beta(2,2) integrals
# ---------------------------------------------------------------------------


def _beta22_cdf(t: float) -> float:
    """CDF of Beta(2,2): F(t) = 3t^2 - 2t^3."""
    return 3.0 * t**2 - 2.0 * t**3


def _beta22_antiderivative_numerator(t: float) -> float:
    """Antiderivative of t * p(t) for Beta(2,2): 2t^3 - (3/2)t^4."""
    return 2.0 * t**3 - 1.5 * t**4


def _beta22_antiderivative_denominator(t: float) -> float:
    """Antiderivative of p(t) for Beta(2,2): 3t^2 - 2t^3."""
    return 3.0 * t**2 - 2.0 * t**3


def _centroid(lo: float, hi: float) -> float:
    """Optimal centroid for a quantization cell [lo, hi] under Beta(2,2).

    Centroid = integral(t * p(t), lo, hi) / integral(p(t), lo, hi)
             = [A_num(hi) - A_num(lo)] / [A_den(hi) - A_den(lo)]
    """
    num = _beta22_antiderivative_numerator(hi) - _beta22_antiderivative_numerator(lo)
    den = _beta22_antiderivative_denominator(hi) - _beta22_antiderivative_denominator(lo)
    if den < 1e-12:
        return (lo + hi) / 2.0  # fallback for empty cells
    return num / den


# ---------------------------------------------------------------------------
# Lloyd-Max iteration
# ---------------------------------------------------------------------------


def compute_lloyd_max_codebook(n_codes: int, n_iter: int = 100) -> torch.Tensor:
    """Compute the Lloyd-Max quantization codebook for Beta(2,2).

    Uses closed-form centroid updates — no numerical integration required.
    Result is cached at module level after the first call.

    Args:
        n_codes: Number of quantization levels (e.g. 256 for 8-bit).
        n_iter: Number of Lloyd-Max iterations (default 100 is more than enough).

    Returns:
        Sorted centroids of shape (n_codes,) with values in [0, 1].
    """
    if n_codes in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[n_codes]

    # Initialize centroids uniformly
    centroids = [(i + 0.5) / n_codes for i in range(n_codes)]

    for _ in range(n_iter):
        # Decision boundaries: midpoints between adjacent centroids
        boundaries = (
            [0.0] + [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_codes - 1)] + [1.0]
        )

        # Update centroids using closed-form Beta(2,2) integrals
        new_centroids = []
        for i in range(n_codes):
            lo, hi = boundaries[i], boundaries[i + 1]
            new_centroids.append(_centroid(lo, hi))

        # Check convergence
        max_delta = max(abs(new_centroids[i] - centroids[i]) for i in range(n_codes))
        centroids = new_centroids
        if max_delta < 1e-10:
            break

    result = torch.tensor(centroids, dtype=torch.float32)
    _CODEBOOK_CACHE[n_codes] = result
    return result
