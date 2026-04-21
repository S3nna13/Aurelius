"""Coreset Selector — k-center greedy and coverage-based training data pruning.

Two selection strategies:

* **k_center** — greedy facility-location / farthest-point sampling over dense
  embeddings; minimises the maximum distance from any point to its nearest
  selected centre.
* **coverage** — greedy set-cover over token n-gram bigrams; maximises the
  number of unique n-grams covered within the budget.

Pure PyTorch only — no scipy, sklearn, or numpy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CoresetConfig:
    budget: int = 1000               # number of samples to select
    method: str = "k_center"         # "k_center" or "coverage"
    ngram_n: int = 2                 # n for n-gram coverage method
    seed: int = 42
    distance_metric: str = "cosine"  # "cosine" or "euclidean"


# ---------------------------------------------------------------------------
# CoresetSelector
# ---------------------------------------------------------------------------


class CoresetSelector:
    """Select a representative coreset from a large dataset.

    Supports two modes:

    * ``k_center``  — works on dense float embeddings (shape ``[N, D]``).
    * ``coverage``  — works on token-ID sequences (list of list of int).

    Example::

        cfg = CoresetConfig(budget=50, method="k_center")
        selector = CoresetSelector(cfg)
        indices = selector.select(embeddings=emb_tensor)
    """

    def __init__(self, config: Optional[CoresetConfig] = None) -> None:
        self.config = config or CoresetConfig()

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _pairwise_distances(self, X: Tensor) -> Tensor:
        """Compute an [N, N] pairwise distance matrix for *X* ([N, D]).

        * ``cosine``:    ``1 - (X @ X.T) / (‖x_i‖ · ‖x_j‖)``
        * ``euclidean``: ``‖x_i - x_j‖``  (via broadcast squared form)
        """
        metric = self.config.distance_metric
        if metric == "cosine":
            norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
            X_n = X / norms                        # [N, D]
            sim = X_n @ X_n.T                      # [N, N]  cosine similarities
            return (1.0 - sim).clamp(min=0.0)      # distances in [0, 2]
        elif metric == "euclidean":
            # ‖x_i - x_j‖² = ‖x_i‖² + ‖x_j‖² - 2 x_i·x_j
            sq = (X * X).sum(dim=1, keepdim=True)  # [N, 1]
            dot = X @ X.T                           # [N, N]
            dist_sq = (sq + sq.T - 2.0 * dot).clamp(min=0.0)
            return dist_sq.sqrt()                  # [N, N]
        else:
            raise ValueError(
                f"Unknown distance_metric '{metric}'. "
                "Choose 'cosine' or 'euclidean'."
            )

    # ------------------------------------------------------------------
    # k-Center greedy
    # ------------------------------------------------------------------

    def select_k_center(self, embeddings: Tensor) -> list[int]:
        """Greedy k-center / farthest-point sampling over *embeddings* [N, D].

        Returns a list of *budget* unique indices.
        """
        cfg = self.config
        N = embeddings.shape[0]
        budget = min(cfg.budget, N)

        # Seed the RNG and pick an initial point.
        rng = random.Random(cfg.seed)
        start = rng.randint(0, N - 1)

        selected: list[int] = [start]

        # min_dist[i] = distance from point i to its nearest selected centre.
        # Initialise to +inf so the first update (from the seed point) controls.
        min_dist = torch.full((N,), float("inf"),
                              dtype=embeddings.dtype,
                              device=embeddings.device)

        for _ in range(budget - 1):
            last_idx = selected[-1]
            last_emb = embeddings[last_idx].unsqueeze(0)   # [1, D]

            # Compute distances from the latest centre to all points.
            metric = cfg.distance_metric
            if metric == "cosine":
                norm_last = last_emb / last_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
                norm_all = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                d = (1.0 - (norm_last @ norm_all.T)).clamp(min=0.0).squeeze(0)
            else:  # euclidean
                sq_last = (last_emb * last_emb).sum(dim=1, keepdim=True)  # [1, 1]
                sq_all = (embeddings * embeddings).sum(dim=1, keepdim=True)  # [N, 1]
                dot = last_emb @ embeddings.T                               # [1, N]
                d = (sq_last + sq_all.T - 2.0 * dot).clamp(min=0.0).sqrt().squeeze(0)

            # Update running minimum distances.
            min_dist = torch.minimum(min_dist, d)

            # Mask already-selected points so they cannot be chosen again.
            for idx in selected:
                min_dist[idx] = -1.0

            next_idx = int(min_dist.argmax().item())
            selected.append(next_idx)

        return selected

    # ------------------------------------------------------------------
    # Coverage-based (n-gram set cover)
    # ------------------------------------------------------------------

    def _extract_ngrams(self, seq: list[int]) -> set[tuple[int, ...]]:
        """Return all n-grams of length ``self.config.ngram_n`` in *seq*."""
        n = self.config.ngram_n
        if len(seq) < n:
            return set()
        return {tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)}

    def select_coverage(self, token_sequences: list[list[int]]) -> list[int]:
        """Greedy set-cover selection maximising unique n-gram coverage.

        Returns a list of *budget* unique indices (may be fewer if the dataset
        is smaller than the budget).
        """
        cfg = self.config
        budget = min(cfg.budget, len(token_sequences))

        # Pre-compute n-grams for every sequence.
        seq_ngrams: list[set[tuple[int, ...]]] = [
            self._extract_ngrams(seq) for seq in token_sequences
        ]

        covered: set[tuple[int, ...]] = set()
        remaining = list(range(len(token_sequences)))
        selected: list[int] = []

        for _ in range(budget):
            if not remaining:
                break

            # Find the sample that adds the most new n-grams.
            best_idx = -1
            best_gain = -1
            for i in remaining:
                gain = len(seq_ngrams[i] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                # All remaining add 0 new n-grams; break early.
                break

            selected.append(best_idx)
            covered |= seq_ngrams[best_idx]
            remaining.remove(best_idx)

        return selected

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def select(
        self,
        embeddings: Optional[Tensor] = None,
        token_sequences: Optional[list[list[int]]] = None,
    ) -> list[int]:
        """Dispatch to the configured selection method.

        * ``k_center``  requires *embeddings* to be provided.
        * ``coverage``  requires *token_sequences* to be provided.
        """
        method = self.config.method
        if method == "k_center":
            if embeddings is None:
                raise ValueError(
                    "k_center method requires 'embeddings' (Tensor [N, D])."
                )
            return self.select_k_center(embeddings)
        elif method == "coverage":
            if token_sequences is None:
                raise ValueError(
                    "coverage method requires 'token_sequences' (list[list[int]])."
                )
            return self.select_coverage(token_sequences)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'k_center' or 'coverage'."
            )

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def coverage_ratio(
        self,
        selected_indices: list[int],
        token_sequences: list[list[int]],
    ) -> float:
        """Fraction of unique n-grams in the full dataset covered by the subset.

        Returns a float in ``[0.0, 1.0]``.  Returns ``0.0`` if the full
        dataset contains no n-grams.
        """
        all_ngrams: set[tuple[int, ...]] = set()
        for seq in token_sequences:
            all_ngrams |= self._extract_ngrams(seq)

        if not all_ngrams:
            return 0.0

        covered: set[tuple[int, ...]] = set()
        for i in selected_indices:
            covered |= self._extract_ngrams(token_sequences[i])

        return len(covered & all_ngrams) / len(all_ngrams)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.data import DATA_REGISTRY  # noqa: E402  (import after class definition)

DATA_REGISTRY["coreset_selector"] = CoresetSelector
