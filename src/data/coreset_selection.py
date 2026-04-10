"""Coreset / subset selection methods for data pruning."""

import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class CoresetConfig:
    n_select: int = 100
    method: str = "greedy"       # "greedy" | "kcenter" | "random"
    distance: str = "cosine"     # "cosine" | "l2"
    seed: int = 42


# ---------------------------------------------------------------------------
# Pairwise distance helpers
# ---------------------------------------------------------------------------

def pairwise_cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise cosine similarities.

    Args:
        a: (M, D) float tensor
        b: (N, D) float tensor

    Returns:
        (M, N) cosine similarity matrix
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)  # (M, D)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)  # (N, D)
    return a_norm @ b_norm.T                                 # (M, N)


def pairwise_l2(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise L2 distances.

    Args:
        a: (M, D) float tensor
        b: (N, D) float tensor

    Returns:
        (M, N) L2 distance matrix
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a_sq = (a * a).sum(dim=-1, keepdim=True)   # (M, 1)
    b_sq = (b * b).sum(dim=-1, keepdim=True)   # (N, 1)
    dot = a @ b.T                              # (M, N)
    dist_sq = a_sq + b_sq.T - 2.0 * dot
    # Clamp to avoid negative values from floating-point error
    return dist_sq.clamp(min=0.0).sqrt()       # (M, N)


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def greedy_coreset(
    embeddings: Tensor,
    n_select: int,
    distance: str = "cosine",
) -> Tensor:
    """Greedy farthest-point sampling (greedy coreset construction).

    Iteratively adds the point that is farthest from the current selected set.

    Args:
        embeddings: (N, D) float tensor
        n_select:   number of points to select
        distance:   "cosine" (uses 1 - similarity as distance) or "l2"

    Returns:
        selected: LongTensor of shape (n_select,) — indices into embeddings
    """
    n = embeddings.shape[0]
    n_select = min(n_select, n)

    # Seed with index 0 deterministically (caller can shuffle before calling)
    selected = [0]

    # min-distance of each point to the current set; init to +inf
    min_dist = torch.full((n,), float("inf"), dtype=embeddings.dtype,
                          device=embeddings.device)

    for _ in range(n_select - 1):
        # Distances from the last added point to all points
        last = embeddings[selected[-1]].unsqueeze(0)  # (1, D)
        if distance == "cosine":
            # Convert similarity → distance: 1 - cos_sim
            sim = pairwise_cosine_sim(last, embeddings).squeeze(0)  # (N,)
            d = 1.0 - sim
        else:
            d = pairwise_l2(last, embeddings).squeeze(0)            # (N,)

        # Update running min-distance to selected set
        min_dist = torch.minimum(min_dist, d)

        # Zero out already-selected so they won't be picked again
        for idx in selected:
            min_dist[idx] = -1.0

        selected.append(int(min_dist.argmax().item()))

    return torch.tensor(selected, dtype=torch.long, device=embeddings.device)


def kcenter_coreset(
    embeddings: Tensor,
    n_select: int,
    seed: int = 42,
) -> Tensor:
    """K-center approximation via greedy farthest-point sampling.

    Uses a fixed random seed to pick the initial point, then delegates to the
    same farthest-point algorithm as greedy_coreset.

    Args:
        embeddings: (N, D) float tensor
        n_select:   number of cluster centres to select
        seed:       RNG seed for reproducible initial point

    Returns:
        selected: LongTensor of shape (n_select,)
    """
    n = embeddings.shape[0]
    n_select = min(n_select, n)

    rng = torch.Generator(device=embeddings.device)
    rng.manual_seed(seed)
    start = int(torch.randint(n, (1,), generator=rng).item())

    selected = [start]
    min_dist = torch.full((n,), float("inf"), dtype=embeddings.dtype,
                          device=embeddings.device)

    for _ in range(n_select - 1):
        last = embeddings[selected[-1]].unsqueeze(0)  # (1, D)
        d = pairwise_l2(last, embeddings).squeeze(0)  # (N,)
        min_dist = torch.minimum(min_dist, d)
        for idx in selected:
            min_dist[idx] = -1.0
        selected.append(int(min_dist.argmax().item()))

    return torch.tensor(selected, dtype=torch.long, device=embeddings.device)


def random_coreset(
    embeddings: Tensor,
    n_select: int,
    seed: int = 42,
) -> Tensor:
    """Random subset selection without replacement.

    Args:
        embeddings: (N, D) float tensor (only N is used)
        n_select:   number of points to select
        seed:       RNG seed for reproducibility

    Returns:
        selected: LongTensor of shape (n_select,)
    """
    n = embeddings.shape[0]
    n_select = min(n_select, n)
    g = torch.Generator(device=embeddings.device)
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=embeddings.device)
    return perm[:n_select]


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

class CoresetSelector:
    """Unified interface for coreset / subset selection.

    Usage::

        cfg = CoresetConfig(n_select=50, method="greedy", distance="cosine")
        selector = CoresetSelector(cfg)
        indices = selector.select(embeddings)          # LongTensor (50,)
        subset  = selector.get_subset(data, embeddings)  # list of 50 items
    """

    def __init__(self, config: CoresetConfig | None = None):
        self.config = config or CoresetConfig()

    def select(self, embeddings: Tensor) -> Tensor:
        """Select a coreset from *embeddings*.

        Args:
            embeddings: (N, D) float tensor

        Returns:
            LongTensor of shape (n_select,) — indices into the N examples
        """
        cfg = self.config
        if cfg.method == "greedy":
            return greedy_coreset(embeddings, cfg.n_select, cfg.distance)
        elif cfg.method == "kcenter":
            return kcenter_coreset(embeddings, cfg.n_select, cfg.seed)
        elif cfg.method == "random":
            return random_coreset(embeddings, cfg.n_select, cfg.seed)
        else:
            raise ValueError(
                f"Unknown coreset method '{cfg.method}'. "
                "Choose from 'greedy', 'kcenter', 'random'."
            )

    def get_subset(self, data: list, embeddings: Tensor) -> list:
        """Return the coreset as a list of data items.

        Args:
            data:       list of N items (e.g. sentences, dicts, tensors)
            embeddings: (N, D) float tensor

        Returns:
            List of n_select items corresponding to selected indices
        """
        indices = self.select(embeddings)
        return [data[i] for i in indices.tolist()]
