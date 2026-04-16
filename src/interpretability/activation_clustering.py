"""
src/interpretability/activation_clustering.py

Activation clustering utilities: discover emergent structure in transformer
hidden representations via pure-PyTorch k-means and silhouette scoring.

Pure PyTorch -- no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# ClusteringConfig
# ---------------------------------------------------------------------------

@dataclass
class ClusteringConfig:
    """Configuration for an activation clustering experiment."""
    n_clusters: int = 4        # for k-means
    n_init: int = 3            # k-means restarts
    max_iter: int = 100        # k-means iterations
    tol: float = 1e-4          # convergence threshold
    layer_idx: int = -1        # which layer to cluster
    position: str = "last"     # "last", "mean", or "all" token positions


# ---------------------------------------------------------------------------
# ClusterResult
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Result from a clustering run."""
    centroids: Tensor   # (n_clusters, d_model) cluster centers
    labels: Tensor      # (N,) cluster assignment per sample
    inertia: float      # within-cluster sum of squared distances
    n_iter: int         # iterations until convergence


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------

def pairwise_distances(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distances.

    Parameters
    ----------
    a : Tensor  shape (N, d)
    b : Tensor  shape (M, d)

    Returns
    -------
    Tensor  shape (N, M) of squared Euclidean distances.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b^T
    a_sq = (a * a).sum(dim=1, keepdim=True)    # (N, 1)
    b_sq = (b * b).sum(dim=1, keepdim=True).T  # (1, M)
    ab   = a @ b.T                              # (N, M)
    dist = a_sq + b_sq - 2.0 * ab
    # Clamp negative values caused by floating-point error
    return dist.clamp(min=0.0)


# ---------------------------------------------------------------------------
# kmeans_cluster
# ---------------------------------------------------------------------------

def kmeans_cluster(
    activations: Tensor,
    n_clusters: int,
    n_init: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> ClusterResult:
    """Pure PyTorch k-means clustering.

    Runs n_init times with random initialisation (k-means++) and returns
    the result with the lowest inertia (Lloyd's algorithm).

    Parameters
    ----------
    activations : Tensor  shape (N, d)
    n_clusters  : int     number of clusters
    n_init      : int     number of random restarts
    max_iter    : int     maximum Lloyd iterations per restart
    tol         : float   convergence tolerance (centroid shift)
    seed        : int     base RNG seed

    Returns
    -------
    ClusterResult with lowest inertia across restarts.
    """
    N, d = activations.shape
    k = min(n_clusters, N)

    best: Optional[ClusterResult] = None

    for run in range(n_init):
        rng = torch.Generator()
        rng.manual_seed(seed + run)

        # --- k-means++ initialisation ---
        idx0 = torch.randint(0, N, (1,), generator=rng).item()
        centroids = activations[int(idx0)].unsqueeze(0).clone()  # (1, d)

        for _ in range(k - 1):
            dists = pairwise_distances(activations, centroids)    # (N, k_so_far)
            min_dists = dists.min(dim=1).values                   # (N,)
            probs = min_dists / min_dists.sum().clamp(min=1e-10)
            chosen = torch.multinomial(probs, num_samples=1, generator=rng).item()
            centroids = torch.cat(
                [centroids, activations[int(chosen)].unsqueeze(0)], dim=0
            )

        # --- Lloyd iterations ---
        n_iter = 0
        for it in range(max_iter):
            # Assignment step
            dists = pairwise_distances(activations, centroids)   # (N, k)
            labels = dists.argmin(dim=1)                          # (N,)

            # Update step
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = activations[mask].mean(dim=0)
                else:
                    rand_idx = torch.randint(0, N, (1,), generator=rng).item()
                    new_centroids[c] = activations[int(rand_idx)]

            # Convergence check: max centroid shift
            shift = (new_centroids - centroids).norm(dim=1).max().item()
            centroids = new_centroids
            n_iter = it + 1
            if shift < tol:
                break

        # Compute inertia
        dists = pairwise_distances(activations, centroids)   # (N, k)
        labels = dists.argmin(dim=1)                          # (N,)
        min_sq_dists = dists[torch.arange(N), labels]
        inertia = min_sq_dists.sum().item()

        result = ClusterResult(
            centroids=centroids.clone(),
            labels=labels.clone(),
            inertia=inertia,
            n_iter=n_iter,
        )

        if best is None or inertia < best.inertia:
            best = result

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# silhouette_score
# ---------------------------------------------------------------------------

def silhouette_score(activations: Tensor, labels: Tensor) -> float:
    """Compute mean silhouette score: (b - a) / max(a, b).

    a = mean intra-cluster distance, b = mean nearest-cluster distance.

    Returns
    -------
    float in [-1, 1]. Higher = better separation.
    Pure PyTorch implementation.
    """
    N = activations.shape[0]
    unique_labels = labels.unique()
    n_clusters = unique_labels.shape[0]

    if n_clusters <= 1:
        return 0.0

    # Pairwise Euclidean distances
    sq_dists = pairwise_distances(activations, activations)  # (N, N)
    dists = sq_dists.sqrt()                                   # (N, N)

    scores = torch.zeros(N, dtype=activations.dtype)

    for i in range(N):
        ci = labels[i].item()
        cluster_mask = labels == ci

        # Intra-cluster mean distance (excluding self)
        intra_mask = cluster_mask.clone()
        intra_mask[i] = False
        intra_count = intra_mask.sum().item()

        if intra_count == 0:
            a_i = 0.0
        else:
            a_i = dists[i][intra_mask].mean().item()

        # Mean distance to each other cluster
        b_i = math.inf
        for other_label in unique_labels:
            if other_label.item() == ci:
                continue
            other_mask = labels == other_label
            mean_dist = dists[i][other_mask].mean().item()
            if mean_dist < b_i:
                b_i = mean_dist

        if math.isinf(b_i):
            scores[i] = 0.0
        else:
            denom = max(a_i, b_i)
            if denom < 1e-10:
                scores[i] = 0.0
            else:
                scores[i] = (b_i - a_i) / denom

    return scores.mean().item()


# ---------------------------------------------------------------------------
# extract_layer_activations
# ---------------------------------------------------------------------------

def extract_layer_activations(
    model: nn.Module,
    input_ids: Tensor,
    layer_idx: int = -1,
    position: str = "last",
    batch_size: int = 8,
) -> Tensor:
    """Extract activations from a specific layer.

    Uses forward hooks on model.layers[layer_idx].

    Parameters
    ----------
    model     : nn.Module with a ``layers`` ModuleList attribute.
    input_ids : Tensor  shape (N, T)
    layer_idx : int     index into model.layers (negative indices supported).
    position  : str     "last" -> (N, d_model)
                        "mean" -> (N, d_model)
                        "all"  -> (N*T, d_model)
    batch_size: int     micro-batch size for forward passes.

    Returns
    -------
    2-D Tensor (N_out, d_model).
    """
    N, T = input_ids.shape
    all_acts: List[Tensor] = []

    target_layer = model.layers[layer_idx]

    captured: List[Optional[Tensor]] = [None]

    def _hook(module: nn.Module, inp, output) -> None:
        if isinstance(output, tuple):
            captured[0] = output[0].detach()
        else:
            captured[0] = output.detach()

    hook = target_layer.register_forward_hook(_hook)

    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_ids = input_ids[start:end]

                captured[0] = None

                # AureliusTransformer.forward returns (loss, logits, present_key_values)
                _ = model(batch_ids)

                act = captured[0]
                if act is None:
                    raise RuntimeError("Hook did not capture any activation")

                # act shape: (B_local, T, d_model)
                if position == "last":
                    all_acts.append(act[:, -1, :])
                elif position == "mean":
                    all_acts.append(act.mean(dim=1))
                elif position == "all":
                    B_local = act.shape[0]
                    all_acts.append(act.reshape(B_local * T, -1))
                else:
                    raise ValueError(
                        f"Unknown position mode '{position}'. "
                        "Expected 'last', 'mean', or 'all'."
                    )
    finally:
        hook.remove()

    return torch.cat(all_acts, dim=0)


# ---------------------------------------------------------------------------
# cluster_token_sequences
# ---------------------------------------------------------------------------

def cluster_token_sequences(
    model: nn.Module,
    input_ids: Tensor,
    config: ClusteringConfig,
) -> ClusterResult:
    """Full pipeline: extract activations -> cluster -> return result.

    Parameters
    ----------
    model     : nn.Module with a ``layers`` ModuleList attribute.
    input_ids : Tensor  shape (N, T)
    config    : ClusteringConfig

    Returns
    -------
    ClusterResult
    """
    activations = extract_layer_activations(
        model,
        input_ids,
        layer_idx=config.layer_idx,
        position=config.position,
    )
    return kmeans_cluster(
        activations,
        n_clusters=config.n_clusters,
        n_init=config.n_init,
        max_iter=config.max_iter,
        tol=config.tol,
    )


# ---------------------------------------------------------------------------
# nearest_cluster_examples
# ---------------------------------------------------------------------------

def nearest_cluster_examples(
    activations: Tensor,
    result: ClusterResult,
    cluster_idx: int,
    top_k: int = 3,
) -> Tensor:
    """Return indices of top_k examples closest to the centroid of cluster_idx.

    Parameters
    ----------
    activations : Tensor  shape (N, d)
    result      : ClusterResult from kmeans_cluster
    cluster_idx : int     which cluster centroid to use
    top_k       : int     how many nearest examples to return

    Returns
    -------
    Tensor  shape (top_k,) of integer indices into activations.
    """
    centroid = result.centroids[cluster_idx].unsqueeze(0)           # (1, d)
    sq_dists = pairwise_distances(activations, centroid).squeeze(1)  # (N,)
    k = min(top_k, activations.shape[0])
    _, indices = sq_dists.topk(k, largest=False)
    return indices
