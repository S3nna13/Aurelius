"""Attention analysis: head importance ranking, clustering, and induction head detection.

Provides extended analysis components complementing attention_patterns.py:
- Head importance ranking by specialization (inverse entropy)
- Attention head clustering via k-means on (entropy, mean_distance) features
- Induction head detection via sub-diagonal attention heuristics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionAnalysisConfig:
    """Configuration for attention head analysis."""

    n_heads: int = 2
    n_layers: int = 2
    induction_threshold: float = 0.5
    cluster_k: int = 3


@dataclass
class AttentionPattern:
    """Analysis result for a single attention head."""

    layer: int
    head: int
    pattern_type: str  # "local" | "global" | "induction" | "diagonal"
    entropy: float
    mean_distance: float


def extract_attention_weights(
    model,
    input_ids: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    """Extract per-head attention weight matrices from all layers.

    Uses forward hooks on each transformer layer to capture hidden states,
    then synthesizes attention-like matrices via softmax(h @ h.T / sqrt(D))
    for each head slice.

    Args:
        model: AureliusTransformer instance.
        input_ids: (B, T) token IDs.

    Returns:
        Dict mapping (layer, head) -> Tensor of shape (1, T, T).
    """
    hidden_states: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            # output is (hidden, kv_cache_tuple); capture hidden state
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states[layer_idx] = h.detach()

        return hook_fn

    for i, layer in enumerate(model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Build per-head attention matrices from hidden states
    result: dict[tuple[int, int], torch.Tensor] = {}

    for layer_idx, h in hidden_states.items():
        # h: (B, T, D)
        B, T, D = h.shape
        getattr(model, "config", None)
        # Determine number of heads from config or default to 2
        n_heads = 2  # default; will be updated from model config
        try:
            n_heads = model.config.n_heads
        except AttributeError:
            try:
                # Try accessing layers config attributes
                n_heads = model.layers[layer_idx].attn.n_heads
            except AttributeError:
                pass

        head_dim = D // n_heads

        for head_idx in range(n_heads):
            # Slice head dimension
            start = head_idx * head_dim
            end = start + head_dim
            h_head = h[:, :, start:end]  # (B, T, head_dim)

            # Compute attention-like matrix: softmax(Q @ K^T / sqrt(d))
            scale = math.sqrt(head_dim)
            scores = torch.bmm(h_head, h_head.transpose(1, 2)) / scale  # (B, T, T)
            attn = F.softmax(scores, dim=-1)  # (B, T, T)

            result[(layer_idx, head_idx)] = attn

    return result


def compute_head_entropy(attn: torch.Tensor) -> float:
    """Compute mean entropy across all positions for a (T, T) attention matrix.

    Entropy is normalized by log(T) so that uniform attention gives ~1.0.

    Args:
        attn: (T, T) attention matrix (rows sum to 1).

    Returns:
        Scalar float — mean normalized entropy.
    """
    # attn may be (1, T, T) or (T, T)
    if attn.dim() == 3:
        attn = attn.squeeze(0)  # (T, T)
    T = attn.shape[-1]
    eps = 1e-9
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (T,) per-row entropy
    mean_entropy = entropy.mean().item()
    log_T = math.log(T) if T > 1 else 1.0
    return mean_entropy / log_T


def compute_mean_attention_distance(attn: torch.Tensor) -> float:
    """Compute mean |j - i| weighted by attention weights.

    Args:
        attn: (T, T) attention matrix where attn[i, j] = weight from i to j.
              May also be (1, T, T).

    Returns:
        Scalar float — mean weighted position distance.
    """
    if attn.dim() == 3:
        attn = attn.squeeze(0)  # (T, T)
    T = attn.shape[-1]
    # Build distance matrix |j - i|
    positions = torch.arange(T, dtype=attn.dtype, device=attn.device)
    i_idx = positions.unsqueeze(1)  # (T, 1)
    j_idx = positions.unsqueeze(0)  # (1, T)
    dist = (j_idx - i_idx).abs()  # (T, T)
    # Weighted mean distance per row, then overall mean
    weighted = (attn * dist).sum(dim=-1)  # (T,)
    return weighted.mean().item()


def classify_attention_pattern(attn: torch.Tensor) -> str:
    """Classify a (T, T) attention matrix into a pattern type.

    Priority order: diagonal > induction > local > global

    Args:
        attn: (T, T) or (1, T, T) attention matrix.

    Returns:
        One of: "diagonal", "induction", "local", "global".
    """
    if attn.dim() == 3:
        attn = attn.squeeze(0)
    T = attn.shape[-1]

    # Diagonal: main diagonal has high average weight
    diag_vals = torch.diagonal(attn)
    avg_diag = diag_vals.mean().item()
    if avg_diag > 0.5:
        return "diagonal"

    # Induction: check for high sub-diagonal values (off-diagonal periodic)
    # Check sub-diagonals for offsets n=1..T//2
    best_subdiag = 0.0
    for n in range(2, max(3, T // 2 + 1)):
        # Diagonal offset -n: attn[i, i-n] for i >= n
        if T > n:
            subdiag = torch.diagonal(attn, offset=-n)
            if subdiag.numel() > 0:
                val = subdiag.mean().item()
                if val > best_subdiag:
                    best_subdiag = val
    if best_subdiag > 0.3 and T > 4:
        return "induction"

    # Local: mean position distance < T/4
    mean_dist = compute_mean_attention_distance(attn)
    if mean_dist < T / 4:
        return "local"

    # Global: fairly uniform (high entropy)
    log_T = math.log(T) if T > 1 else 1.0
    eps = 1e-9
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
    if entropy > 0.9 * log_T:
        return "global"

    return "local"


def detect_induction_heads(
    attn_weights: dict[tuple[int, int], torch.Tensor],
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Detect induction heads based on sub-diagonal attention pattern.

    An induction head attends to position i-n for some fixed offset n > 1.
    We check whether any sub-diagonal of the attention matrix has mean
    weight above the threshold.

    Args:
        attn_weights: Dict mapping (layer, head) -> (B, T, T) or (T, T) tensor.
        threshold: Sub-diagonal mean weight threshold.

    Returns:
        List of (layer, head) tuples identified as induction heads.
    """
    induction_heads = []

    for (layer, head), attn in attn_weights.items():
        if attn.dim() == 3:
            attn = attn.squeeze(0)  # (T, T)
        T = attn.shape[-1]
        if T <= 3:
            continue

        is_induction = False
        for n in range(2, max(3, T // 2 + 1)):
            if T > n:
                subdiag = torch.diagonal(attn, offset=-n)
                if subdiag.numel() > 0:
                    val = subdiag.mean().item()
                    if val > threshold:
                        is_induction = True
                        break

        if is_induction:
            induction_heads.append((layer, head))

    return induction_heads


def _kmeans(
    points: list[tuple[float, float]],
    k: int,
    n_iter: int = 100,
    seed: int = 42,
) -> list[int]:
    """Simple k-means clustering on 2D points.

    Args:
        points: List of (x, y) tuples.
        k: Number of clusters.
        n_iter: Maximum iterations.
        seed: Random seed for centroid initialization.

    Returns:
        List of cluster assignments (length = len(points)).
    """
    import random

    rng = random.Random(seed)  # noqa: S311
    n = len(points)
    k = min(k, n)

    if k == 0 or n == 0:
        return []

    # Initialize centroids by picking k random points
    indices = rng.sample(range(n), k)
    centroids = [points[i] for i in indices]

    assignments = [0] * n

    for _ in range(n_iter):
        # Assign each point to nearest centroid
        new_assignments = []
        for x, y in points:
            best_c = 0
            best_dist = float("inf")
            for c_idx, (cx, cy) in enumerate(centroids):
                dist = (x - cx) ** 2 + (y - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_c = c_idx
            new_assignments.append(best_c)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update centroids
        new_centroids = []
        for c_idx in range(k):
            members = [(x, y) for i, (x, y) in enumerate(points) if assignments[i] == c_idx]
            if members:
                mx = sum(x for x, y in members) / len(members)
                my = sum(y for x, y in members) / len(members)
                new_centroids.append((mx, my))
            else:
                # Keep old centroid if no members
                new_centroids.append(centroids[c_idx])
        centroids = new_centroids

    return assignments


def cluster_attention_heads(
    patterns: list[AttentionPattern],
    k: int = 3,
) -> dict[int, list[tuple[int, int]]]:
    """Cluster attention heads by (entropy, mean_distance) using k-means.

    Args:
        patterns: List of AttentionPattern objects (one per head).
        k: Number of clusters.

    Returns:
        Dict mapping cluster_id -> list of (layer, head) tuples.
    """
    if not patterns:
        return {}

    points = [(p.entropy, p.mean_distance) for p in patterns]
    assignments = _kmeans(points, k=k)

    clusters: dict[int, list[tuple[int, int]]] = {}
    for i, pattern in enumerate(patterns):
        c_id = assignments[i]
        clusters.setdefault(c_id, []).append((pattern.layer, pattern.head))

    return clusters


class HeadAnalyzer:
    """High-level attention head analyzer.

    Combines extraction, classification, importance ranking, and clustering.
    """

    def __init__(self, model, config: AttentionAnalysisConfig) -> None:
        self.model = model
        self.config = config

    def analyze(self, input_ids: torch.Tensor) -> list[AttentionPattern]:
        """Extract and classify all attention heads.

        Args:
            input_ids: (B, T) token IDs.

        Returns:
            List of AttentionPattern (one per head per layer).
        """
        attn_weights = extract_attention_weights(self.model, input_ids)
        patterns = []

        for (layer, head), attn in sorted(attn_weights.items()):
            # Work on the (T, T) slice
            if attn.dim() == 3:
                attn_2d = attn.squeeze(0)
            else:
                attn_2d = attn

            pattern_type = classify_attention_pattern(attn_2d)
            entropy = compute_head_entropy(attn_2d)
            mean_dist = compute_mean_attention_distance(attn_2d)

            patterns.append(
                AttentionPattern(
                    layer=layer,
                    head=head,
                    pattern_type=pattern_type,
                    entropy=entropy,
                    mean_distance=mean_dist,
                )
            )

        return patterns

    def rank_by_importance(
        self,
        patterns: list[AttentionPattern],
    ) -> list[tuple[float, int, int]]:
        """Rank heads by importance = (1 - entropy).

        Lower entropy = more specialized = more important.

        Args:
            patterns: List of AttentionPattern objects.

        Returns:
            List of (importance, layer, head) sorted descending by importance.
        """
        ranked = [(1.0 - p.entropy, p.layer, p.head) for p in patterns]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def report(self, patterns: list[AttentionPattern]) -> dict:
        """Summarize pattern type distribution.

        Args:
            patterns: List of AttentionPattern objects.

        Returns:
            Dict with keys: n_local, n_global, n_induction, n_diagonal, mean_entropy.
        """
        counts = {"local": 0, "global": 0, "induction": 0, "diagonal": 0}
        total_entropy = 0.0

        for p in patterns:
            if p.pattern_type in counts:
                counts[p.pattern_type] += 1
            total_entropy += p.entropy

        mean_entropy = total_entropy / len(patterns) if patterns else 0.0

        return {
            "n_local": counts["local"],
            "n_global": counts["global"],
            "n_induction": counts["induction"],
            "n_diagonal": counts["diagonal"],
            "mean_entropy": mean_entropy,
        }
