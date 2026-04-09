"""Dense vector retrieval with product quantization (PQ) for memory-efficient
approximate nearest neighbor search. Pure PyTorch — no external FAISS dependency."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from torch import Tensor


@dataclass
class DenseRetrievalConfig:
    embed_dim: int = 64
    n_subspaces: int = 4        # PQ subspaces (embed_dim must be divisible)
    n_centroids: int = 16       # centroids per subspace (codebook size)
    n_probe: int = 4            # IVF: number of clusters to probe
    n_clusters: int = 8         # IVF: number of top-level clusters
    metric: str = "cosine"      # "cosine" | "l2"


def compute_similarity(
    query: Tensor,      # (D,) or (Q, D)
    corpus: Tensor,     # (N, D)
    metric: str = "cosine",
) -> Tensor:
    """Compute similarity between query and all corpus vectors.

    cosine: normalized dot product. l2: negative squared L2 distance.
    Returns (Q, N) or (N,) if query is 1D.
    """
    squeeze = query.dim() == 1
    if squeeze:
        query = query.unsqueeze(0)  # (1, D)

    if metric == "cosine":
        q_norm = torch.nn.functional.normalize(query, dim=-1)    # (Q, D)
        c_norm = torch.nn.functional.normalize(corpus, dim=-1)   # (N, D)
        sim = q_norm @ c_norm.T                                   # (Q, N)
    elif metric == "l2":
        # negative squared L2 distance: -(||q - c||^2)
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
        q_sq = (query ** 2).sum(dim=-1, keepdim=True)   # (Q, 1)
        c_sq = (corpus ** 2).sum(dim=-1).unsqueeze(0)   # (1, N)
        dot = query @ corpus.T                           # (Q, N)
        sim = -(q_sq + c_sq - 2 * dot)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'cosine' or 'l2'.")

    if squeeze:
        return sim.squeeze(0)  # (N,)
    return sim  # (Q, N)


def kmeans_cluster(
    vectors: Tensor,    # (N, D)
    k: int,
    n_iter: int = 10,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Simple Lloyd's k-means.

    Returns (centroids, assignments): centroids shape (k, D), assignments (N,).
    Initialize with random subset of vectors.
    Handles degenerate case where k > n by clamping k to n.
    """
    n, d = vectors.shape
    k = min(k, n)  # handle degenerate case

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Initialize centroids as random subset of vectors
    perm = torch.randperm(n, generator=rng)[:k]
    centroids = vectors[perm].clone().float()  # (k, D)

    vectors_f = vectors.float()

    for _ in range(n_iter):
        # Assign each vector to nearest centroid
        # distances: (N, k) — use negative L2 as similarity
        dists = torch.cdist(vectors_f, centroids)  # (N, k)
        assignments = dists.argmin(dim=1)           # (N,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.float32)
        for i in range(n):
            new_centroids[assignments[i]] += vectors_f[i]
            counts[assignments[i]] += 1.0

        # Avoid division by zero for empty clusters — keep old centroid
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = centroids[j]

        centroids = new_centroids

    # Final assignment
    dists = torch.cdist(vectors_f, centroids)
    assignments = dists.argmin(dim=1)

    return centroids, assignments


class ProductQuantizer:
    """Product quantization for memory-efficient vector compression."""

    def __init__(self, cfg: DenseRetrievalConfig) -> None:
        self.cfg = cfg
        assert cfg.embed_dim % cfg.n_subspaces == 0, \
            f"embed_dim ({cfg.embed_dim}) must be divisible by n_subspaces ({cfg.n_subspaces})"
        self.sub_dim = cfg.embed_dim // cfg.n_subspaces
        # codebooks: list of n_subspaces tensors, each (n_centroids, sub_dim) after fit
        self.codebooks: list[Tensor] = []

    def fit(self, vectors: Tensor) -> None:
        """Fit n_subspaces codebooks on split vectors using kmeans_cluster."""
        vectors_f = vectors.float()
        self.codebooks = []
        for s in range(self.cfg.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            sub = vectors_f[:, start:end]  # (N, sub_dim)
            centroids, _ = kmeans_cluster(
                sub,
                k=self.cfg.n_centroids,
                seed=42 + s,
            )
            self.codebooks.append(centroids)

    def encode(self, vectors: Tensor) -> Tensor:
        """Encode vectors to (N, n_subspaces) integer codes."""
        assert len(self.codebooks) > 0, "Call fit() before encode()."
        vectors_f = vectors.float()
        n = vectors_f.shape[0]
        codes = torch.zeros(n, self.cfg.n_subspaces, dtype=torch.long)
        for s in range(self.cfg.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            sub = vectors_f[:, start:end]                      # (N, sub_dim)
            cb = self.codebooks[s]                              # (n_centroids, sub_dim)
            dists = torch.cdist(sub, cb)                       # (N, n_centroids)
            codes[:, s] = dists.argmin(dim=1)
        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """Decode (N, n_subspaces) codes back to (N, embed_dim) approximate vectors."""
        assert len(self.codebooks) > 0, "Call fit() before decode()."
        n = codes.shape[0]
        result = torch.zeros(n, self.cfg.embed_dim)
        for s in range(self.cfg.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            cb = self.codebooks[s]          # (n_centroids, sub_dim)
            idx = codes[:, s]               # (N,)
            result[:, start:end] = cb[idx]
        return result

    def asymmetric_distance(self, query: Tensor, codes: Tensor) -> Tensor:
        """Compute distances between query (D,) and encoded corpus (N, n_subspaces).

        For each subspace s: distance[n] += ||query_sub_s - codebook_s[codes[n,s]]||^2
        Returns (N,) distances (lower = more similar).
        """
        assert len(self.codebooks) > 0, "Call fit() before asymmetric_distance()."
        query_f = query.float()
        n = codes.shape[0]
        distances = torch.zeros(n)
        for s in range(self.cfg.n_subspaces):
            start = s * self.sub_dim
            end = start + self.sub_dim
            q_sub = query_f[start:end]      # (sub_dim,)
            cb = self.codebooks[s]           # (n_centroids, sub_dim)
            idx = codes[:, s]               # (N,)
            centroid_vecs = cb[idx]          # (N, sub_dim)
            diff = q_sub.unsqueeze(0) - centroid_vecs  # (N, sub_dim)
            distances += (diff ** 2).sum(dim=1)
        return distances


class IVFIndex:
    """Inverted File Index for coarse-to-fine search."""

    def __init__(self, cfg: DenseRetrievalConfig) -> None:
        self.cfg = cfg
        self.coarse_centroids: Tensor | None = None
        self.inverted_lists: dict[int, list[int]] = {}
        self._vectors: Tensor | None = None
        self._ids: list[int] = []

    def build(self, vectors: Tensor, ids: list[int] | None = None) -> None:
        """Build IVF: cluster vectors into cfg.n_clusters coarse centroids,
        assign each vector to its nearest centroid.
        """
        n = vectors.shape[0]
        if ids is None:
            ids = list(range(n))

        self._vectors = vectors.float()
        self._ids = list(ids)

        k = min(self.cfg.n_clusters, n)
        centroids, assignments = kmeans_cluster(vectors, k=k)
        self.coarse_centroids = centroids  # (k, D)

        self.inverted_lists = {i: [] for i in range(k)}
        for vec_idx, cluster_idx in enumerate(assignments.tolist()):
            self.inverted_lists[cluster_idx].append(vec_idx)

    def search(self, query: Tensor, top_k: int) -> tuple[Tensor, list[int]]:
        """Two-stage search:
        1. Find cfg.n_probe nearest coarse centroids to query
        2. Within those clusters, compute exact distances to all candidates
        3. Return top-k (scores, ids)
        """
        assert self.coarse_centroids is not None, "Call build() before search()."

        query_f = query.float()
        if query_f.dim() > 1:
            query_f = query_f.squeeze(0)

        # Stage 1: find n_probe nearest coarse centroids
        n_probe = min(self.cfg.n_probe, self.coarse_centroids.shape[0])
        coarse_dists = torch.cdist(
            query_f.unsqueeze(0), self.coarse_centroids
        ).squeeze(0)  # (k,)
        probe_indices = coarse_dists.argsort()[:n_probe].tolist()

        # Stage 2: gather candidates from probed clusters
        candidate_vec_indices: list[int] = []
        for cluster_idx in probe_indices:
            candidate_vec_indices.extend(self.inverted_lists.get(cluster_idx, []))

        # Deduplicate while preserving order
        seen = set()
        unique_candidates: list[int] = []
        for idx in candidate_vec_indices:
            if idx not in seen:
                seen.add(idx)
                unique_candidates.append(idx)

        if not unique_candidates:
            # Fall back: search all
            unique_candidates = list(range(len(self._ids)))

        # Exact distance computation for candidates
        candidate_vecs = self._vectors[unique_candidates]  # (C, D)
        scores_vec = compute_similarity(query_f, candidate_vecs, metric=self.cfg.metric)

        # Select top-k from candidates
        k = min(top_k, len(unique_candidates))
        top_local_idx = scores_vec.argsort(descending=True)[:k]

        top_scores = scores_vec[top_local_idx]
        top_ids = [self._ids[unique_candidates[i]] for i in top_local_idx.tolist()]

        return top_scores, top_ids

    def __len__(self) -> int:
        """Total number of indexed vectors."""
        return len(self._ids)


class DenseRetriever:
    """End-to-end dense retrieval system."""

    def __init__(self, cfg: DenseRetrievalConfig) -> None:
        self.cfg = cfg
        self._ivf: IVFIndex | None = None
        self._corpus_vectors: Tensor | None = None
        self._corpus_texts: list[str] | None = None

    def index(self, corpus_vectors: Tensor, corpus_texts: list[str] | None = None) -> None:
        """Build IVF index on corpus_vectors. Store corpus_texts for retrieval."""
        self._corpus_vectors = corpus_vectors.float()
        self._corpus_texts = corpus_texts
        self._ivf = IVFIndex(self.cfg)
        self._ivf.build(corpus_vectors)

    def search(self, query: Tensor, top_k: int = 5) -> list[dict]:
        """Search for top-k similar vectors.

        Returns list of {"score": float, "id": int, "text": str | None}
        """
        assert self._ivf is not None, "Call index() before search()."
        scores, ids = self._ivf.search(query, top_k=top_k)

        results = []
        for score, vid in zip(scores.tolist(), ids):
            text = None
            if self._corpus_texts is not None and 0 <= vid < len(self._corpus_texts):
                text = self._corpus_texts[vid]
            results.append({"score": float(score), "id": int(vid), "text": text})
        return results

    def exact_search(self, query: Tensor, top_k: int = 5) -> list[dict]:
        """Brute-force exact search (for comparison/small corpora)."""
        assert self._corpus_vectors is not None, "Call index() before exact_search()."

        q = query.float()
        if q.dim() > 1:
            q = q.squeeze(0)

        scores = compute_similarity(q, self._corpus_vectors, metric=self.cfg.metric)
        k = min(top_k, self._corpus_vectors.shape[0])
        top_idx = scores.argsort(descending=True)[:k]

        results = []
        for idx in top_idx.tolist():
            text = None
            if self._corpus_texts is not None and 0 <= idx < len(self._corpus_texts):
                text = self._corpus_texts[idx]
            results.append({
                "score": float(scores[idx].item()),
                "id": int(idx),
                "text": text,
            })
        return results

    def evaluate_recall(
        self,
        query_vectors: Tensor,      # (Q, D)
        ground_truth_ids: list[int],  # correct id for each query
        top_k: int = 10,
    ) -> float:
        """Recall@k: fraction of queries where correct id is in top-k results."""
        assert self._ivf is not None, "Call index() before evaluate_recall()."
        n_queries = query_vectors.shape[0]
        hits = 0
        for q_idx in range(n_queries):
            q = query_vectors[q_idx]
            results = self.search(q, top_k=top_k)
            retrieved_ids = {r["id"] for r in results}
            if ground_truth_ids[q_idx] in retrieved_ids:
                hits += 1
        return hits / n_queries if n_queries > 0 else 0.0
