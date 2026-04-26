from __future__ import annotations

import torch


def kmeans_quantize(
    x: torch.Tensor, n_clusters: int = 8, n_iter: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.flatten().unsqueeze(1).float()
    idxs = torch.randint(0, n_clusters, (min(n_clusters, len(flat)),))
    centroids = flat[idxs].squeeze(1)
    for _ in range(n_iter):
        dists = (flat - centroids.unsqueeze(0)).abs()
        labels = dists.argmin(dim=1)
        for i in range(n_clusters):
            mask = labels == i
            if mask.any():
                centroids[i] = flat[mask].mean()
    quantized = centroids[labels].reshape(x.shape)
    return quantized, centroids


class WeightClustering:
    def __init__(self, n_clusters: int = 8) -> None:
        self.n_clusters = n_clusters
        self._centroids: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._shape: torch.Size | None = None

    def compress(self, module: torch.nn.Module) -> None:
        w = module.weight.data.clone()
        self._shape = w.shape
        q, centroids = kmeans_quantize(w, self.n_clusters)
        module.weight.data = q
        module.register_buffer("weight_cluster_ids", centroids)
        self._centroids = centroids

    def decompress(self, module: torch.nn.Module) -> torch.Tensor:
        if self._centroids is not None and self._shape is not None:
            return self._centroids[: self.n_clusters].mean() * torch.ones(self._shape)
        return module.weight.data
