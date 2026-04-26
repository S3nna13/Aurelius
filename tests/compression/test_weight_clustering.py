"""Tests for weight_clustering."""

from __future__ import annotations

import torch

from src.compression.weight_clustering import WeightClustering, kmeans_quantize


class TestKmeansQuantize:
    def test_quantize_roundtrip(self):
        x = torch.randn(50)
        q, centroids = kmeans_quantize(x, n_clusters=4)
        assert len(centroids) <= 4
        assert q.shape == x.shape

    def test_few_clusters(self):
        x = torch.randn(100)
        q, c = kmeans_quantize(x, 2)
        assert len(c) == 2


class TestWeightClustering:
    def test_compress_layer(self):
        m = torch.nn.Linear(16, 16)
        wc = WeightClustering(n_clusters=8)
        wc.compress(m)
        assert hasattr(m, "weight_cluster_ids")

    def test_decompress_restores_shape(self):
        m = torch.nn.Linear(8, 8)
        wc = WeightClustering(4)
        wc.compress(m)
        x = wc.decompress(m)
        assert x.shape == (8, 8)
