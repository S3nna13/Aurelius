"""Dense passage embedder: bi-encoder for dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy
import torch
import torch.nn as nn


@dataclass
class EmbedderConfig:
    dim: int = 768
    normalize: bool = True
    pooling: str = "mean"


class DenseEmbedder:
    """Bi-encoder dense embedder for retrieval (stub token embeddings)."""

    def __init__(
        self,
        config: EmbedderConfig | None = None,
        vocab_size: int = 32000,
    ) -> None:
        self.config = config if config is not None else EmbedderConfig()
        self._table = nn.Embedding(vocab_size, self.config.dim)
        self._vocab_size = vocab_size

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % self._vocab_size for w in text.split()] or [0]

    def embed(self, text: str) -> numpy.ndarray:
        ids = self.tokenize(text)
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            vecs = self._table(t)

        pooling = self.config.pooling
        if pooling == "cls":
            pooled = vecs[0, 0]
        elif pooling == "max":
            pooled = vecs[0].max(dim=0).values
        else:
            pooled = vecs[0].mean(dim=0)

        arr: numpy.ndarray = pooled.numpy()
        if self.config.normalize:
            norm = float(numpy.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
        return arr

    def embed_batch(self, texts: list[str]) -> numpy.ndarray:
        return numpy.stack([self.embed(t) for t in texts])

    def similarity(self, a: str, b: str) -> float:
        va = self.embed(a)
        vb = self.embed(b)
        return float(numpy.dot(va, vb))

    def top_k_similar(self, query: str, corpus: list[str], k: int = 5) -> list[tuple[int, float]]:
        if not corpus:
            return []
        qv = self.embed(query)
        scores = [(i, float(numpy.dot(qv, self.embed(doc)))) for i, doc in enumerate(corpus)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


DENSE_EMBEDDER = DenseEmbedder()
