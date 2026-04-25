"""HyDE query expansion embedder — arXiv 2212.10496."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HyDEConfig:
    n_hypothetical: int = 5
    embedding_dim: int = 768
    normalize: bool = True


class HypotheticalDocEmbedder:
    """HyDE: generate N hypothetical documents, mean-embed them, L2-normalize."""

    def __init__(self, config: HyDEConfig | None = None) -> None:
        self.config = config or HyDEConfig()

    def generate_hypothetical_docs(self, query: str, n: int) -> list[str]:
        return [f"Document about {query}: [detail {i}]" for i in range(n)]

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.md5(text.encode(), usedforsecurity=False).digest()
        rng = np.random.default_rng(seed=list(digest))
        vec = rng.random(self.config.embedding_dim).astype(np.float32)
        return vec

    def embed_query(self, query: str) -> np.ndarray:
        docs = self.generate_hypothetical_docs(query, self.config.n_hypothetical)
        embeddings = np.stack([self.embed(d) for d in docs])
        pooled = embeddings.mean(axis=0)
        if self.config.normalize:
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
        return pooled

    def batch_embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(q) for q in queries])
