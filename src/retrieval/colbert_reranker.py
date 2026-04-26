"""ColBERT-style late-interaction reranker with MaxSim scoring.

Uses hash-based stub token embeddings (L2-normalized) so the module is
self-contained and requires no pretrained weights. Swap encode_query /
encode_doc for real encoder outputs to get production-quality reranking.

score(q, d) = sum_i max_j cos(q_i, d_j)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from src.retrieval.reciprocal_rank_fusion import RankedDoc

__all__ = ["ColBERTConfig", "ColBERTReranker", "RankedDoc"]


@dataclass
class ColBERTConfig:
    dim: int = 128
    max_query_len: int = 32
    max_doc_len: int = 180


def _token_embedding(token: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(token.encode()).digest()
    indices = [b % dim for b in digest[:dim]]
    vec = torch.zeros(dim)
    for i, idx in enumerate(indices):
        vec[idx] += 1.0 - i * 0.001
    norm = vec.norm(p=2)
    if norm < 1e-12:
        vec[0] = 1.0
        norm = vec.norm(p=2)
    return vec / norm


def _encode_text(text: str, max_len: int, dim: int) -> torch.Tensor:
    tokens = text.lower().split()[:max_len] or ["[empty]"]
    embeddings = torch.stack([_token_embedding(t, dim) for t in tokens])
    norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return embeddings / norms


class ColBERTReranker:
    """ColBERT-style late-interaction reranker using MaxSim (stub embeddings)."""

    def __init__(self, config: ColBERTConfig | None = None) -> None:
        self.config = config or ColBERTConfig()

    def encode_query(self, query: str) -> torch.Tensor:
        return _encode_text(query, self.config.max_query_len, self.config.dim)

    def encode_doc(self, doc: str) -> torch.Tensor:
        return _encode_text(doc, self.config.max_doc_len, self.config.dim)

    def maxsim(self, Q: torch.Tensor, D: torch.Tensor) -> float:
        sim = Q @ D.T
        return sim.max(dim=1).values.sum().item()

    def rerank(
        self,
        query: str,
        docs: list[RankedDoc],
        top_k: int | None = None,
    ) -> list[RankedDoc]:
        Q = self.encode_query(query)
        scored: list[tuple[float, RankedDoc]] = []
        for doc in docs:
            D = self.encode_doc(doc.text)
            s = self.maxsim(Q, D)
            scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [
            RankedDoc(
                doc_id=d.doc_id,
                text=d.text,
                score=s,
                source="colbert_reranked",
            )
            for s, d in scored
        ]
        if top_k is not None:
            return result[:top_k]
        return result

    def batch_rerank(
        self,
        queries: list[str],
        docs_per_query: list[list[RankedDoc]],
    ) -> list[list[RankedDoc]]:
        return [self.rerank(q, docs) for q, docs in zip(queries, docs_per_query)]
