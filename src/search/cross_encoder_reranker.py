"""Cross-encoder reranker: lightweight query-document scoring for result reranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CrossEncoderConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class RankedResult:
    doc_id: str
    text: str
    score: float
    rank: int


class CrossEncoderModel(nn.Module):
    """Lightweight cross-encoder for query-document scoring."""

    def __init__(self, config: CrossEncoderConfig, vocab_size: int = 32000) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        layers: list[nn.Module] = []
        for _ in range(config.n_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        x = self.mlp(x)
        return self.head(x)


class CrossEncoderReranker:
    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        config: CrossEncoderConfig | None = None,
    ) -> None:
        cfg = config if config is not None else CrossEncoderConfig()
        self.model = model if model is not None else CrossEncoderModel(cfg)
        self.model.train(False)
        self._vocab_size: int = self.model.embedding.num_embeddings

    def encode_pair(self, query: str, doc: str) -> torch.Tensor:
        words = (query + " " + doc).split()
        ids = [hash(w) % self._vocab_size for w in words[: min(32, len(words))]]
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long)

    def score(self, query: str, doc: str) -> float:
        input_ids = self.encode_pair(query, doc)
        with torch.no_grad():
            return float(self.model(input_ids).item())

    def rerank(
        self,
        query: str,
        results: list[RankedResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        scored = [(r, self.score(query, r.text)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        limit = top_k if top_k is not None else len(scored)
        out: list[RankedResult] = []
        for new_rank, (r, s) in enumerate(scored[:limit], start=1):
            out.append(RankedResult(doc_id=r.doc_id, text=r.text, score=s, rank=new_rank))
        return out

    def batch_rerank(
        self,
        queries: list[str],
        results_per_query: list[list[RankedResult]],
    ) -> list[list[RankedResult]]:
        return [self.rerank(q, results) for q, results in zip(queries, results_per_query)]


CROSS_ENCODER_RERANKER = CrossEncoderReranker()
