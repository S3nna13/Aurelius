"""Knowledge graph construction, embedding, and retrieval for RAG-style grounding.

Provides:
- KnowledgeGraph: stores (head, relation, tail) triples and supports BFS retrieval
- TransEEmbedder: TransE knowledge graph embedding (h + r ≈ t)
- KGRetriever: retrieves relevant triples for a text query via entity string matching
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KGConfig:
    embed_dim: int = 64
    n_relations: int = 8
    margin: float = 1.0       # TransE margin loss margin
    norm: int = 2             # L-norm for distance
    negative_k: int = 4      # negative samples per positive


# ---------------------------------------------------------------------------
# Triple
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    head: str
    relation: str
    tail: str


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Store and index (head, relation, tail) triples."""

    def __init__(self) -> None:
        self._triples: list[Triple] = []
        # head -> list of (relation, tail)
        self._adjacency: dict[str, list[tuple[str, str]]] = {}
        self._entities: set[str] = set()
        self._relations: set[str] = set()

    def add_triple(self, head: str, relation: str, tail: str) -> None:
        triple = Triple(head=head, relation=relation, tail=tail)
        self._triples.append(triple)
        self._adjacency.setdefault(head, []).append((relation, tail))
        self._entities.add(head)
        self._entities.add(tail)
        self._relations.add(relation)

    def get_neighbors(self, entity: str) -> list[tuple[str, str]]:
        """Return [(relation, tail)] for all triples with this head."""
        return list(self._adjacency.get(entity, []))

    def get_entities(self) -> list[str]:
        return sorted(self._entities)

    def get_relations(self) -> list[str]:
        return sorted(self._relations)

    def __len__(self) -> int:
        return len(self._triples)

    def sample_negative(self, triple: Triple, rng: random.Random) -> Triple:
        """Corrupt head or tail randomly."""
        entities = self.get_entities()
        if rng.random() < 0.5:
            # corrupt head
            new_head = rng.choice(entities)
            return Triple(head=new_head, relation=triple.relation, tail=triple.tail)
        else:
            # corrupt tail
            new_tail = rng.choice(entities)
            return Triple(head=triple.head, relation=triple.relation, tail=new_tail)


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_entity_index(kg: KnowledgeGraph) -> dict[str, int]:
    """Map entity strings to integer indices (sorted for determinism)."""
    return {entity: idx for idx, entity in enumerate(kg.get_entities())}


def build_relation_index(kg: KnowledgeGraph) -> dict[str, int]:
    """Map relation strings to integer indices (sorted for determinism)."""
    return {relation: idx for idx, relation in enumerate(kg.get_relations())}


# ---------------------------------------------------------------------------
# TransE Embedder
# ---------------------------------------------------------------------------

class TransEEmbedder(nn.Module):
    """TransE knowledge graph embedding: h + r ≈ t."""

    def __init__(self, n_entities: int, n_relations: int, embed_dim: int) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
        # Normalize entity embeddings at init
        nn.init.uniform_(self.entity_embeddings.weight, -6 / (embed_dim ** 0.5), 6 / (embed_dim ** 0.5))
        nn.init.uniform_(self.relation_embeddings.weight, -6 / (embed_dim ** 0.5), 6 / (embed_dim ** 0.5))

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """TransE score: -||h_emb + r_emb - t_emb||_2. Shape (B,) -> (B,)."""
        h_emb = self.entity_embeddings(h)   # (B, D)
        r_emb = self.relation_embeddings(r) # (B, D)
        t_emb = self.entity_embeddings(t)   # (B, D)
        diff = h_emb + r_emb - t_emb       # (B, D)
        return -torch.norm(diff, p=2, dim=-1)  # (B,)

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Return score tensor."""
        return self.score(h, r, t)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def transe_margin_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    margin: float,
) -> Tensor:
    """max(0, margin - pos_score + neg_score), mean over batch."""
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0.0)
    return loss.mean()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_by_entity(
    kg: KnowledgeGraph,
    query_entity: str,
    max_hops: int = 1,
) -> list[Triple]:
    """BFS up to max_hops, return all reachable triples. No duplicates."""
    visited_entities: set[str] = set()
    seen_triples: set[tuple[str, str, str]] = set()
    result: list[Triple] = []

    queue: deque[tuple[str, int]] = deque()
    queue.append((query_entity, 0))
    visited_entities.add(query_entity)

    while queue:
        entity, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for relation, tail in kg.get_neighbors(entity):
            triple_key = (entity, relation, tail)
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                result.append(Triple(head=entity, relation=relation, tail=tail))
            if tail not in visited_entities:
                visited_entities.add(tail)
                queue.append((tail, depth + 1))

    return result


def format_triples_as_text(triples: list[Triple]) -> str:
    """Format as 'head relation tail\\n' for each triple."""
    return "".join(f"{t.head} {t.relation} {t.tail}\n" for t in triples)


# ---------------------------------------------------------------------------
# KGRetriever
# ---------------------------------------------------------------------------

class KGRetriever:
    """Retrieve relevant triples for a text query using entity string matching."""

    def __init__(self, kg: KnowledgeGraph, max_hops: int = 1) -> None:
        self.kg = kg
        self.max_hops = max_hops

    def retrieve(self, query: str, top_k: int = 5) -> list[Triple]:
        """Find entities mentioned in query (simple substring match against kg.get_entities()),
        then retrieve_by_entity for each, deduplicate, return up to top_k triples."""
        query_lower = query.lower()
        matched_entities = [
            entity for entity in self.kg.get_entities()
            if entity.lower() in query_lower
        ]

        seen: set[tuple[str, str, str]] = set()
        result: list[Triple] = []

        for entity in matched_entities:
            for triple in retrieve_by_entity(self.kg, entity, self.max_hops):
                key = (triple.head, triple.relation, triple.tail)
                if key not in seen:
                    seen.add(key)
                    result.append(triple)
                    if len(result) >= top_k:
                        return result

        return result

    def format_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve then format_triples_as_text."""
        triples = self.retrieve(query, top_k=top_k)
        return format_triples_as_text(triples)
