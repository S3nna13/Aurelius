"""TransE and RotatE knowledge graph embedding models for KG-augmented RAG.

References:
    - TransE: Bordes et al. 2013, "Translating Embeddings for Modeling Multi-relational Data"
    - RotatE: Sun et al. 2019, "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KGEConfig:
    """Configuration for knowledge graph embedding models."""

    n_entities: int
    n_relations: int
    embed_dim: int = 256
    margin: float = 1.0          # margin for ranking loss
    norm: int = 1                # L1 or L2 norm for TransE
    epsilon: float = 1e-12       # RotatE numerical stability
    regularization: float = 0.0  # L2 weight regularization


# ---------------------------------------------------------------------------
# TransE
# ---------------------------------------------------------------------------


class TransE(nn.Module):
    """TransE: h + r ≈ t  (Bordes et al. 2013).

    Score function: -||h + r - t||_p   (negative distance, higher = better)
    Loss: max-margin ranking: max(0, margin + score(h,r,t') - score(h,r,t))
        equivalently: max(0, margin - score(h,r,t) + score(h,r,t_neg))
    """

    def __init__(self, cfg: KGEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.entity_embed = nn.Embedding(cfg.n_entities, cfg.embed_dim)
        self.relation_embed = nn.Embedding(cfg.n_relations, cfg.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.entity_embed.weight, -6 / math.sqrt(self.cfg.embed_dim), 6 / math.sqrt(self.cfg.embed_dim))
        nn.init.uniform_(self.relation_embed.weight, -6 / math.sqrt(self.cfg.embed_dim), 6 / math.sqrt(self.cfg.embed_dim))
        # Normalize entity embeddings to unit sphere at init
        with torch.no_grad():
            self.entity_embed.weight.div_(self.entity_embed.weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12))

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Compute TransE scores for a batch of triples.

        Args:
            h: (B,) integer entity indices for heads
            r: (B,) integer relation indices
            t: (B,) integer entity indices for tails

        Returns:
            (B,) scores — negative L_p distance (higher = more plausible)
        """
        h_emb = self.entity_embed(h)     # (B, D)
        r_emb = self.relation_embed(r)   # (B, D)
        t_emb = self.entity_embed(t)     # (B, D)

        # Normalize entity embeddings during forward (TransE convention)
        h_emb = F.normalize(h_emb, p=2, dim=-1)
        t_emb = F.normalize(t_emb, p=2, dim=-1)

        diff = h_emb + r_emb - t_emb    # (B, D)
        dist = diff.norm(p=self.cfg.norm, dim=-1)  # (B,)
        return -dist

    def loss(self, h: Tensor, r: Tensor, t: Tensor, t_neg: Tensor) -> Tensor:
        """Margin ranking loss.

        Args:
            h:     (B,) head entity indices
            r:     (B,) relation indices
            t:     (B,) true tail entity indices
            t_neg: (B,) corrupted (negative) tail entity indices

        Returns:
            Scalar loss tensor.
        """
        pos_score = self.score(h, r, t)        # (B,) — should be high (close to 0)
        neg_score = self.score(h, r, t_neg)    # (B,) — should be low (very negative)

        # margin + neg_score - pos_score: penalise when neg is not worse than pos by margin
        margin_loss = F.relu(self.cfg.margin + neg_score - pos_score)
        loss = margin_loss.mean()

        if self.cfg.regularization > 0.0:
            h_emb = self.entity_embed(h)
            r_emb = self.relation_embed(r)
            t_emb = self.entity_embed(t)
            t_neg_emb = self.entity_embed(t_neg)
            reg = (h_emb.norm(p=2, dim=-1) ** 2
                   + r_emb.norm(p=2, dim=-1) ** 2
                   + t_emb.norm(p=2, dim=-1) ** 2
                   + t_neg_emb.norm(p=2, dim=-1) ** 2).mean()
            loss = loss + self.cfg.regularization * reg

        return loss


# ---------------------------------------------------------------------------
# RotatE
# ---------------------------------------------------------------------------


class RotatE(nn.Module):
    """RotatE: relation = rotation in complex space (Sun et al. 2019).

    Entities are represented as complex vectors of dimension embed_dim/2
    (stored as real tensors of shape embed_dim, interpreted as pairs).
    Relations are unit complex numbers (phases only).

    Score: -||h ∘ r - t||  where ∘ is element-wise complex multiplication.
    """

    def __init__(self, cfg: KGEConfig) -> None:
        super().__init__()
        if cfg.embed_dim % 2 != 0:
            raise ValueError(
                f"RotatE requires an even embed_dim (got {cfg.embed_dim}). "
                "Entities are complex vectors: embed_dim real values = embed_dim/2 complex numbers."
            )
        self.cfg = cfg
        self.half_dim = cfg.embed_dim // 2

        # Entity embeddings stored as real; interpret as complex pairs
        self.entity_embed = nn.Embedding(cfg.n_entities, cfg.embed_dim)
        # Relation embeddings are phases (angles), one per complex dimension
        self.relation_phases = nn.Embedding(cfg.n_relations, self.half_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.entity_embed.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_phases.weight, -math.pi, math.pi)

    def _to_complex(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Split real tensor (..., embed_dim) into real and imag parts (..., half_dim)."""
        re = x[..., : self.half_dim]
        im = x[..., self.half_dim :]
        return re, im

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Compute RotatE scores for a batch of triples.

        Args:
            h: (B,) integer entity indices for heads
            r: (B,) integer relation indices
            t: (B,) integer entity indices for tails

        Returns:
            (B,) scores (negative L2 distance in complex space, higher = better)
        """
        h_emb = self.entity_embed(h)      # (B, D)
        t_emb = self.entity_embed(t)      # (B, D)
        phases = self.relation_phases(r)  # (B, D/2) — angles

        # Build unit complex rotation: r = cos(phase) + i*sin(phase)
        r_re = torch.cos(phases)  # (B, D/2)
        r_im = torch.sin(phases)  # (B, D/2)

        # Decompose head into real/imag
        h_re, h_im = self._to_complex(h_emb)  # each (B, D/2)
        t_re, t_im = self._to_complex(t_emb)

        # Element-wise complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Distance: ||h ∘ r - t||
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        dist = (diff_re ** 2 + diff_im ** 2 + self.cfg.epsilon).sqrt().sum(dim=-1)  # (B,)
        return -dist

    def loss(self, h: Tensor, r: Tensor, t: Tensor, t_neg: Tensor) -> Tensor:
        """Self-adversarial negative sampling loss (simplified, Sun et al. 2019).

        Loss = -log σ(score(pos) - margin) - log σ(margin - score(neg))

        Args:
            h:     (B,) head entity indices
            r:     (B,) relation indices
            t:     (B,) true tail entity indices
            t_neg: (B,) corrupted tail entity indices

        Returns:
            Scalar loss tensor.
        """
        pos_score = self.score(h, r, t)      # (B,)
        neg_score = self.score(h, r, t_neg)  # (B,)

        pos_loss = -F.logsigmoid(pos_score - self.cfg.margin)
        neg_loss = -F.logsigmoid(self.cfg.margin - neg_score)
        loss = (pos_loss + neg_loss).mean()

        if self.cfg.regularization > 0.0:
            h_emb = self.entity_embed(h)
            t_emb = self.entity_embed(t)
            t_neg_emb = self.entity_embed(t_neg)
            reg = (h_emb.norm(p=2, dim=-1) ** 2
                   + t_emb.norm(p=2, dim=-1) ** 2
                   + t_neg_emb.norm(p=2, dim=-1) ** 2).mean()
            loss = loss + self.cfg.regularization * reg

        return loss


# ---------------------------------------------------------------------------
# KGRetriever
# ---------------------------------------------------------------------------


class KGRetriever:
    """Retrieve nearest entities from embedding space given a query vector.

    Supports both TransE and RotatE models. Uses cosine similarity against
    the precomputed entity embedding table.
    """

    def __init__(self, model: TransE | RotatE) -> None:
        self.model = model
        self._entity_embeds: Tensor | None = None  # (n_entities, D), L2-normalised

    @torch.no_grad()
    def build_index(self) -> None:
        """Precompute and L2-normalize all entity embeddings."""
        weight = self.model.entity_embed.weight  # (n_entities, D)
        self._entity_embeds = F.normalize(weight, p=2, dim=-1)  # (n_entities, D)

    @torch.no_grad()
    def retrieve(self, query_embeds: Tensor, top_k: int = 10) -> tuple[Tensor, Tensor]:
        """Retrieve top-k entities for each query vector using cosine similarity.

        Args:
            query_embeds: (B, embed_dim) — e.g. projected LLM hidden states
            top_k: number of nearest neighbours to return

        Returns:
            entity_ids: (B, top_k) — entity indices sorted by descending similarity
            scores:     (B, top_k) — cosine similarity scores
        """
        if self._entity_embeds is None:
            self.build_index()

        # L2-normalize queries
        q = F.normalize(query_embeds, p=2, dim=-1)  # (B, D)

        # Cosine similarity: (B, n_entities)
        sim = q @ self._entity_embeds.T

        # Top-k
        scores, entity_ids = sim.topk(top_k, dim=-1, largest=True, sorted=True)
        return entity_ids, scores


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------


def negative_sample(
    n_entities: int,
    batch_h: Tensor,
    batch_t: Tensor,
    n_neg: int = 1,
) -> tuple[Tensor, Tensor]:
    """Sample random negative head and tail entity indices.

    Each true (h, t) pair is replicated n_neg times and corrupted by
    replacing h or t with a random entity (uniformly sampled). The
    implementation makes a best-effort attempt to avoid sampling the
    true entity, but does not guarantee it for simplicity.

    Args:
        n_entities: total number of entities in the KG
        batch_h:    (B,) true head entity indices
        batch_t:    (B,) true tail entity indices
        n_neg:      number of negatives per positive

    Returns:
        neg_h: (B * n_neg,) corrupted head indices
        neg_t: (B * n_neg,) corrupted tail indices
    """
    b = batch_h.shape[0]
    device = batch_h.device

    # Expand true pairs n_neg times
    h_rep = batch_h.repeat_interleave(n_neg)  # (B * n_neg,)
    t_rep = batch_t.repeat_interleave(n_neg)  # (B * n_neg,)

    total = b * n_neg

    # Sample random replacements
    rand_entities = torch.randint(0, n_entities, (total,), device=device)

    # Randomly decide whether to corrupt head or tail for each sample
    corrupt_head = torch.rand(total, device=device) < 0.5

    neg_h = torch.where(corrupt_head, rand_entities, h_rep)
    neg_t = torch.where(corrupt_head, t_rep, rand_entities)

    return neg_h, neg_t
