"""Knowledge graph embedding and reasoning: TransE, RotatE, entity linking for LLM grounding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class KGConfig:
    """Configuration for knowledge graph embedding training."""

    n_entities: int = 100
    n_relations: int = 10
    embedding_dim: int = 64
    scoring_fn: str = "transe"  # "transe" | "rotate" | "distmult"
    margin: float = 1.0
    neg_samples: int = 10
    learning_rate: float = 1e-3
    regularization: float = 1e-4


# ── Embedding modules ─────────────────────────────────────────────────────────


class EntityEmbeddings(nn.Module):
    """L2-normalized entity embeddings."""

    def __init__(self, n_entities: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_entities, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, ids: Tensor) -> Tensor:
        """Return L2-normalized embeddings for given entity ids."""
        raw = self.emb(ids)
        return F.normalize(raw, p=2, dim=-1)


class RelationEmbeddings(nn.Module):
    """Raw (unnormalized) relation embeddings."""

    def __init__(self, n_relations: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, ids: Tensor) -> Tensor:
        """Return raw relation embeddings (no normalization)."""
        return self.emb(ids)


# ── Scoring functions ─────────────────────────────────────────────────────────


def transe_score(h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """TransE scoring: -||h + r - t||_2.

    Args:
        h: head embeddings, shape (B, dim)
        r: relation embeddings, shape (B, dim)
        t: tail embeddings, shape (B, dim)

    Returns:
        Scores shape (B,). Higher = more plausible.
    """
    return -torch.norm(h + r - t, p=2, dim=-1)


def rotate_score(h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """RotatE scoring: entity embeddings as complex numbers, relation as rotation.

    Splits dim in half: first half = real, second half = imaginary.
    Relation treated as angles: r_re = cos(r), r_im = sin(r).
    Returns -||h_rot - t||_2, shape (B,).

    Args:
        h: head embeddings, shape (B, dim) — dim must be even
        r: relation embeddings (angles), shape (B, dim)
        t: tail embeddings, shape (B, dim) — dim must be even

    Returns:
        Scores shape (B,). Higher = more plausible.
    """
    d = h.shape[-1] // 2
    h_re, h_im = h[..., :d], h[..., d:]
    t_re, t_im = t[..., :d], t[..., d:]
    r_re = torch.cos(r[..., :d])
    r_im = torch.sin(r[..., :d])

    # Complex multiplication: h * r
    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re

    diff_re = rot_re - t_re
    diff_im = rot_im - t_im
    diff = torch.cat([diff_re, diff_im], dim=-1)
    return -torch.norm(diff, p=2, dim=-1)


def distmult_score(h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """DistMult scoring: sum(h * r * t, dim=-1).

    Args:
        h: head embeddings, shape (B, dim)
        r: relation embeddings, shape (B, dim)
        t: tail embeddings, shape (B, dim)

    Returns:
        Scores shape (B,). Higher = more plausible.
    """
    return (h * r * t).sum(dim=-1)


# ── Loss and sampling ─────────────────────────────────────────────────────────


def margin_ranking_loss(pos_scores: Tensor, neg_scores: Tensor, margin: float) -> Tensor:
    """Margin-based ranking loss.

    Loss = mean(max(0, margin - pos_scores + neg_scores)).

    Args:
        pos_scores: positive triple scores, shape (B,) or (B * n_neg,)
        neg_scores: negative triple scores, same shape as pos_scores
        margin: scalar margin

    Returns:
        Scalar loss tensor.
    """
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0.0)
    return loss.mean()


def sample_negatives(batch: Tensor, n_entities: int, n_neg: int) -> Tensor:
    """Generate negative triples by corrupting heads or tails.

    Args:
        batch: positive triples, shape (B, 3) — [head_id, rel_id, tail_id]
        n_entities: total number of entities
        n_neg: number of negatives per positive triple

    Returns:
        Negative triples, shape (B * n_neg, 3).
    """
    B = batch.shape[0]
    device = batch.device

    # Repeat each triple n_neg times: (B * n_neg, 3)
    pos_repeated = batch.repeat_interleave(n_neg, dim=0)
    negatives = pos_repeated.clone()

    total = B * n_neg
    # Randomly corrupt head (0) or tail (2)
    corrupt_tail = torch.randint(0, 2, (total,), device=device).bool()

    random_entities = torch.randint(0, n_entities, (total,), device=device)

    # Corrupt tail where corrupt_tail is True, else corrupt head
    negatives[:, 2] = torch.where(corrupt_tail, random_entities, pos_repeated[:, 2])
    negatives[:, 0] = torch.where(corrupt_tail, pos_repeated[:, 0], random_entities)

    return negatives


# ── KGTrainer ─────────────────────────────────────────────────────────────────

_SCORING_FNS = {
    "transe": transe_score,
    "rotate": rotate_score,
    "distmult": distmult_score,
}


class KGTrainer:
    """Knowledge graph embedding trainer supporting TransE, RotatE, DistMult."""

    def __init__(self, config: KGConfig) -> None:
        self.config = config

        self.entity_emb = EntityEmbeddings(config.n_entities, config.embedding_dim)
        self.relation_emb = RelationEmbeddings(config.n_relations, config.embedding_dim)

        self.optimizer = torch.optim.Adam(
            list(self.entity_emb.parameters()) + list(self.relation_emb.parameters()),
            lr=config.learning_rate,
        )

        if config.scoring_fn not in _SCORING_FNS:
            raise ValueError(
                f"Unknown scoring_fn '{config.scoring_fn}'. "
                f"Choose from {list(_SCORING_FNS.keys())}."
            )
        self._score_fn = _SCORING_FNS[config.scoring_fn]

    def score_triples(self, triples: Tensor) -> Tensor:
        """Score a batch of triples using the configured scoring function.

        Args:
            triples: shape (B, 3) — [head_id, rel_id, tail_id]

        Returns:
            Scores shape (B,).
        """
        h = self.entity_emb(triples[:, 0])
        r = self.relation_emb(triples[:, 1])
        t = self.entity_emb(triples[:, 2])
        return self._score_fn(h, r, t)

    def train_step(self, triples: Tensor) -> dict[str, float]:
        """Single training step: sample negatives, compute loss, backward, step.

        Args:
            triples: positive triples, shape (B, 3)

        Returns:
            dict with keys: "loss", "mean_pos_score", "mean_neg_score"
        """
        self.optimizer.zero_grad()

        # Positive scores
        pos_scores = self.score_triples(triples)  # (B,)

        # Negative triples and scores
        neg_triples = sample_negatives(triples, self.config.n_entities, self.config.neg_samples)
        neg_scores = self.score_triples(neg_triples)  # (B * n_neg,)

        # Expand positive scores to match negatives: (B * n_neg,)
        pos_scores_expanded = pos_scores.repeat_interleave(self.config.neg_samples)

        # Margin ranking loss
        loss = margin_ranking_loss(pos_scores_expanded, neg_scores, self.config.margin)

        # L2 regularization on embeddings
        reg = self.config.regularization * (
            self.entity_emb.emb.weight.norm(p=2) ** 2 + self.relation_emb.emb.weight.norm(p=2) ** 2
        )
        total_loss = loss + reg

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_pos_score": pos_scores.mean().item(),
            "mean_neg_score": neg_scores.mean().item(),
        }

    def predict_tail(self, head_id: int, rel_id: int, top_k: int = 5) -> Tensor:
        """Predict the most plausible tail entities for a given (head, relation) pair.

        Args:
            head_id: integer entity id
            rel_id: integer relation id
            top_k: number of top entity ids to return

        Returns:
            Tensor of top_k entity ids, shape (top_k,), sorted by descending score.
        """
        n = self.config.n_entities
        device = self.entity_emb.emb.weight.device

        all_entity_ids = torch.arange(n, device=device)
        head_ids = torch.full((n,), head_id, dtype=torch.long, device=device)
        rel_ids = torch.full((n,), rel_id, dtype=torch.long, device=device)

        triples = torch.stack([head_ids, rel_ids, all_entity_ids], dim=1)  # (n, 3)

        with torch.no_grad():
            scores = self.score_triples(triples)  # (n,)

        top_k = min(top_k, n)
        _, top_ids = torch.topk(scores, k=top_k)
        return top_ids
