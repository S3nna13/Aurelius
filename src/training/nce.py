"""
Noise Contrastive Estimation (NCE) and InfoNCE for the Aurelius LLM project.

This module provides InfoNCE/SimCSE-style contrastive losses for training
better text embeddings, including in-batch negatives, hard negative mining,
and a full NCEEmbeddingTrainer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NCEConfig:
    """Configuration for Noise Contrastive Estimation embedding training."""

    temperature: float = 0.07
    """InfoNCE temperature (lower = sharper distribution)."""

    n_negatives: int = 7
    """Number of negatives per positive sample."""

    embedding_dim: int = 64
    """Projected embedding dimension."""

    pooling: str = "mean"
    """Pooling strategy: 'mean' | 'last' | 'cls'."""

    normalize: bool = True
    """Whether to L2-normalize embeddings."""


# ---------------------------------------------------------------------------
# InfoNCE loss (explicit negatives)
# ---------------------------------------------------------------------------


def info_nce_loss(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE loss with explicit negatives.

    Loss = -log(exp(sim(a,p)/T) / (exp(sim(a,p)/T) + sum_k exp(sim(a,n_k)/T)))

    Uses cosine similarity. Returns mean loss scalar.

    Args:
        anchors:   (B, D) anchor embeddings.
        positives: (B, D) positive embeddings.
        negatives: (B, K, D) negative embeddings.
        temperature: InfoNCE temperature.

    Returns:
        Scalar loss tensor.
    """
    # Positive cosine similarity: (B,)
    sim_pos = F.cosine_similarity(anchors, positives, dim=-1)  # (B,)

    # Negative similarities via bmm: (B, K)
    anchors_norm = F.normalize(anchors, dim=-1)  # (B, D)
    negs_norm = F.normalize(negatives, dim=-1)  # (B, K, D)
    # (B, K, D) x (B, D, 1) -> (B, K, 1) -> (B, K)
    sim_neg = torch.bmm(negs_norm, anchors_norm.unsqueeze(-1)).squeeze(-1)

    # Scale by temperature
    sim_pos_scaled = sim_pos / temperature  # (B,)
    sim_neg_scaled = sim_neg / temperature  # (B, K)

    # loss_i = logsumexp([sim_pos, sim_neg_0..K]) - sim_pos
    logits = torch.cat([sim_pos_scaled.unsqueeze(1), sim_neg_scaled], dim=1)  # (B, K+1)
    loss = torch.logsumexp(logits, dim=1) - sim_pos_scaled  # (B,)
    return loss.mean()


# ---------------------------------------------------------------------------
# In-batch InfoNCE loss (SimCSE-style)
# ---------------------------------------------------------------------------


def in_batch_nce_loss(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """In-batch negatives InfoNCE (SimCSE-style).

    Similarity matrix = embeddings_a @ embeddings_b.T / temperature.
    Cross-entropy loss with diagonal targets.

    Args:
        embeddings_a: (B, D) anchor embeddings.
        embeddings_b: (B, D) positive embeddings; other items are negatives.
        temperature:  InfoNCE temperature.

    Returns:
        Scalar loss tensor.
    """
    a = F.normalize(embeddings_a, dim=-1)  # (B, D)
    b = F.normalize(embeddings_b, dim=-1)  # (B, D)

    logits = torch.matmul(a, b.T) / temperature  # (B, B)

    B = embeddings_a.shape[0]
    targets = torch.arange(B, device=embeddings_a.device)

    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------


def hard_negative_mining(
    anchor: Tensor,
    candidates: Tensor,
    top_k: int,
    exclude_idx: int | None = None,
) -> Tensor:
    """Return indices of top-k most similar (hardest) negative candidates.

    Args:
        anchor:      (D,) anchor embedding.
        candidates:  (N, D) candidate embeddings.
        top_k:       Number of hard negatives to return.
        exclude_idx: Index to exclude (e.g., the known positive).

    Returns:
        (top_k,) LongTensor of candidate indices.
    """
    a_norm = F.normalize(anchor.unsqueeze(0), dim=-1)  # (1, D)
    c_norm = F.normalize(candidates, dim=-1)  # (N, D)
    sims = (c_norm @ a_norm.T).squeeze(-1)  # (N,)

    if exclude_idx is not None:
        sims = sims.clone()
        sims[exclude_idx] = float("-inf")

    _, indices = torch.topk(sims, k=top_k)
    return indices


# ---------------------------------------------------------------------------
# Random negative sampling
# ---------------------------------------------------------------------------


def sample_random_negatives(
    batch_size: int,
    n_negatives: int,
    vocab_size: int,
    seq_len: int,
    rng: random.Random | None = None,
) -> Tensor:
    """Sample random negative token sequences.

    Args:
        batch_size:  Number of sequences in the batch.
        n_negatives: Number of negatives per sequence.
        vocab_size:  Vocabulary size; tokens sampled from [0, vocab_size).
        seq_len:     Length of each sampled sequence.
        rng:         Optional Python random.Random for reproducibility.

    Returns:
        (batch_size, n_negatives, seq_len) LongTensor.
    """
    if rng is not None:
        seed = rng.randint(0, 2**31 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randint(
            0,
            vocab_size,
            (batch_size, n_negatives, seq_len),
            generator=generator,
            dtype=torch.long,
        )
    return torch.randint(0, vocab_size, (batch_size, n_negatives, seq_len), dtype=torch.long)


# ---------------------------------------------------------------------------
# Embedding projector
# ---------------------------------------------------------------------------


class EmbeddingProjector(nn.Module):
    """Project backbone hidden states to a lower-dimensional embedding space.

    Architecture: Linear(d_model, embed_dim) + LayerNorm(embed_dim), then L2-normalize.
    """

    def __init__(self, d_model: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalize embeddings.

        Args:
            x: (B, D) pooled hidden states.

        Returns:
            (B, embed_dim) L2-normalized embeddings.
        """
        out = self.linear(x)  # (B, embed_dim)
        out = self.norm(out)  # (B, embed_dim)
        out = F.normalize(out, dim=-1)  # unit norm
        return out


# ---------------------------------------------------------------------------
# NCE Embedding Trainer
# ---------------------------------------------------------------------------


class NCEEmbeddingTrainer:
    """Train embeddings using noise contrastive estimation."""

    def __init__(
        self,
        backbone: nn.Module,
        projector: EmbeddingProjector,
        cfg: NCEConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.backbone = backbone
        self.projector = projector
        self.cfg = cfg
        self.optimizer = optimizer

    def get_embeddings(self, input_ids: Tensor) -> Tensor:
        """Extract and project embeddings from the backbone.

        Uses a forward hook on backbone.layers[-1] to capture hidden states,
        pools per cfg.pooling, and projects via the projector.

        Args:
            input_ids: (B, T) token IDs.

        Returns:
            (B, embed_dim) embeddings, L2-normalized if cfg.normalize is True.
        """
        captured: list[Tensor] = []

        def _hook(module, inp, output):
            # TransformerBlock returns (hidden_states, kv_cache)
            if isinstance(output, tuple):
                captured.append(output[0])
            else:
                captured.append(output)

        handle = self.backbone.layers[-1].register_forward_hook(_hook)
        try:
            _ = self.backbone(input_ids)
        finally:
            handle.remove()

        hidden = captured[0]  # (B, T, D)

        if self.cfg.pooling == "mean":
            pooled = hidden.mean(dim=1)  # (B, D)
        elif self.cfg.pooling == "last":
            pooled = hidden[:, -1, :]  # (B, D)
        elif self.cfg.pooling == "cls":
            pooled = hidden[:, 0, :]  # (B, D)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.cfg.pooling!r}")

        # Project — EmbeddingProjector always L2-normalizes
        embeddings = self.projector(pooled)  # (B, embed_dim)
        return embeddings

    def train_step_in_batch(
        self,
        anchor_ids: Tensor,
        positive_ids: Tensor,
    ) -> dict[str, float]:
        """Compute in-batch NCE loss, backward, step optimizer.

        Args:
            anchor_ids:   (B, T) anchor token IDs.
            positive_ids: (B, T) positive token IDs.

        Returns:
            Dict with keys "loss", "mean_similarity", "alignment".
        """
        self.backbone.train()
        self.projector.train()
        self.optimizer.zero_grad()

        emb_a = self.get_embeddings(anchor_ids)  # (B, E)
        emb_p = self.get_embeddings(positive_ids)  # (B, E)

        loss = in_batch_nce_loss(emb_a, emb_p, temperature=self.cfg.temperature)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            a_norm = F.normalize(emb_a.detach(), dim=-1)
            p_norm = F.normalize(emb_p.detach(), dim=-1)

            sim_matrix = torch.matmul(a_norm, p_norm.T)  # (B, B)
            mean_sim = sim_matrix.mean().item()
            alignment = sim_matrix.diagonal().mean().item()

        return {
            "loss": loss.item(),
            "mean_similarity": mean_sim,
            "alignment": alignment,
        }

    def evaluate_retrieval(
        self,
        query_ids: Tensor,
        corpus_ids: Tensor,
        top_k: int = 5,
    ) -> dict[str, float]:
        """Compute embedding-based retrieval metrics.

        For each query i, finds the top-k most similar corpus items.
        Assumes query i matches corpus i (identity mapping).

        Args:
            query_ids:  (Q, T) query token IDs.
            corpus_ids: (C, T) corpus token IDs.
            top_k:      Number of top results to retrieve.

        Returns:
            Dict with keys "recall@1", "recall@5", "mrr".
        """
        self.backbone.eval()
        self.projector.eval()

        with torch.no_grad():
            q_emb = self.get_embeddings(query_ids)  # (Q, E)
            c_emb = self.get_embeddings(corpus_ids)  # (C, E)

            q_norm = F.normalize(q_emb, dim=-1)
            c_norm = F.normalize(c_emb, dim=-1)

            sim = torch.matmul(q_norm, c_norm.T)  # (Q, C)

            Q = query_ids.shape[0]
            recall_at_1 = 0.0
            recall_at_5 = 0.0
            mrr = 0.0

            for i in range(Q):
                ranked = torch.argsort(sim[i], descending=True).tolist()
                gold = i

                if ranked[0] == gold:
                    recall_at_1 += 1.0
                if gold in ranked[:5]:
                    recall_at_5 += 1.0

                rank = ranked.index(gold) + 1
                mrr += 1.0 / rank

        return {
            "recall@1": recall_at_1 / Q,
            "recall@5": recall_at_5 / Q,
            "mrr": mrr / Q,
        }
