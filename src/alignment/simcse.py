"""SimCSE contrastive learning for better semantic embeddings.

Unsupervised SimCSE: pass each sentence through the model twice with different
dropout masks -> two views. In-batch negatives provide contrastive signal.

Loss: cross-entropy over (N, N) cosine similarity matrix with temperature scaling.
Diagonal entries are positives (same sentence, different dropout); off-diagonal are negatives.

Reference: Gao et al. 2021 "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
           arXiv:2104.08821
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimCSEConfig:
    temperature: float = 0.05  # contrastive temperature (lower = sharper)
    pooling: str = "mean"  # "mean", "last", "max"
    normalize: bool = True  # L2-normalize embeddings before cosine sim


def simcse_loss(
    embeddings_a: torch.Tensor,  # (N, D) -- first view
    embeddings_b: torch.Tensor,  # (N, D) -- second view
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss.

    Positive pairs: (embeddings_a[i], embeddings_b[i]) -- same sentence, diff dropout
    Negative pairs: all cross-pairs (i, j) where i != j

    Loss = cross_entropy of cosine similarity matrix, where label i = i (diagonal).

    Args:
        embeddings_a: (N, D) L2-normalized embeddings, first forward pass
        embeddings_b: (N, D) L2-normalized embeddings, second forward pass
        temperature: Scaling factor for similarities

    Returns:
        Scalar loss
    """
    # Normalize both views
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)

    # (N, N) similarity matrix; diagonal = positive pairs
    sim = a @ b.T / temperature  # (N, N)

    # Labels: for row i, the positive is column i
    labels = torch.arange(sim.size(0), device=sim.device)

    return F.cross_entropy(sim, labels)


def extract_embeddings(
    model: nn.Module,
    input_ids: torch.Tensor,  # (B, S)
    cfg: SimCSEConfig,
) -> torch.Tensor:
    """Extract pooled embeddings from the model.

    Hooks into model.norm (the final RMSNorm before lm_head) to get hidden states.
    Applies pooling according to cfg.pooling. L2-normalizes if cfg.normalize.

    Returns: (B, D) embeddings
    """
    hidden_state: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        hidden_state.append(output)

    hook = model.norm.register_forward_hook(hook_fn)
    try:
        model(input_ids)
    finally:
        hook.remove()

    hidden = hidden_state[0]  # (B, S, D)

    if cfg.pooling == "mean":
        emb = hidden.mean(dim=1)  # (B, D)
    elif cfg.pooling == "last":
        emb = hidden[:, -1, :]  # (B, D)
    elif cfg.pooling == "max":
        emb = hidden.max(dim=1).values  # (B, D)
    else:
        raise ValueError(f"Unknown pooling strategy: {cfg.pooling!r}")

    if cfg.normalize:
        emb = F.normalize(emb, dim=-1)

    return emb


class SimCSETrainer:
    """Contrastive embedding trainer using unsupervised SimCSE.

    Training loop: for each batch, run two forward passes (same input, dropout
    stays on), compute SimCSE loss, backprop.

    The model must have dropout > 0 for the two views to differ.
    """

    def __init__(
        self,
        model: nn.Module,  # AureliusTransformer
        cfg: SimCSEConfig | None = None,
        lr: float = 3e-5,
    ) -> None:
        self.model = model
        self.cfg = cfg or SimCSEConfig()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(
        self,
        input_ids: torch.Tensor,  # (B, S)
    ) -> float:
        """One gradient step. Returns loss value."""
        self.model.train()
        self.optimizer.zero_grad()

        # Two forward passes with dropout active -> two different views
        emb_a = extract_embeddings(self.model, input_ids, self.cfg)
        emb_b = extract_embeddings(self.model, input_ids, self.cfg)

        loss = simcse_loss(emb_a, emb_b, self.cfg.temperature)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def encode(
        self,
        input_ids: torch.Tensor,  # (B, S)
    ) -> torch.Tensor:
        """Encode inputs to embeddings (inference mode, no grad)."""
        self.model.eval()
        with torch.no_grad():
            return extract_embeddings(self.model, input_ids, self.cfg)
