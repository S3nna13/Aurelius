"""
Noise Contrastive Estimation Objectives — NCE / InfoNCE / NT-Xent
=================================================================
Pure-PyTorch implementations of three related contrastive self-supervised
learning objectives:

  * NCE      — Gutmann & Hyvärinen 2010 (binary real-vs-noise classifier)
  * InfoNCE  — van den Oord et al. 2018 / CPC (k-way softmax)
  * NT-Xent  — Chen et al. 2020 / SimCLR (symmetric InfoNCE over 2N views)

Alignment & uniformity diagnostics follow Wang & Isola 2020.

Registered under TRAINING_REGISTRY["nce_objectives"].
"""

from __future__ import annotations

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
    """Hyper-parameters for NCEObjectives."""

    temperature: float = 0.07   # τ — scaling for InfoNCE / NT-Xent logits
    n_negatives: int = 64       # k — number of noise samples per real sample
    normalize: bool = True      # L2-normalise embeddings before scoring


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class NCEObjectives(nn.Module):
    """
    Contrastive objectives: NCE, InfoNCE, NT-Xent.

    Parameters
    ----------
    config : NCEConfig
        Hyper-parameter bundle.
    """

    def __init__(self, config: NCEConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def maybe_normalize(self, x: Tensor) -> Tensor:
        """L2-normalise along the last dimension when config.normalize is True."""
        if self.config.normalize:
            return F.normalize(x, dim=-1)
        return x

    # ------------------------------------------------------------------
    # NCE loss
    # ------------------------------------------------------------------

    def nce_loss(
        self,
        real_scores: Tensor,   # [B]   — scores assigned to real samples
        noise_scores: Tensor,  # [B, k] — scores assigned to k noise samples
    ) -> Tensor:
        """
        Binary NCE loss (Gutmann & Hyvärinen 2010).

        L = -mean( log σ(s_real) + mean_k( log(1 - σ(s_noise_k)) ) )
          = -mean( log σ(s_real) + mean_k( log σ(-s_noise_k) ) )
        """
        # [B]   log-prob real sample is "real"
        log_real = F.logsigmoid(real_scores)
        # [B, k] log-prob noise sample is "noise"
        log_noise = F.logsigmoid(-noise_scores)
        # average over negatives, then over batch
        loss = -(log_real + log_noise.mean(dim=1)).mean()
        return loss

    # ------------------------------------------------------------------
    # InfoNCE loss
    # ------------------------------------------------------------------

    def infonce_loss(
        self,
        queries: Tensor,                       # [B, D]
        keys: Tensor,                          # [B, D]
        negatives: Tensor | None = None,       # [B, N, D] or None
    ) -> Tensor:
        """
        InfoNCE / CPC loss (van den Oord et al. 2018).

        When *negatives* is None the full batch is used as in-batch negatives:
          scores = (q @ K.T) / τ   shape [B, B]
          positive is the diagonal (i.e. label = arange(B))

        When *negatives* is provided [B, N, D]:
          For each query i the logits are [score(q_i, k_i), score(q_i, n_i_1), ...]
          shape [B, 1+N] — positive is always index 0.
        """
        queries = self.maybe_normalize(queries)
        keys = self.maybe_normalize(keys)

        if negatives is None:
            # In-batch negatives: [B, B]
            logits = torch.mm(queries, keys.t()) / self.config.temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # Explicit negatives: [B, N, D]
            negatives = self.maybe_normalize(negatives)
            B, N, D = negatives.shape
            # positive scores [B, 1]
            pos_scores = (queries * keys).sum(dim=-1, keepdim=True) / self.config.temperature
            # negative scores [B, N]
            neg_scores = torch.bmm(negatives, queries.unsqueeze(-1)).squeeze(-1) / self.config.temperature
            # concatenate [B, 1+N]; positive is at index 0
            logits = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)

        return loss

    # ------------------------------------------------------------------
    # NT-Xent loss (SimCLR)
    # ------------------------------------------------------------------

    def nt_xent_loss(
        self,
        z1: Tensor,  # [B, D] — view-1 embeddings
        z2: Tensor,  # [B, D] — view-2 embeddings
    ) -> Tensor:
        """
        NT-Xent loss (Chen et al. 2020, SimCLR).

        Concatenate [z1; z2] → Z ∈ R^{2B × D}.
        For each sample i its positive is i+B (or i-B).
        All other 2B-2 samples are negatives.
        Self-similarities (diagonal) are masked out via -inf before softmax.
        """
        z1 = self.maybe_normalize(z1)
        z2 = self.maybe_normalize(z2)
        B = z1.size(0)

        # Z: [2B, D]
        Z = torch.cat([z1, z2], dim=0)

        # Full similarity matrix [2B, 2B]
        sim = torch.mm(Z, Z.t()) / self.config.temperature

        # Mask self-similarities (diagonal) with -inf
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Positive targets:
        #   for row i in [0, B):   positive = i + B
        #   for row i in [B, 2B):  positive = i - B
        targets = torch.cat(
            [
                torch.arange(B, 2 * B, device=z1.device),
                torch.arange(0, B, device=z1.device),
            ]
        )

        loss = F.cross_entropy(sim, targets)
        return loss

    # ------------------------------------------------------------------
    # Alignment & Uniformity diagnostics (Wang & Isola 2020)
    # ------------------------------------------------------------------

    @staticmethod
    def _alignment(z1: Tensor, z2: Tensor) -> Tensor:
        """
        Alignment: -mean || z1_i - z2_i ||^2   (higher = better aligned, i.e. closer to 0).
        """
        return -((z1 - z2) ** 2).sum(dim=-1).mean()

    @staticmethod
    def _uniformity(z: Tensor) -> Tensor:
        """
        Uniformity: log( mean( exp(-2 || z_i - z_j ||^2) ) )  over all pairs.
        Lower (more negative) = more uniform distribution on the hypersphere.
        """
        # Pairwise squared distances [N, N]
        sq_dists = torch.cdist(z, z, p=2).pow(2)
        return torch.log(torch.exp(-2.0 * sq_dists).mean())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        mode: str = "nt_xent",
    ) -> dict[str, Tensor]:
        """
        Compute the chosen contrastive loss plus alignment & uniformity metrics.

        Parameters
        ----------
        z1, z2 : Tensor  [B, D]
            Paired embeddings (two augmented views of the same samples).
        mode : str
            One of ``"nt_xent"``, ``"infonce"``.

        Returns
        -------
        dict with keys:
            ``"loss"``        — scalar contrastive loss
            ``"alignment"``   — scalar alignment metric
            ``"uniformity"``  — scalar uniformity metric
        """
        z1_n = self.maybe_normalize(z1)
        z2_n = self.maybe_normalize(z2)

        if mode == "nt_xent":
            loss = self.nt_xent_loss(z1, z2)
        elif mode == "infonce":
            loss = self.infonce_loss(z1, z2)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'nt_xent' or 'infonce'.")

        alignment = self._alignment(z1_n, z2_n)
        # concatenate both views for uniformity estimate
        z_all = torch.cat([z1_n, z2_n], dim=0)
        uniformity = self._uniformity(z_all)

        return {
            "loss": loss,
            "alignment": alignment,
            "uniformity": uniformity,
        }


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["nce_objectives"] = NCEObjectives

__all__ = ["NCEConfig", "NCEObjectives"]
