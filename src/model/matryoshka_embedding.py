"""Matryoshka Representation Learning (MRL) — Kusupati et al. 2022.

Train embeddings at multiple nested granularities simultaneously.
A d=512 embedding should also be useful when truncated to d=256, d=128, d=64,
etc.  This is achieved by summing InfoNCE losses at each nested scale:

    L_MRL = Σ_{m ∈ M} w_m · L(f(x)[:m], y)

where M = [64, 128, 256, 512] are sorted nested subset sizes and f(x)[:m]
denotes the first m dimensions of the full normalised embedding.

References
----------
- Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022.
  https://arxiv.org/abs/2205.13147
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MatryoshkaConfig:
    """Configuration for :class:`MatryoshkaEmbedding`.

    Attributes:
        full_dim:      Full output embedding dimension.
        nested_dims:   Sorted list of nested subset sizes.  The last entry
                       must equal ``full_dim``.
        scale_weights: Per-scale loss weights.  ``None`` means uniform (1.0
                       for every scale).
        temperature:   Softmax temperature used in the InfoNCE loss.
    """

    full_dim: int = 512
    nested_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    scale_weights: Optional[list[float]] = None
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if not isinstance(self.full_dim, int) or isinstance(self.full_dim, bool):
            raise ValueError("full_dim must be a positive int.")
        if self.full_dim <= 0:
            raise ValueError("full_dim must be positive.")
        if not self.nested_dims:
            raise ValueError("nested_dims must be non-empty.")
        if sorted(self.nested_dims) != self.nested_dims:
            raise ValueError("nested_dims must be sorted in ascending order.")
        if self.nested_dims[-1] != self.full_dim:
            raise ValueError(
                f"Last entry of nested_dims ({self.nested_dims[-1]}) must equal "
                f"full_dim ({self.full_dim})."
            )
        for d in self.nested_dims:
            if not isinstance(d, int) or isinstance(d, bool):
                raise ValueError(f"All nested_dims must be ints, got {type(d).__name__}.")
            if d <= 0:
                raise ValueError(f"All nested_dims must be positive, got {d}.")
        if self.scale_weights is not None:
            if len(self.scale_weights) != len(self.nested_dims):
                raise ValueError(
                    "scale_weights must have the same length as nested_dims."
                )
            for w in self.scale_weights:
                if not isinstance(w, (int, float)) or isinstance(w, bool):
                    raise ValueError(
                        f"scale_weights entries must be numeric, got {type(w).__name__}."
                    )
                if not math.isfinite(float(w)) or float(w) < 0.0:
                    raise ValueError(
                        f"scale_weights entries must be finite and non-negative, got {w}."
                    )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class MatryoshkaEmbedding(nn.Module):
    """Projects input vectors to a full-dim embedding space and computes a
    Matryoshka multi-scale InfoNCE loss.

    Parameters
    ----------
    config:
        :class:`MatryoshkaConfig` instance.
    in_features:
        Dimensionality of the input vectors fed to :meth:`forward`.
    """

    def __init__(self, config: MatryoshkaConfig, in_features: int) -> None:
        super().__init__()
        if not isinstance(config, MatryoshkaConfig):
            raise TypeError(
                f"config must be a MatryoshkaConfig instance, got {type(config).__name__}"
            )
        if not isinstance(in_features, int) or isinstance(in_features, bool):
            raise TypeError("in_features must be a positive int")
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        self.config = config
        self.in_features = in_features

        # Single linear projection: in_features → full_dim
        self.projection = nn.Linear(in_features, config.full_dim, bias=False)

        # Materialise scale weights as a buffer so they're device-portable.
        if config.scale_weights is not None:
            weights = torch.tensor(config.scale_weights, dtype=torch.float32)
        else:
            weights = torch.ones(len(config.nested_dims), dtype=torch.float32)
        self.register_buffer("_scale_weights", weights)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalise input vectors.

        Parameters
        ----------
        x:
            ``[B, in_features]`` input tensor.

        Returns
        -------
        Tensor
            ``[B, full_dim]`` unit-norm embedding.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"forward expects a Tensor, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"forward expects a 2D [B, in_features] tensor, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"forward expects last dimension {self.in_features}, got {x.shape[-1]}"
            )
        z = self.projection(x)  # [B, full_dim]
        return F.normalize(z, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Nested extraction
    # ------------------------------------------------------------------

    def get_nested(self, embeddings: Tensor, dim: int) -> Tensor:
        """Slice the first *dim* dimensions and re-normalise.

        Parameters
        ----------
        embeddings:
            ``[B, full_dim]`` unit-norm embeddings (output of :meth:`forward`).
        dim:
            Number of dimensions to keep.

        Returns
        -------
        Tensor
            ``[B, dim]`` unit-norm tensor.
        """
        if not isinstance(embeddings, Tensor):
            raise TypeError(
                f"get_nested expects a Tensor, got {type(embeddings).__name__}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"get_nested expects a 2D [B, full_dim] tensor, got shape {tuple(embeddings.shape)}"
            )
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim must be a positive int")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim > embeddings.shape[-1]:
            raise ValueError(
                f"dim={dim} exceeds embedding width {embeddings.shape[-1]}"
            )
        sliced = embeddings[..., :dim]  # [B, dim]
        return F.normalize(sliced, p=2, dim=-1)

    # ------------------------------------------------------------------
    # InfoNCE loss
    # ------------------------------------------------------------------

    def infonce(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """In-batch InfoNCE (NT-Xent) loss.

        Assumes *z_a* and *z_b* are already L2-normalised.  Positive pairs are
        on the diagonal; all other pairs in the batch are negatives.

        Parameters
        ----------
        z_a, z_b:
            ``[B, d]`` unit-norm embeddings.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        if not isinstance(z_a, Tensor) or not isinstance(z_b, Tensor):
            raise TypeError("infonce expects Tensor inputs")
        if z_a.ndim != 2 or z_b.ndim != 2:
            raise ValueError("infonce expects 2D [B, d] tensors")
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"infonce expects matching shapes, got {tuple(z_a.shape)} and {tuple(z_b.shape)}"
            )
        # Cosine similarity matrix: [B, B]
        sim = torch.mm(z_a, z_b.t()) / self.config.temperature  # [B, B]
        B = z_a.size(0)
        targets = torch.arange(B, device=z_a.device)
        # Symmetric loss: average over both directions
        loss_ab = F.cross_entropy(sim, targets)
        loss_ba = F.cross_entropy(sim.t(), targets)
        return (loss_ab + loss_ba) * 0.5

    # ------------------------------------------------------------------
    # Matryoshka multi-scale loss
    # ------------------------------------------------------------------

    def matryoshka_loss(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the weighted sum of InfoNCE losses at every nested scale.

        Parameters
        ----------
        embeddings_a, embeddings_b:
            ``[B, full_dim]`` unit-norm positive-pair embeddings.

        Returns
        -------
        dict
            * ``"loss"`` — weighted total loss (scalar).
            * ``"loss_{dim}"`` — per-scale loss for each dim in nested_dims.
        """
        if not isinstance(embeddings_a, Tensor) or not isinstance(embeddings_b, Tensor):
            raise TypeError("matryoshka_loss expects Tensor inputs")
        if embeddings_a.shape != embeddings_b.shape:
            raise ValueError(
                f"matryoshka_loss expects matching shapes, got {tuple(embeddings_a.shape)} and {tuple(embeddings_b.shape)}"
            )
        losses: dict[str, Tensor] = {}
        total = torch.zeros((), device=embeddings_a.device, dtype=embeddings_a.dtype)

        for idx, dim in enumerate(self.config.nested_dims):
            e_a = self.get_nested(embeddings_a, dim)
            e_b = self.get_nested(embeddings_b, dim)
            loss_m = self.infonce(e_a, e_b)
            weight = self._scale_weights[idx]  # type: ignore[index]
            losses[f"loss_{dim}"] = loss_m
            total = total + weight * loss_m

        losses["loss"] = total
        return losses

    # ------------------------------------------------------------------
    # Diagnostic: anisotropy
    # ------------------------------------------------------------------

    def anisotropy(self, embeddings: Tensor) -> float:
        """Average pairwise cosine similarity — lower is better.

        A high value indicates *embedding space collapse* (all vectors point in
        roughly the same direction).  Ideal value is close to 0.

        Parameters
        ----------
        embeddings:
            ``[B, d]`` unit-norm embeddings.

        Returns
        -------
        float
            Scalar in [-1, 1].
        """
        if not isinstance(embeddings, Tensor):
            raise TypeError(
                f"anisotropy expects a Tensor, got {type(embeddings).__name__}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"anisotropy expects a 2D [B, d] tensor, got shape {tuple(embeddings.shape)}"
            )
        B = embeddings.size(0)
        if B < 2:
            return 0.0
        # Cosine similarity matrix [B, B]; skip diagonal (self-similarity = 1)
        sim = torch.mm(embeddings, embeddings.t())  # [B, B]
        # Zero out the diagonal
        mask = torch.ones_like(sim) - torch.eye(B, device=sim.device, dtype=sim.dtype)
        off_diag_sum = (sim * mask).sum()
        num_pairs = B * (B - 1)
        return (off_diag_sum / num_pairs).item()


# Keep the module importable through both legacy and canonical package paths.
_module = sys.modules[__name__]
sys.modules.setdefault("src.model.matryoshka_embedding", _module)
sys.modules.setdefault("model.matryoshka_embedding", _module)

__all__ = [
    "MatryoshkaConfig",
    "MatryoshkaEmbedding",
]
