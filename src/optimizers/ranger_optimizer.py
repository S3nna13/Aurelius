"""
src/optimizers/ranger_optimizer.py

Ranger — RAdam + LookAhead optimizer wrapper (Lester 2019).

Pure-PyTorch implementation.  Combines RAdam's variance-rectified adaptive
learning rates with LookAhead's slow-weight synchronisation for smoother
convergence.

Usage:
    base = RAdam(model.parameters(), lr=1e-3)
    opt = Ranger(base, k=6, alpha=0.5)
"""

from __future__ import annotations

import copy

import torch
from torch.optim import Optimizer

from .lookahead import Lookahead
from .radam import RAdam


class Ranger(Lookahead):
    """Ranger: RAdam wrapped inside a LookAhead synchroniser.

    Args:
        params:               Iterable of parameters or parameter groups.
        lr:                   Learning rate passed to the inner RAdam.
                              Default: ``1e-3``.
        betas:                ``(beta1, beta2)`` for RAdam.
                              Default: ``(0.95, 0.999)``.
        eps:                  Numerical stabiliser.  Default: ``1e-6``.
        weight_decay:         Decoupled weight-decay coefficient.
                              Default: ``0.0``.
        k:                    LookAhead synchronisation frequency.
                              Default: ``6``.
        alpha:                LookAhead slow-weight interpolation factor.
                              Default: ``0.5``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        k: int = 6,
        alpha: float = 0.5,
    ) -> None:
        base = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(base, k=k, alpha=alpha)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def __repr__(self) -> str:
        return (
            f"Ranger(RAdam(lr={self.base_optimizer.param_groups[0]['lr']}, "
            f"betas={self.base_optimizer.param_groups[0]['betas']}), "
            f"LookAhead(k={self.k}, alpha={self.alpha}))"
        )
