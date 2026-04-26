"""Sparse gradient optimizer that skips zero-gradient parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseUpdate:
    """Metadata about a single optimiser step."""

    param_id: str
    grad_nnz: int
    grad_total: int
    applied: bool

    @property
    def sparsity(self) -> float:
        """Fraction of gradient elements that are zero (or below threshold)."""
        if self.grad_total == 0:
            return 1.0
        return 1.0 - self.grad_nnz / self.grad_total


@dataclass(frozen=True)
class SparseOptimizerConfig:
    """Configuration for the sparse SGD optimiser."""

    lr: float = 0.01
    sparsity_threshold: float = 0.0
    skip_sparse_ratio: float = 0.0


class SparseOptimizer:
    """SGD optimiser that skips updates when gradients are too sparse."""

    def __init__(self, config: SparseOptimizerConfig | None = None) -> None:
        self.config = config or SparseOptimizerConfig()
        self._total_updates: int = 0
        self._skipped: int = 0
        self._applied: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        param_id: str,
        params: list[float],
        grads: list[float],
    ) -> tuple[list[float], SparseUpdate]:
        """Apply a sparse SGD update.

        Non-zero grads are defined as ``abs(g) > sparsity_threshold``.
        If the resulting sparsity exceeds ``skip_sparse_ratio`` the update is
        skipped entirely (``applied=False``).  Otherwise SGD is applied only to
        the non-zero gradient positions.

        Returns:
            Tuple of (updated_params, SparseUpdate).
        """
        cfg = self.config
        total = len(grads)
        nnz = sum(1 for g in grads if abs(g) > cfg.sparsity_threshold)
        grad_sparsity = 1.0 - nnz / total if total > 0 else 1.0

        self._total_updates += 1

        if grad_sparsity > cfg.skip_sparse_ratio:
            # Too sparse — skip the update
            self._skipped += 1
            update = SparseUpdate(
                param_id=param_id,
                grad_nnz=nnz,
                grad_total=total,
                applied=False,
            )
            return list(params), update

        # Apply SGD only for non-zero gradient positions
        updated = list(params)
        for i, g in enumerate(grads):
            if abs(g) > cfg.sparsity_threshold:
                updated[i] -= cfg.lr * g

        self._applied += 1
        update = SparseUpdate(
            param_id=param_id,
            grad_nnz=nnz,
            grad_total=total,
            applied=True,
        )
        return updated, update

    def stats(self) -> dict[str, int]:
        """Return running statistics for this optimiser instance."""
        return {
            "total_updates": self._total_updates,
            "skipped": self._skipped,
            "applied": self._applied,
        }

    def reset_stats(self) -> None:
        """Reset all running statistics to zero."""
        self._total_updates = 0
        self._skipped = 0
        self._applied = 0


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

SPARSE_OPTIMIZER_REGISTRY: dict[str, type[SparseOptimizer]] = {
    "default": SparseOptimizer,
}
