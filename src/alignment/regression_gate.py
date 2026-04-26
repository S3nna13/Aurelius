"""Regression gate for DoRA adapter quality control.

Computes perplexity before and after adapter application.
Rejects adapters that regress beyond a configurable threshold.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.transformer import AureliusTransformer


@dataclass
class GateResult:
    """Result of a regression gate check."""

    accepted: bool
    baseline_ppl: float
    new_ppl: float
    regression_pct: float  # positive = regression, negative = improvement
    reason: str


class RegressionGate:
    """Gate that rejects adapters with perplexity regression above threshold.

    Args:
        threshold_pct: Maximum allowed perplexity regression in percent.
                       E.g., 5.0 means reject if new_ppl > baseline_ppl * 1.05.
        batch_size: Batch size for perplexity evaluation.
    """

    def __init__(self, threshold_pct: float = 5.0, batch_size: int = 4) -> None:
        self.threshold_pct = threshold_pct
        self.batch_size = batch_size

    def compute_perplexity(
        self,
        model: AureliusTransformer,
        dataset: Dataset,
        device: str = "cpu",
    ) -> float:
        """Compute perplexity of model on dataset.

        Args:
            model: The model to evaluate.
            dataset: Dataset returning (input_ids, labels) pairs.
            device: Device to run evaluation on.

        Returns:
            Perplexity = exp(mean cross-entropy).
        """
        model.eval()
        model.to(device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for input_ids, labels in loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                loss, logits, _ = model(input_ids, labels=labels)
                # loss is mean CE over non-ignored tokens
                # We need token-weighted average for correct PPL
                n_tokens = labels.numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        mean_ce = total_loss / total_tokens
        return math.exp(mean_ce)

    def check(
        self,
        baseline_model: AureliusTransformer,
        new_model: AureliusTransformer,
        dataset: Dataset,
        adapter_path: str | Path | None = None,
        device: str = "cpu",
    ) -> GateResult:
        """Compare baseline and new model perplexity, accept or reject adapter.

        Args:
            baseline_model: The original model without adapter.
            new_model: The model with the adapter applied.
            dataset: Held-out evaluation dataset.
            adapter_path: Optional path to archive rejected adapter checkpoint.
            device: Device to run evaluation on.

        Returns:
            GateResult with acceptance decision and metrics.
        """
        baseline_ppl = self.compute_perplexity(baseline_model, dataset, device)
        new_ppl = self.compute_perplexity(new_model, dataset, device)

        regression_pct = (new_ppl - baseline_ppl) / baseline_ppl * 100.0
        accepted = regression_pct <= self.threshold_pct

        if accepted:
            reason = (
                f"Accepted: regression {regression_pct:.1f}% <= threshold {self.threshold_pct:.1f}%"
            )
        else:
            reason = (
                f"Rejected: regression {regression_pct:.1f}% > threshold {self.threshold_pct:.1f}%"
            )
            if adapter_path is not None:
                self._archive_adapter(Path(adapter_path))
                reason += f" (archived to {adapter_path}.rejected)"

        return GateResult(
            accepted=accepted,
            baseline_ppl=baseline_ppl,
            new_ppl=new_ppl,
            regression_pct=regression_pct,
            reason=reason,
        )

    @staticmethod
    def _archive_adapter(adapter_path: Path) -> None:
        """Move a rejected adapter to <path>.rejected."""
        dest = adapter_path.with_suffix(".rejected")
        if adapter_path.is_dir():
            shutil.move(str(adapter_path), str(dest))
        elif adapter_path.exists():
            adapter_path.rename(dest)
