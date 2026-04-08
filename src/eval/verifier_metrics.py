"""Metrics for reasoning/verifier outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def verifier_precision(predicted: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Precision over binary verifier labels."""
    if predicted.shape != gold.shape:
        raise ValueError("predicted and gold must match")
    tp = ((predicted == 1) & (gold == 1)).sum().float()
    fp = ((predicted == 1) & (gold == 0)).sum().float()
    return tp / (tp + fp).clamp_min(1.0)


def verifier_recall(predicted: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Recall over binary verifier labels."""
    if predicted.shape != gold.shape:
        raise ValueError("predicted and gold must match")
    tp = ((predicted == 1) & (gold == 1)).sum().float()
    fn = ((predicted == 0) & (gold == 1)).sum().float()
    return tp / (tp + fn).clamp_min(1.0)


def verifier_f1(predicted: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """F1 score over binary verifier labels."""
    precision = verifier_precision(predicted, gold)
    recall = verifier_recall(predicted, gold)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


@dataclass(frozen=True)
class VerifierReport:
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    accuracy: torch.Tensor


def verifier_report(predicted: torch.Tensor, gold: torch.Tensor) -> VerifierReport:
    """Bundle verifier metrics into one report."""
    if predicted.shape != gold.shape:
        raise ValueError("predicted and gold must match")
    return VerifierReport(
        precision=verifier_precision(predicted, gold),
        recall=verifier_recall(predicted, gold),
        f1=verifier_f1(predicted, gold),
        accuracy=(predicted == gold).float().mean(),
    )
