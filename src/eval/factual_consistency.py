"""
Factual Consistency Evaluation and Hallucination Detection.

Provides NLI-based entailment scoring, token-level fact verification,
and uncertainty-based hallucination detection. Pure PyTorch only.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# NLIClassifier
# ---------------------------------------------------------------------------

class NLIClassifier(nn.Module):
    """Entailment / contradiction / neutral classifier.

    Classes:
        0 = entailment
        1 = neutral
        2 = contradiction
    """

    def __init__(self, d_model: int, n_classes: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Concatenate premise and hypothesis representations → d_model*2
        self.proj = nn.Linear(d_model * 2, d_model)
        self.act = nn.GELU()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, premise_repr: Tensor, hypothesis_repr: Tensor) -> Tensor:
        """
        Args:
            premise_repr:    (B, D)
            hypothesis_repr: (B, D)

        Returns:
            logits: (B, n_classes)
        """
        combined = torch.cat([premise_repr, hypothesis_repr], dim=-1)  # (B, 2D)
        hidden = self.act(self.proj(combined))                          # (B, D)
        logits = self.classifier(hidden)                                # (B, n_classes)
        return logits

    def entailment_score(self, premise_repr: Tensor, hypothesis_repr: Tensor) -> Tensor:
        """Return probability of entailment (class 0) for each item in batch.

        Returns:
            (B,) float tensor in (0, 1)
        """
        logits = self.forward(premise_repr, hypothesis_repr)  # (B, 3)
        probs = F.softmax(logits, dim=-1)                     # (B, 3)
        return probs[:, 0]                                    # (B,)


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------

class TextEncoder:
    """Encode text sequences to fixed-size representations.

    Args:
        model:   callable (input_ids: Tensor) → (B, T, D)
        pooling: one of "mean", "cls", "last"
    """

    VALID_POOLING = {"mean", "cls", "last"}

    def __init__(self, model: Callable[[Tensor], Tensor], pooling: str = "mean") -> None:
        if pooling not in self.VALID_POOLING:
            raise ValueError(f"pooling must be one of {self.VALID_POOLING}, got '{pooling}'")
        self.model = model
        self.pooling = pooling

    def encode(self, input_ids: Tensor) -> Tensor:
        """Encode a batch of sequences.

        Args:
            input_ids: (B, T)

        Returns:
            (B, D)
        """
        hidden = self.model(input_ids)  # (B, T, D)
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        elif self.pooling == "cls":
            return hidden[:, 0, :]
        else:  # "last"
            return hidden[:, -1, :]

    def batch_encode(self, input_ids_list: list[Tensor]) -> Tensor:
        """Pad a list of sequences to the same length and encode them.

        Args:
            input_ids_list: list of (T_i,) or (1, T_i) tensors

        Returns:
            (N, D)
        """
        # Normalise to 1-D
        seqs = []
        for t in input_ids_list:
            if t.dim() == 2:
                t = t.squeeze(0)
            seqs.append(t)

        max_len = max(s.shape[0] for s in seqs)
        device = seqs[0].device
        dtype = seqs[0].dtype

        padded = torch.zeros(len(seqs), max_len, dtype=dtype, device=device)
        for i, s in enumerate(seqs):
            padded[i, : s.shape[0]] = s

        return self.encode(padded)  # (N, D)


# ---------------------------------------------------------------------------
# FactConsistencyScorer
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


class FactConsistencyScorer:
    """Score how consistent a claim is with a source document."""

    def __init__(self, encoder: TextEncoder, nli_classifier: NLIClassifier) -> None:
        self.encoder = encoder
        self.nli = nli_classifier

    def score(self, source_ids: Tensor, claim_ids: Tensor) -> tuple[float, str]:
        """Encode source and claim then classify.

        Args:
            source_ids: (1, T) or (T,) — the premise / source document
            claim_ids:  (1, T) or (T,) — the hypothesis / claim

        Returns:
            (entailment_prob, label) where label ∈ {"entailment","neutral","contradiction"}
        """
        # Ensure (1, T)
        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(0)
        if claim_ids.dim() == 1:
            claim_ids = claim_ids.unsqueeze(0)

        src_repr = self.encoder.encode(source_ids)   # (1, D)
        clm_repr = self.encoder.encode(claim_ids)    # (1, D)

        with torch.no_grad():
            logits = self.nli(src_repr, clm_repr)    # (1, 3)
            probs = F.softmax(logits, dim=-1)        # (1, 3)

        entailment_prob = probs[0, 0].item()
        label_idx = int(probs[0].argmax().item())
        label = _LABEL_MAP[label_idx]

        return entailment_prob, label

    def batch_score(self, source_ids: Tensor, claims: list[Tensor]) -> list[float]:
        """Score multiple claims against a single source.

        Args:
            source_ids: (1, T) or (T,)
            claims:     list of (T_i,) or (1, T_i) tensors

        Returns:
            list of entailment probabilities, one per claim
        """
        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(0)

        src_repr = self.encoder.encode(source_ids)  # (1, D)
        claim_reprs = self.encoder.batch_encode(claims)  # (N, D)

        # Expand source to match number of claims
        n = claim_reprs.shape[0]
        src_expanded = src_repr.expand(n, -1)  # (N, D)

        with torch.no_grad():
            probs = F.softmax(self.nli(src_expanded, claim_reprs), dim=-1)  # (N, 3)

        return probs[:, 0].tolist()

    def consistency_threshold(
        self, scores: list[float], threshold: float = 0.5
    ) -> list[bool]:
        """Return True for each score that meets or exceeds threshold.

        Args:
            scores:    list of entailment probabilities
            threshold: float in [0, 1]

        Returns:
            list[bool] of the same length
        """
        return [s >= threshold for s in scores]


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Detect potential hallucinations in generated text using token entropy."""

    def __init__(self, model: Callable[[Tensor], Tensor], threshold: float = 3.0) -> None:
        self.model = model
        self.threshold = threshold

    def token_entropy(self, input_ids: Tensor) -> Tensor:
        """Compute per-token entropy of the model's output distribution.

        Args:
            input_ids: (1, T) or (T,)

        Returns:
            (T,) entropy at each position, all ≥ 0
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_ids)  # (1, T, V)  or (1, T, D)

        probs = F.softmax(logits[0], dim=-1)  # (T, V)
        # Clamp for numerical stability; entropy = -sum(p * log(p))
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
        return entropy

    def detect_hallucinations(
        self, input_ids: Tensor, context_ids: Tensor
    ) -> dict:
        """Detect hallucinations by combining entropy and NLI-free consistency.

        Args:
            input_ids:   (1, T) or (T,) — the generation to evaluate
            context_ids: (1, T) or (T,) — the grounding / source of truth

        Returns:
            dict with keys:
                'high_entropy_positions': list[int]
                'consistency_score':      float in [0, 1]
                'hallucination_risk':     str — "low" / "medium" / "high"
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)

        entropy = self.token_entropy(input_ids)  # (T,)
        high_entropy_positions = [
            int(i) for i, e in enumerate(entropy.tolist()) if e > self.threshold
        ]

        # Consistency score: fraction of tokens with entropy below threshold
        # (higher = more consistent / confident generation)
        T = entropy.shape[0]
        consistency_score = float((entropy <= self.threshold).sum().item()) / max(T, 1)

        # Hallucination risk heuristic
        high_frac = len(high_entropy_positions) / max(T, 1)
        if high_frac < 0.2 and consistency_score >= 0.8:
            hallucination_risk = "low"
        elif high_frac >= 0.5 or consistency_score < 0.5:
            hallucination_risk = "high"
        else:
            hallucination_risk = "medium"

        return {
            "high_entropy_positions": high_entropy_positions,
            "consistency_score": consistency_score,
            "hallucination_risk": hallucination_risk,
        }


# ---------------------------------------------------------------------------
# FactualConsistencyTrainer
# ---------------------------------------------------------------------------

class FactualConsistencyTrainer:
    """Fine-tune encoder + NLI classifier on consistency (NLI) data."""

    def __init__(
        self,
        encoder: TextEncoder,
        nli_classifier: NLIClassifier,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.encoder = encoder
        self.nli = nli_classifier
        self.optimizer = optimizer

    def train_step(
        self,
        premise_ids: Tensor,
        hypothesis_ids: Tensor,
        labels: Tensor,
    ) -> dict:
        """Single training step.

        Args:
            premise_ids:    (B, T)
            hypothesis_ids: (B, T)
            labels:         (B,) LongTensor with values in {0, 1, 2}

        Returns:
            dict with 'loss' (float) and 'accuracy' (float in [0, 1])
        """
        self.optimizer.zero_grad()

        prem_repr = self.encoder.encode(premise_ids)   # (B, D)
        hyp_repr = self.encoder.encode(hypothesis_ids)  # (B, D)

        logits = self.nli(prem_repr, hyp_repr)  # (B, 3)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}


# ---------------------------------------------------------------------------
# ConsistencyBenchmark
# ---------------------------------------------------------------------------

class ConsistencyBenchmark:
    """Evaluate consistency detection metrics."""

    def __init__(self) -> None:
        pass

    def precision_recall(
        self,
        predictions: list[bool],
        ground_truth: list[bool],
    ) -> dict:
        """Compute precision, recall, and F1.

        Returns:
            dict with 'precision', 'recall', 'f1' all in [0, 1]
        """
        tp = sum(p and g for p, g in zip(predictions, ground_truth))
        fp = sum(p and not g for p, g in zip(predictions, ground_truth))
        fn = sum(not p and g for p, g in zip(predictions, ground_truth))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def hallucination_rate(self, detector_outputs: list[dict]) -> float:
        """Fraction of outputs flagged as high hallucination risk.

        Args:
            detector_outputs: list of dicts from HallucinationDetector.detect_hallucinations

        Returns:
            float in [0, 1]
        """
        if not detector_outputs:
            return 0.0
        high_count = sum(
            1 for d in detector_outputs if d.get("hallucination_risk") == "high"
        )
        return high_count / len(detector_outputs)

    def calibration_error(
        self,
        scores: list[float],
        labels: list[bool],
        n_bins: int = 5,
    ) -> float:
        """Expected Calibration Error (ECE).

        Splits scores into n_bins equal-width bins in [0, 1] and measures
        the weighted average of |avg_confidence - fraction_positive| per bin.

        Returns:
            float in [0, 1]
        """
        if not scores:
            return 0.0

        n = len(scores)
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        ece = 0.0

        for b in range(n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            # Include right edge in last bin
            if b < n_bins - 1:
                in_bin = [(s, int(l)) for s, l in zip(scores, labels) if lo <= s < hi]
            else:
                in_bin = [(s, int(l)) for s, l in zip(scores, labels) if lo <= s <= hi]

            if not in_bin:
                continue

            avg_conf = sum(s for s, _ in in_bin) / len(in_bin)
            frac_pos = sum(l for _, l in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(avg_conf - frac_pos)

        return ece
