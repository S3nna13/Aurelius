"""
semantic_entropy.py -- Semantic Entropy and Generative Uncertainty

Measures uncertainty in free-form text generation at the semantic (meaning) level
rather than the token level.  Pure native PyTorch only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SemanticClusterer
# ---------------------------------------------------------------------------


class SemanticClusterer:
    """Groups generated sequences into semantic clusters."""

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold

    def jaccard_similarity(self, a: list[int], b: list[int]) -> float:
        """Jaccard similarity between two token sequences (treated as sets)."""
        set_a = set(a)
        set_b = set(b)
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def cluster_by_token_overlap(self, sequences: list[list[int]]) -> list[int]:
        """Greedy Jaccard clustering.

        Assign each sequence to the first existing cluster whose representative
        has Jaccard similarity > threshold; otherwise open a new cluster.
        """
        if not sequences:
            return []

        cluster_ids: list[int] = []
        representatives: list[list[int]] = []

        for seq in sequences:
            assigned = -1
            for cid, rep in enumerate(representatives):
                if self.jaccard_similarity(seq, rep) > self.similarity_threshold:
                    assigned = cid
                    break
            if assigned == -1:
                assigned = len(representatives)
                representatives.append(seq)
            cluster_ids.append(assigned)

        return cluster_ids

    def cluster_by_embedding(self, embeddings: torch.Tensor) -> list[int]:
        """Distance-based clustering using cosine similarity.

        Greedy: assign to the first existing cluster centre whose cosine
        similarity to the new point exceeds the threshold.

        Args:
            embeddings: Tensor of shape [N, d]
        Returns:
            List of integer cluster ids, length N.
        """
        n = embeddings.shape[0]
        if n == 0:
            return []

        normed = F.normalize(embeddings.float(), dim=-1)  # [N, d]

        cluster_ids: list[int] = []
        centres: list[torch.Tensor] = []

        for i in range(n):
            vec = normed[i]
            assigned = -1
            for cid, centre in enumerate(centres):
                sim = float(torch.dot(vec, centre).item())
                if sim > self.similarity_threshold:
                    assigned = cid
                    break
            if assigned == -1:
                assigned = len(centres)
                centres.append(vec.clone())
            cluster_ids.append(assigned)

        return cluster_ids


# ---------------------------------------------------------------------------
# SemanticEntropy
# ---------------------------------------------------------------------------


class SemanticEntropy:
    """Semantic entropy for free-form generation."""

    def __init__(self, n_samples: int = 10) -> None:
        self.n_samples = n_samples
        self._clusterer = SemanticClusterer()

    def compute(
        self,
        sequences: list[list[int]],
        log_probs: list[float],
    ) -> float:
        """Estimate semantic entropy.

        1. Cluster sequences by meaning (Jaccard token overlap).
        2. p(cluster_c) = sum_{s in c} exp(log_prob_s) / Z
        3. H_semantic = -sum_c p(c) * log p(c)
        """
        if not sequences:
            return 0.0

        cluster_ids = self._clusterer.cluster_by_token_overlap(sequences)
        n_clusters = max(cluster_ids) + 1

        log_probs_t = torch.tensor(log_probs, dtype=torch.float64)
        probs = torch.exp(log_probs_t)
        Z = probs.sum()
        if Z.item() == 0.0:
            return 0.0
        probs = probs / Z

        cluster_probs = torch.zeros(n_clusters, dtype=torch.float64)
        for i, cid in enumerate(cluster_ids):
            cluster_probs[cid] += probs[i]

        entropy = 0.0
        for p in cluster_probs:
            p_val = float(p.item())
            if p_val > 0.0:
                entropy -= p_val * math.log(p_val)

        return float(entropy)

    def predictive_entropy(self, log_probs: list[float]) -> float:
        """Standard entropy over individual sequences: -sum_i p_i * log p_i."""
        if not log_probs:
            return 0.0

        log_probs_t = torch.tensor(log_probs, dtype=torch.float64)
        probs = torch.exp(log_probs_t)
        Z = probs.sum()
        if Z.item() == 0.0:
            return 0.0
        probs = probs / Z

        entropy = 0.0
        for p in probs:
            p_val = float(p.item())
            if p_val > 0.0:
                entropy -= p_val * math.log(p_val)
        return float(entropy)

    def excess_entropy(
        self,
        sequences: list[list[int]],
        log_probs: list[float],
    ) -> float:
        """predictive_entropy - semantic_entropy (within-cluster ambiguity).

        Always >= 0 by construction.
        """
        pe = self.predictive_entropy(log_probs)
        se = self.compute(sequences, log_probs)
        return max(0.0, pe - se)


# ---------------------------------------------------------------------------
# GenerationSampler
# ---------------------------------------------------------------------------


class GenerationSampler:
    """Samples token sequences from a language model with temperature + nucleus."""

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Zero out tokens outside the top-p probability mass.

        Args:
            probs: 1-D tensor [V].
            top_p: cumulative probability threshold.
        Returns:
            Re-normalised tensor with low-prob tokens zeroed.
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)

        # Keep tokens until the cumsum first exceeds top_p
        remove_mask = cumsum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices, sorted_probs)

        total = filtered.sum()
        if total > 0:
            filtered = filtered / total
        return filtered

    def sample_sequence(
        self,
        input_ids: list[int],
        max_new: int,
    ) -> tuple[list[int], float]:
        """Temperature + nucleus sampling.

        Returns:
            (generated_tokens, sum_of_log_probs)
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        context = list(input_ids)
        generated: list[int] = []
        total_log_prob = 0.0

        with torch.no_grad():
            for _ in range(max_new):
                ids_t = torch.tensor([context], dtype=torch.long, device=device)
                logits = self.model(ids_t)  # [1, T, V] or [1, V]

                if logits.dim() == 3:
                    next_logits = logits[0, -1, :]
                else:
                    next_logits = logits[0]

                if self.temperature > 0:
                    next_logits = next_logits / self.temperature

                probs = F.softmax(next_logits, dim=-1)
                probs = GenerationSampler.nucleus_filter(probs, top_p=0.9)

                next_token = int(torch.multinomial(probs, num_samples=1).item())
                token_log_prob = math.log(float(probs[next_token].item()) + 1e-40)
                total_log_prob += token_log_prob

                context.append(next_token)
                generated.append(next_token)

        return generated, total_log_prob

    def sample_n(
        self,
        input_ids: list[int],
        n: int,
        max_new: int,
    ) -> list[tuple[list[int], float]]:
        """Draw n independent samples."""
        return [self.sample_sequence(input_ids, max_new) for _ in range(n)]


# ---------------------------------------------------------------------------
# SetBasedUncertainty
# ---------------------------------------------------------------------------


class SetBasedUncertainty:
    """Set-based uncertainty metrics."""

    def __init__(self) -> None:
        pass

    def answer_set_size(
        self,
        sequences: list[list[int]],
        clusterer: SemanticClusterer,
    ) -> int:
        """Number of distinct semantic clusters."""
        if not sequences:
            return 0
        cluster_ids = clusterer.cluster_by_token_overlap(sequences)
        return max(cluster_ids) + 1

    def p_true_estimate(
        self,
        sequences: list[list[int]],
        log_probs: list[float],
        verifier_scores: list[float],
    ) -> float:
        """Weighted average of verifier scores by sequence probability."""
        if not sequences:
            return 0.0

        log_probs_t = torch.tensor(log_probs, dtype=torch.float64)
        probs = torch.exp(log_probs_t)
        Z = probs.sum()
        if Z.item() == 0.0:
            return 0.0
        probs = probs / Z

        scores_t = torch.tensor(verifier_scores, dtype=torch.float64)
        return float(torch.dot(probs, scores_t).item())

    def confidence_score(self, log_probs: list[float]) -> float:
        """Geometric mean of probabilities: exp(mean(log_probs))."""
        if not log_probs:
            return 0.0
        mean_lp = sum(log_probs) / len(log_probs)
        return float(math.exp(mean_lp))


# ---------------------------------------------------------------------------
# UncertaintyCalibrator
# ---------------------------------------------------------------------------


class UncertaintyCalibrator:
    """Bin-based isotonic calibrator for uncertainty scores."""

    def __init__(self) -> None:
        self._bin_edges: list[float] = []
        self._bin_values: list[float] = []
        self._fitted = False

    def fit(
        self,
        predictions: list[float],
        labels: list[int],
        n_bins: int = 10,
    ) -> None:
        """Bin predictions and compute average label per bin."""
        edges = [i / n_bins for i in range(n_bins + 1)]
        bin_sum = [0.0] * n_bins
        bin_cnt = [0] * n_bins

        for pred, lab in zip(predictions, labels):
            b = min(int(pred * n_bins), n_bins - 1)
            bin_sum[b] += float(lab)
            bin_cnt[b] += 1

        bin_values: list[float] = []
        for b in range(n_bins):
            if bin_cnt[b] > 0:
                bin_values.append(bin_sum[b] / bin_cnt[b])
            else:
                bin_values.append((edges[b] + edges[b + 1]) / 2.0)

        bin_values = self._isotonic_regression(bin_values)

        self._bin_edges = edges
        self._bin_values = bin_values
        self._fitted = True

    @staticmethod
    def _isotonic_regression(values: list[float]) -> list[float]:
        """Pool adjacent violators algorithm for non-decreasing isotonic fit."""
        n = len(values)
        if n == 0:
            return []
        result = list(values)
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                mean_val = (result[i] + result[i + 1]) / 2.0
                result[i] = mean_val
                result[i + 1] = mean_val
                if i > 0:
                    i -= 1
            else:
                i += 1
        return result

    def calibrate(self, uncertainty: float) -> float:
        """Map raw uncertainty to calibrated confidence in [0, 1]."""
        if not self._fitted:
            return float(max(0.0, min(1.0, uncertainty)))

        b = min(int(uncertainty * len(self._bin_values)), len(self._bin_values) - 1)
        return float(max(0.0, min(1.0, self._bin_values[b])))

    def ece(
        self,
        uncertainties: list[float],
        labels: list[int],
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error in [0, 1].

        Treats (1 - uncertainty) as confidence, label as accuracy indicator.
        """
        n = len(uncertainties)
        if n == 0:
            return 0.0

        bin_acc_sum = [0.0] * n_bins
        bin_conf_sum = [0.0] * n_bins
        bin_cnt = [0] * n_bins

        for unc, lab in zip(uncertainties, labels):
            conf = 1.0 - unc
            b = min(int(conf * n_bins), n_bins - 1)
            bin_acc_sum[b] += float(lab)
            bin_conf_sum[b] += conf
            bin_cnt[b] += 1

        ece = 0.0
        for b in range(n_bins):
            cnt = bin_cnt[b]
            if cnt > 0:
                acc = bin_acc_sum[b] / cnt
                conf = bin_conf_sum[b] / cnt
                ece += (cnt / n) * abs(acc - conf)

        return float(ece)


# ---------------------------------------------------------------------------
# SemanticEntropyConfig
# ---------------------------------------------------------------------------


@dataclass
class SemanticEntropyConfig:
    """Configuration for semantic entropy pipeline."""

    n_samples: int = 10
    temperature: float = 1.0
    top_p: float = 0.9
    similarity_threshold: float = 0.5
    n_bins: int = 10
