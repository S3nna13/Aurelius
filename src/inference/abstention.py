"""Confidence estimation and selective prediction (abstention) for AureliusTransformer.

When the model is uncertain about a response, it can choose to abstain rather than
produce an unreliable answer. Uncertainty is measured via token entropy, Monte Carlo
dropout, and semantic clustering of sampled responses.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceEstimator:
    """Estimate model confidence from output distributions.

    Args:
        model: language model
        threshold: confidence threshold for abstention (default 0.5)
    """

    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-position entropy of token distribution.

        Args:
            logits: (B, S, vocab_size)

        Returns:
            (B, S) entropy in nats  H = -sum_v p(v) * log(p(v))
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def sequence_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Overall sequence confidence as mean max-probability per position.

        Higher value = more confident.

        Args:
            logits: (B, S, vocab_size)

        Returns:
            (B,) confidence scores in [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        return max_probs.mean(dim=-1)

    def should_abstain(self, logits: torch.Tensor) -> torch.Tensor:
        """Decide per-batch-element whether to abstain.

        Args:
            logits: (B, S, vocab_size)

        Returns:
            (B,) bool tensor -- True if model should abstain
        """
        confidence = self.sequence_confidence(logits)
        return confidence < self.threshold

    def top_k_confidence(self, logits: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Cumulative probability of top-k tokens per position.

        High = model concentrated on few tokens (confident).
        Low  = model spread over many tokens (uncertain).

        Args:
            logits: (B, S, vocab_size)
            k: number of top tokens

        Returns:
            (B, S) -- sum of top-k probabilities
        """
        probs = F.softmax(logits, dim=-1)
        vocab_size = probs.size(-1)
        actual_k = min(k, vocab_size)
        top_k_probs, _ = probs.topk(actual_k, dim=-1)
        return top_k_probs.sum(dim=-1)


class MCDropoutEstimator:
    """Monte Carlo Dropout uncertainty estimation.

    Run the model multiple times with dropout enabled during inference.
    Variance across runs = uncertainty estimate.

    Args:
        model: model with dropout layers
        n_samples: number of forward passes (default 10)
    """

    def __init__(self, model, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def estimate(
        self,
        input_ids: torch.Tensor,
        return_all_samples: bool = False,
    ) -> dict:
        """Run n_samples forward passes with dropout enabled.

        Returns dict with keys:
            'mean_logits': (B, S, vocab_size)
            'variance':    (B, S, vocab_size)  variance across samples
            'mean_entropy': (B, S)             mean entropy across samples
            'predictive_entropy': (B, S)       entropy of mean prediction
            'mutual_information': (B, S)       predictive_entropy - mean_entropy
            'samples': list of logits tensors  (if return_all_samples=True)
        """
        self._enable_dropout()
        all_logits = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(input_ids)
                if isinstance(out, tuple):
                    logits = out[1]
                else:
                    logits = out
                all_logits.append(logits)

        self._disable_dropout()

        stacked = torch.stack(all_logits, dim=0)       # (N, B, S, V)
        mean_logits = stacked.mean(dim=0)              # (B, S, V)
        variance = stacked.var(dim=0)                  # (B, S, V)

        sample_probs = F.softmax(stacked, dim=-1)      # (N, B, S, V)
        eps = 1e-9

        p_clamp = sample_probs.clamp(min=eps)
        per_sample_entropy = -(p_clamp * p_clamp.log()).sum(dim=-1)  # (N, B, S)
        mean_entropy = per_sample_entropy.mean(dim=0)                # (B, S)

        mean_probs = sample_probs.mean(dim=0)          # (B, S, V)
        mp_clamp = mean_probs.clamp(min=eps)
        pred_entropy = -(mp_clamp * mp_clamp.log()).sum(dim=-1)      # (B, S)

        mi = (pred_entropy - mean_entropy).clamp(min=0.0)           # (B, S)

        result = {
            "mean_logits": mean_logits,
            "variance": variance,
            "mean_entropy": mean_entropy,
            "predictive_entropy": pred_entropy,
            "mutual_information": mi,
        }
        if return_all_samples:
            result["samples"] = all_logits

        return result

    def _enable_dropout(self):
        """Set model to training mode (enables dropout) but don't track gradients."""
        self.model.train()

    def _disable_dropout(self):
        """Restore model to eval mode."""
        self.model.eval()


class SelectivePredictor:
    """Selective prediction: abstain when uncertain, answer when confident.

    Tracks abstention rate and accuracy on answered questions.
    """

    def __init__(
        self,
        confidence_estimator: ConfidenceEstimator,
        abstention_message: str = "I'm not confident enough to answer this.",
    ):
        self.estimator = confidence_estimator
        self.abstention_message = abstention_message
        self._n_answered = 0
        self._n_abstained = 0

    def predict(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict:
        """Decide whether to return the generated response or abstain.

        Args:
            input_ids:     (B, S_in) input token IDs (unused, for API compatibility)
            generated_ids: (B, S_gen) generated token IDs
            logits:        (B, S_gen, vocab_size) logits for generated tokens

        Returns dict with keys:
            'abstained':    bool
            'confidence':   float
            'response_ids': tensor or None
            'message':      str or None
        """
        confidence_tensor = self.estimator.sequence_confidence(logits)
        confidence = confidence_tensor.mean().item()
        abstain = confidence < self.estimator.threshold

        if abstain:
            self._n_abstained += 1
            return {
                "abstained": True,
                "confidence": confidence,
                "response_ids": None,
                "message": self.abstention_message,
            }
        else:
            self._n_answered += 1
            return {
                "abstained": False,
                "confidence": confidence,
                "response_ids": generated_ids,
                "message": None,
            }

    def abstention_rate(self) -> float:
        """Fraction of predictions where model abstained."""
        total = self._n_answered + self._n_abstained
        if total == 0:
            return 0.0
        return self._n_abstained / total

    def reset_stats(self):
        """Reset counters."""
        self._n_answered = 0
        self._n_abstained = 0


class SemanticUncertainty:
    """Semantic entropy for uncertainty estimation.

    Key insight: if model generates "Paris" and "The capital of France",
    these are semantically equivalent -> low uncertainty.
    If it generates "Paris" and "London", these are different -> high uncertainty.

    Simplified version: use token-level overlap (Jaccard similarity) as
    a proxy for semantic equivalence.
    """

    def __init__(self, n_samples: int = 5, similarity_threshold: float = 0.5):
        self.n_samples = n_samples
        self.similarity_threshold = similarity_threshold

    def jaccard_similarity(self, seq_a: list, seq_b: list) -> float:
        """Set-based Jaccard similarity between two token sequences."""
        set_a = set(seq_a)
        set_b = set(seq_b)
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def cluster_responses(self, response_list: list) -> list:
        """Group semantically similar responses into clusters.

        Two responses are placed in the same cluster if their Jaccard similarity
        exceeds the threshold. Uses greedy first-match clustering.

        Args:
            response_list: list of token ID sequences

        Returns:
            list of clusters, each cluster is a list of response indices
        """
        clusters = []

        for idx, response in enumerate(response_list):
            placed = False
            for cluster in clusters:
                representative = response_list[cluster[0]]
                if self.jaccard_similarity(response, representative) > self.similarity_threshold:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        return clusters

    def semantic_entropy(self, clusters: list, n_total: int) -> float:
        """Entropy over clusters.

        p(cluster) = cluster_size / n_total
        H = -sum p_c * log(p_c)

        Args:
            clusters: list of clusters (each cluster = list of response indices)
            n_total:  total number of responses

        Returns:
            semantic entropy in nats
        """
        if n_total == 0:
            return 0.0
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n_total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
