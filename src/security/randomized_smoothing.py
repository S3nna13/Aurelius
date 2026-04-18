"""Randomized smoothing for certified l2 robustness of a classifier."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothedClassifier:
    """Wraps a base classifier with Gaussian smoothing to certify l2 robustness."""

    def __init__(
        self,
        model: nn.Module,
        sigma: float,
        n_samples: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.sigma = sigma
        self.n_samples = n_samples
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_noise(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Repeat x n times and add independent N(0, sigma^2) noise to each copy.

        Args:
            x: Float tensor of shape (1, S, d_model).
            n: Number of noisy copies to generate.

        Returns:
            Tensor of shape (n, 1, S, d_model).
        """
        x = x.to(self.device)
        repeated = x.unsqueeze(0).expand(n, *x.shape).clone()
        noise = torch.randn_like(repeated) * self.sigma
        return repeated + noise

    def _majority_vote(self, logits: torch.Tensor) -> Tuple[int, int, int]:
        """Compute plurality-class vote from per-sample logits.

        Args:
            logits: Tensor of shape (n_samples, S, num_classes). The last-token
                position (index -1 along dim 1) is used for classification.

        Returns:
            (top_class, top_count, total) where top_class is the argmax class
            by vote count, top_count is its vote tally, and total is n_samples.
        """
        last_token_logits = logits[:, -1, :]  # (n_samples, num_classes)
        probs = F.softmax(last_token_logits, dim=-1)
        predicted = probs.argmax(dim=-1)  # (n_samples,)

        num_classes = logits.shape[-1]
        counts = torch.bincount(predicted, minlength=num_classes)
        top_class = int(counts.argmax().item())
        top_count = int(counts[top_class].item())
        total = int(predicted.shape[0])
        return top_class, top_count, total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor, n_samples_override: Optional[int] = None) -> int:
        """Return the smoothed classifier's predicted class via majority vote.

        Args:
            x: Float tensor of shape (1, S, d_model).
            n_samples_override: If provided, use this many samples instead of
                self.n_samples.

        Returns:
            Predicted class index as an int.
        """
        n = n_samples_override if n_samples_override is not None else self.n_samples
        noisy = self._add_noise(x, n)  # (n, 1, S, d_model)
        # Flatten batch dimension for model forward pass
        batch = noisy.view(n, *x.shape[1:])  # (n, S, d_model)
        with torch.no_grad():
            logits = self.model(batch)  # (n, S, num_classes)
        top_class, _, _ = self._majority_vote(logits)
        return top_class

    def certify(
        self, x: torch.Tensor, alpha: float = 0.001
    ) -> Tuple[int, float, bool]:
        """Certify the l2 robustness radius around input x.

        Draws n_samples noisy versions, counts votes, computes a lower-confidence
        bound p_A_hat on the probability of the top class via Clopper-Pearson
        (or normal approximation), and derives the certified radius.

        Args:
            x: Float tensor of shape (1, S, d_model).
            alpha: Confidence level for the one-sided binomial lower bound.

        Returns:
            (predicted_class, certified_radius, abstain).
            abstain=True means the classifier cannot certify (p_A_hat <= 0.5).
        """
        noisy = self._add_noise(x, self.n_samples)  # (n, 1, S, d_model)
        batch = noisy.view(self.n_samples, *x.shape[1:])  # (n, S, d_model)
        with torch.no_grad():
            logits = self.model(batch)  # (n, S, num_classes)

        predicted_class, k, n = self._majority_vote(logits)

        # Clopper-Pearson lower bound on the binomial proportion
        p_A_hat = self._clopper_pearson_lower(k, n, alpha)

        if p_A_hat > 0.5:
            normal = torch.distributions.Normal(0.0, 1.0)
            radius = float(self.sigma * normal.icdf(torch.tensor(p_A_hat)).item())
            return predicted_class, radius, False
        else:
            return predicted_class, 0.0, True

    # ------------------------------------------------------------------
    # Statistical helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
        """One-sided Clopper-Pearson lower confidence bound at level alpha.

        Falls back to a normal approximation when scipy is unavailable.

        Args:
            k: Number of successes.
            n: Total trials.
            alpha: Significance level (e.g. 0.001).

        Returns:
            Lower bound on the binomial proportion p.
        """
        if n == 0:
            return 0.0
        try:
            from scipy.stats import beta as beta_dist
            # Lower bound: alpha-th quantile of Beta(k, n-k+1)
            if k == 0:
                return 0.0
            return float(beta_dist.ppf(alpha, k, n - k + 1))
        except ImportError:
            # Normal approximation: p_hat - z * se
            p_hat = k / n
            # z-score for alpha=0.001 is ~3.09; use scipy.stats if available
            # for other alpha values approximate with -Phi_inv(alpha)
            normal = torch.distributions.Normal(0.0, 1.0)
            z = float(normal.icdf(torch.tensor(1.0 - alpha)).item())
            se = (p_hat * (1.0 - p_hat) / n) ** 0.5
            return max(0.0, p_hat - z * se)
