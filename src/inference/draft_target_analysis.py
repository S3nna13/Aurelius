"""Draft-target alignment metrics and analysis for speculative decoding.

Tools for analyzing and improving draft-target token alignment in speculative
decoding. The key metric is the "acceptance rate": how often the target model
approves the draft model's token predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AcceptanceStats:
    total_tokens: int
    accepted_tokens: int
    acceptance_rate: float  # accepted / total
    mean_acceptance_length: float  # avg tokens accepted per speculative step
    token_acceptance_dist: list[float]  # acceptance rate per position (0..max_draft-1)


def compute_token_acceptance(
    draft_tokens: torch.Tensor,  # (B, T_draft) proposed tokens
    target_logits: torch.Tensor,  # (B, T_draft, vocab) target model logits
    draft_logits: torch.Tensor,  # (B, T_draft, vocab) draft model logits
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Speculative decoding acceptance criterion:
    Accept token x if uniform_random < min(1, p_target(x) / p_draft(x))
    Returns: (B, T_draft) boolean mask — True = accepted.
    """
    B, T, V = target_logits.shape

    # Apply temperature scaling
    temp = max(temperature, 1e-8)
    target_probs = F.softmax(target_logits / temp, dim=-1)  # (B, T, V)
    draft_probs = F.softmax(draft_logits / temp, dim=-1)  # (B, T, V)

    # Gather probabilities for the draft tokens
    # draft_tokens: (B, T) -> indices for gather
    idx = draft_tokens.unsqueeze(-1)  # (B, T, 1)
    p_target = target_probs.gather(-1, idx).squeeze(-1)  # (B, T)
    p_draft = draft_probs.gather(-1, idx).squeeze(-1)  # (B, T)

    # Acceptance ratio: min(1, p_target / p_draft)
    ratio = (p_target / (p_draft + 1e-10)).clamp(max=1.0)

    # Draw uniform random samples
    uniform = torch.rand_like(ratio)

    accepted = uniform < ratio  # (B, T) boolean
    return accepted


def acceptance_rate(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Mean acceptance rate across all positions and batch items."""
    mask = compute_token_acceptance(
        draft_tokens, target_logits, draft_logits, temperature=temperature
    )
    return float(mask.float().mean().item())


def compute_acceptance_stats(
    draft_tokens: torch.Tensor,  # (B, T_draft)
    target_logits: torch.Tensor,  # (B, T_draft, vocab)
    draft_logits: torch.Tensor,  # (B, T_draft, vocab)
    temperature: float = 1.0,
) -> AcceptanceStats:
    """Compute comprehensive acceptance statistics."""
    B, T = draft_tokens.shape

    mask = compute_token_acceptance(
        draft_tokens, target_logits, draft_logits, temperature=temperature
    )  # (B, T)

    total_tokens = B * T
    accepted_tokens = int(mask.sum().item())
    rate = accepted_tokens / total_tokens if total_tokens > 0 else 0.0

    # Mean acceptance length per batch item: consecutive accepted tokens from left
    total_accepted_length = 0.0
    for b in range(B):
        row = mask[b]  # (T,)
        # count consecutive accepted starting from position 0
        length = 0
        for t in range(T):
            if row[t]:
                length += 1
            else:
                break
        total_accepted_length += length
    mean_acceptance_length = total_accepted_length / B if B > 0 else 0.0

    # Per-position acceptance rate
    token_acceptance_dist = mask.float().mean(dim=0).tolist()  # length T

    return AcceptanceStats(
        total_tokens=total_tokens,
        accepted_tokens=accepted_tokens,
        acceptance_rate=rate,
        mean_acceptance_length=mean_acceptance_length,
        token_acceptance_dist=token_acceptance_dist,
    )


def calibrate_draft_temperature(
    draft_logits: torch.Tensor,  # (N, vocab) collected from many examples
    target_logits: torch.Tensor,  # (N, vocab)
    n_steps: int = 20,
    lr: float = 0.1,
) -> float:
    """
    Find temperature T* for draft logits that maximizes acceptance rate.
    Grid search over [0.5, 2.0]: return T* with highest acceptance.
    """
    N, V = draft_logits.shape

    # Create dummy draft tokens by sampling from target distribution
    with torch.no_grad():
        target_probs = F.softmax(target_logits, dim=-1)
        draft_tokens = torch.multinomial(target_probs, num_samples=1).squeeze(-1)  # (N,)

    # Reshape to (1, N) batch for compute_token_acceptance
    draft_tokens_2d = draft_tokens.unsqueeze(0)  # (1, N)
    target_logits_2d = target_logits.unsqueeze(0)  # (1, N, V)
    draft_logits_2d = draft_logits.unsqueeze(0)  # (1, N, V)

    best_temp = 1.0
    best_rate = -1.0

    # Grid search over [0.5, 2.0]
    temperatures = torch.linspace(0.5, 2.0, n_steps).tolist()
    for temp in temperatures:
        mask = compute_token_acceptance(
            draft_tokens_2d, target_logits_2d, draft_logits_2d, temperature=temp
        )
        rate = float(mask.float().mean().item())
        if rate > best_rate:
            best_rate = rate
            best_temp = temp

    return float(best_temp)


def kl_divergence_analysis(
    draft_logits: torch.Tensor,  # (N, vocab)
    target_logits: torch.Tensor,  # (N, vocab)
) -> dict[str, float]:
    """
    Compute KL divergence statistics between draft and target distributions.
    Returns {'mean_kl': ..., 'max_kl': ..., 'frac_high_kl': ...}
    where frac_high_kl = fraction of positions with KL > 1.0.
    """
    target_probs = F.softmax(target_logits, dim=-1)  # (N, V)
    draft_probs = F.softmax(draft_logits, dim=-1)  # (N, V)

    # KL(target || draft) per position
    # KL = sum_x target(x) * log(target(x) / draft(x))
    kl_per_pos = (
        target_probs * (torch.log(target_probs + 1e-10) - torch.log(draft_probs + 1e-10))
    ).sum(dim=-1)  # (N,)

    mean_kl = float(kl_per_pos.mean().item())
    max_kl = float(kl_per_pos.max().item())
    frac_high_kl = float((kl_per_pos > 1.0).float().mean().item())

    return {
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "frac_high_kl": frac_high_kl,
    }


def compute_expected_speedup(
    acceptance_rate: float,
    n_draft_tokens: int = 4,
    draft_overhead: float = 0.1,  # draft model cost as fraction of target cost
) -> float:
    """
    Expected speedup from speculative decoding:
    speedup = (1 + alpha + alpha² + ... + alpha^k) / (1 + k * overhead)
    where alpha = acceptance_rate, k = n_draft_tokens.
    """
    alpha = acceptance_rate
    k = n_draft_tokens

    # Numerator: geometric series sum
    numerator = sum(alpha**i for i in range(k + 1))

    # Denominator: cost including draft overhead
    denominator = 1.0 + k * draft_overhead

    return numerator / denominator


class DraftQualityMonitor:
    """
    Online monitoring of draft model quality during inference.
    Tracks acceptance rate with exponential moving average.
    """

    def __init__(
        self,
        window_size: int = 100,
        ema_decay: float = 0.99,
        alert_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.ema_decay = ema_decay
        self.alert_threshold = alert_threshold

        self._ema_acceptance: float = 1.0
        self._ema_initialized: bool = False
        self._window_accepted: list[float] = []
        self._total_accepted: int = 0
        self._total_tokens: int = 0

    def update(
        self,
        accepted: int,
        total: int,
    ) -> None:
        """Update stats with results from one speculative decoding step."""
        if total <= 0:
            return

        step_rate = accepted / total

        # Exponential moving average
        if not self._ema_initialized:
            self._ema_acceptance = step_rate
            self._ema_initialized = True
        else:
            self._ema_acceptance = (
                self.ema_decay * self._ema_acceptance + (1.0 - self.ema_decay) * step_rate
            )

        # Sliding window
        self._window_accepted.append(step_rate)
        if len(self._window_accepted) > self.window_size:
            self._window_accepted.pop(0)

        self._total_accepted += accepted
        self._total_tokens += total

    def get_stats(self) -> dict[str, float]:
        """Return {'ema_acceptance': ..., 'window_acceptance': ..., 'total_accepted': ...}"""
        if self._window_accepted:
            window_acceptance = sum(self._window_accepted) / len(self._window_accepted)
        else:
            window_acceptance = 0.0

        return {
            "ema_acceptance": self._ema_acceptance,
            "window_acceptance": window_acceptance,
            "total_accepted": float(self._total_accepted),
        }

    def should_fallback(self) -> bool:
        """True if acceptance rate dropped below alert_threshold."""
        stats = self.get_stats()
        return stats["ema_acceptance"] < self.alert_threshold

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._ema_acceptance = 1.0
        self._ema_initialized = False
        self._window_accepted = []
        self._total_accepted = 0
        self._total_tokens = 0


def top_k_overlap(
    draft_logits: torch.Tensor,  # (vocab,)
    target_logits: torch.Tensor,  # (vocab,)
    k: int = 10,
) -> float:
    """
    Fraction of target's top-k tokens that appear in draft's top-k.
    Returns float in [0, 1].
    """
    _, draft_top_k = torch.topk(draft_logits, k)
    _, target_top_k = torch.topk(target_logits, k)

    draft_set = set(draft_top_k.tolist())
    target_set = set(target_top_k.tolist())

    overlap = len(draft_set & target_set)
    return overlap / k if k > 0 else 0.0
