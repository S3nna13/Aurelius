"""Domain reweighting for multi-source training: DoReMi / Group DRO style adaptive sampling."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class DomainConfig:
    domain_names: list[str]
    initial_weights: list[float] | None = None  # defaults to uniform
    learning_rate: float = 1.0                  # for weight updates
    smoothing_alpha: float = 0.1                # EMA smoothing
    min_weight: float = 0.01                    # minimum domain weight
    normalize: bool = True                      # keep weights summing to 1
    strategy: str = "ema"                       # "ema" | "exp3" | "doro"


def normalize_weights(weights: Tensor, min_weight: float = 0.01) -> Tensor:
    """Clip weights to min_weight and renormalize to sum=1.

    Args:
        weights: (n_domains,) tensor of raw domain weights
        min_weight: floor value applied before renormalization

    Returns:
        (n_domains,) normalized probability tensor
    """
    w = weights.float().clamp(min=min_weight)
    return w / w.sum()


def ema_update(weights: Tensor, losses: Tensor, alpha: float, lr: float) -> Tensor:
    """EMA update: upweight domains with high loss.

    new_weights = (1 - alpha) * weights + alpha * softmax(losses * lr)

    Args:
        weights: (n_domains,) current domain weights
        losses: (n_domains,) per-domain losses
        alpha: EMA smoothing coefficient (0 < alpha <= 1)
        lr: learning-rate scaling applied to losses before softmax

    Returns:
        Updated (n_domains,) weight tensor
    """
    target = F.softmax(losses.float() * lr, dim=0)
    new_weights = (1.0 - alpha) * weights.float() + alpha * target
    return new_weights


def exp3_update(weights: Tensor, losses: Tensor, lr: float) -> Tensor:
    """EXP3 bandit update for domain weights.

    importance = losses / weights
    log_weights = log(weights) + lr * importance
    Returns softmax(log_weights)

    Args:
        weights: (n_domains,) current domain weights (must be > 0)
        losses: (n_domains,) per-domain losses
        lr: learning-rate scaling

    Returns:
        Updated (n_domains,) probability tensor
    """
    w = weights.float().clamp(min=1e-8)
    importance = losses.float() / w
    log_w = torch.log(w) + lr * importance
    return F.softmax(log_w, dim=0)


def doro_update(
    weights: Tensor,
    losses: Tensor,
    lr: float,
    eta: float = 0.5,
) -> Tensor:
    """DoReMi-style weight update.

    Upweights domains where loss > weighted average loss.

    excess = losses - (weights * losses).sum()
    weights = weights * exp(lr * excess)
    Returns normalized weights.

    Args:
        weights: (n_domains,) current domain weights
        losses: (n_domains,) per-domain losses
        lr: learning-rate scaling
        eta: unused in simplified version, kept for API compatibility

    Returns:
        Normalized (n_domains,) weight tensor
    """
    w = weights.float()
    l = losses.float()
    excess = l - (w * l).sum()
    updated = w * torch.exp(lr * excess)
    return normalize_weights(updated)


class DomainReweighter:
    """Adaptive domain weight manager for multi-source training.

    Maintains a probability distribution over training domains and updates it
    based on per-domain loss signals using the strategy specified in
    DomainConfig (ema, exp3, or doro).

    Usage:
        cfg = DomainConfig(domain_names=["web", "books", "code"])
        reweighter = DomainReweighter(cfg)

        # After each training step, record per-domain losses
        reweighter.update({"web": 2.1, "books": 1.8, "code": 3.4})

        # Sample a domain for the next batch
        domain = reweighter.sample_domain(rng)
    """

    def __init__(self, config: DomainConfig) -> None:
        self.config = config
        n = len(config.domain_names)
        if n == 0:
            raise ValueError("domain_names must not be empty")

        if config.initial_weights is not None:
            if len(config.initial_weights) != n:
                raise ValueError(
                    f"initial_weights length ({len(config.initial_weights)}) "
                    f"must match domain_names length ({n})"
                )
            w = torch.tensor(config.initial_weights, dtype=torch.float)
        else:
            w = torch.ones(n, dtype=torch.float)

        self._weights: Tensor = normalize_weights(w, config.min_weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, domain_losses: dict[str, float]) -> None:
        """Update domain weights given per-domain losses.

        Domains not present in domain_losses keep their current loss
        estimate (set to 0.0 for missing entries so they are not upweighted).

        Args:
            domain_losses: mapping from domain name to scalar loss value
        """
        names = self.config.domain_names
        losses = torch.tensor(
            [domain_losses.get(name, 0.0) for name in names],
            dtype=torch.float,
        )

        strategy = self.config.strategy
        if strategy == "ema":
            new_w = ema_update(
                self._weights, losses,
                alpha=self.config.smoothing_alpha,
                lr=self.config.learning_rate,
            )
        elif strategy == "exp3":
            new_w = exp3_update(
                self._weights, losses,
                lr=self.config.learning_rate,
            )
        elif strategy == "doro":
            new_w = doro_update(
                self._weights, losses,
                lr=self.config.learning_rate,
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of: ema, exp3, doro"
            )

        if self.config.normalize:
            new_w = normalize_weights(new_w, self.config.min_weight)

        self._weights = new_w

    def sample_domain(self, rng: random.Random) -> str:
        """Sample a domain name according to current weights.

        Args:
            rng: Python random.Random instance for reproducibility

        Returns:
            Sampled domain name
        """
        probs = self._weights.tolist()
        # Use rng.choices for weighted sampling
        (chosen,) = rng.choices(self.config.domain_names, weights=probs, k=1)
        return chosen

    def get_weights(self) -> dict[str, float]:
        """Return {domain_name: weight} mapping."""
        return {
            name: w
            for name, w in zip(self.config.domain_names, self._weights.tolist())
        }

    def get_sampling_probs(self) -> Tensor:
        """Return (n_domains,) probability tensor (copy)."""
        return self._weights.clone()


def build_reweighted_batch(
    domain_data: dict[str, list[Tensor]],
    reweighter: DomainReweighter,
    batch_size: int,
    rng: random.Random,
) -> tuple[Tensor, list[str]]:
    """Sample a batch from multiple domains according to reweighter weights.

    For each of the batch_size slots, a domain is sampled according to current
    weights, then a random sequence is drawn from that domain's data pool.
    All sequences are padded (with zeros) to the length of the longest sequence
    in the batch.

    Args:
        domain_data: mapping from domain name to list of 1-D token ID tensors
        reweighter: DomainReweighter holding current domain weights
        batch_size: number of examples to include in the batch
        rng: Python random.Random instance for reproducibility

    Returns:
        (input_ids_batch, domain_labels)
        - input_ids_batch: LongTensor of shape (batch_size, max_seq_len)
        - domain_labels: list of domain names, one per example
    """
    sequences: list[Tensor] = []
    domain_labels: list[str] = []

    for _ in range(batch_size):
        domain = reweighter.sample_domain(rng)
        pool = domain_data[domain]
        seq = rng.choice(pool)
        sequences.append(seq.long())
        domain_labels.append(domain)

    # Pad to common length
    max_len = max(s.size(0) for s in sequences)
    padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq

    return padded, domain_labels
