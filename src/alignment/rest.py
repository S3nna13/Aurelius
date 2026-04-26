"""Aurelius — ReST (Reinforced Self-Training).

Implements the core ReST algorithm from:
  "Beyond Human Data: Scaling Self-Training for Problem-Solving with
   Language Models", Singh et al., 2023, Google DeepMind.  arXiv:2312.06585

Algorithm (Algorithm 1 in the paper):
  - Input: dataset D = {(x_i, reward_fn)}, initial policy π_0
  - For k = 1..K:
      1. Grow:   generate N candidates y_1..y_N per problem x using π_{k-1}
      2. Score:  r_j = reward_fn(x, y_j) for each candidate
      3. Filter: keep (x, y_j) pairs where r_j >= τ_k
                 (τ_k increases each iteration → raises the quality bar)
      4. Improve: fine-tune π_k on filtered set D_k via SFT
                  (maximise log π(y_j|x), i.e. cross-entropy NLL)

Loss:
  L_ReST = -E_{(x,y) ∈ D_k} [log π(y|x)]
         = cross-entropy NLL on accepted completions only
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ReSTDataset
# ---------------------------------------------------------------------------


class ReSTDataset:
    """Container for (prompt_tokens, completion_tokens, reward) triples.

    Attributes:
        prompts:     list of 1-D LongTensors (variable length)
        completions: list of 1-D LongTensors (variable length)
        rewards:     list of float scalars
    """

    def __init__(
        self,
        prompts: list[torch.Tensor],
        completions: list[torch.Tensor],
        rewards: list[float],
    ) -> None:
        if not (len(prompts) == len(completions) == len(rewards)):
            raise ValueError(
                "prompts, completions, and rewards must have the same length; "
                f"got {len(prompts)}, {len(completions)}, {len(rewards)}"
            )
        self.prompts = list(prompts)
        self.completions = list(completions)
        self.rewards = list(rewards)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.prompts)

    # ------------------------------------------------------------------
    def filter(self, threshold: float) -> ReSTDataset:
        """Return a new ReSTDataset containing only pairs with reward >= threshold."""
        kept_p, kept_c, kept_r = [], [], []
        for p, c, r in zip(self.prompts, self.completions, self.rewards):
            if r >= threshold:
                kept_p.append(p)
                kept_c.append(c)
                kept_r.append(r)
        return ReSTDataset(kept_p, kept_c, kept_r)

    # ------------------------------------------------------------------
    def grow(
        self,
        generator_fn: Callable[[torch.Tensor], tuple[torch.Tensor, float]],
        n_samples_per_prompt: int = 4,
    ) -> ReSTDataset:
        """Expand dataset by generating multiple completions per prompt.

        Args:
            generator_fn: Callable (prompt_tokens,) -> (completion_tokens, reward)
                          A mock can return random tensors + random rewards.
            n_samples_per_prompt: Number of candidates to generate per prompt.

        Returns:
            New ReSTDataset with (len(self) * n_samples_per_prompt) entries.
        """
        if n_samples_per_prompt < 1:
            raise ValueError("n_samples_per_prompt must be >= 1")

        new_p: list[torch.Tensor] = []
        new_c: list[torch.Tensor] = []
        new_r: list[float] = []

        # We iterate over the *unique* prompts in the current dataset so that
        # each seed prompt gets n_samples_per_prompt candidates added.
        for prompt in self.prompts:
            for _ in range(n_samples_per_prompt):
                completion, reward = generator_fn(prompt)
                if not isinstance(completion, torch.Tensor):
                    raise TypeError(
                        f"generator_fn must return a torch.Tensor completion; "
                        f"got {type(completion)}"
                    )
                new_p.append(prompt)
                new_c.append(completion)
                new_r.append(float(reward))

        return ReSTDataset(new_p, new_c, new_r)


# ---------------------------------------------------------------------------
# ReSTThresholdSchedule
# ---------------------------------------------------------------------------


class ReSTThresholdSchedule:
    """Computes the reward threshold τ_k for iteration k.

    At each iteration the threshold is set to the q_k-th percentile of the
    current reward distribution, where q_k increases by ``increment`` per step.

    Args:
        base_percentile: Percentile used at k=0 (default: 50 → median).
        increment:       Percentile increase per iteration (default: 10).
        max_percentile:  Ceiling for the percentile (default: 95).
    """

    def __init__(
        self,
        base_percentile: float = 50.0,
        increment: float = 10.0,
        max_percentile: float = 95.0,
    ) -> None:
        if not (0.0 <= base_percentile <= 100.0):
            raise ValueError("base_percentile must be in [0, 100]")
        if increment < 0.0:
            raise ValueError("increment must be non-negative")
        self.base_percentile = base_percentile
        self.increment = increment
        self.max_percentile = max_percentile

    # ------------------------------------------------------------------
    def percentile_for_iteration(self, k: int) -> float:
        """Return the percentile target for iteration k."""
        pct = self.base_percentile + k * self.increment
        return min(pct, self.max_percentile)

    # ------------------------------------------------------------------
    def threshold_for_iteration(self, k: int, rewards: torch.Tensor) -> float:
        """Compute the scalar threshold for iteration k from reward tensor.

        Args:
            k:       Current iteration index (0-based).
            rewards: 1-D float Tensor of reward values.

        Returns:
            Scalar float threshold τ_k.
        """
        if rewards.numel() == 0:
            raise ValueError("rewards tensor must not be empty")
        rewards_f = rewards.float()
        pct = self.percentile_for_iteration(k)
        # torch.quantile expects a value in [0, 1]
        q = torch.tensor(pct / 100.0, dtype=torch.float32)
        threshold = torch.quantile(rewards_f, q).item()
        return float(threshold)


# ---------------------------------------------------------------------------
# ReSTLoss
# ---------------------------------------------------------------------------


class ReSTLoss:
    """Computes the ReST SFT loss on accepted (reward >= threshold) samples.

    The loss is the average negative log-likelihood over accepted completions:

        L = -mean_{(x,y) where r >= τ} [log π(y|x)]

    If no samples pass the threshold the loss is zero (a scalar zero tensor
    that still participates in the autograd graph).

    Args:
        reduction: 'mean' (default) or 'sum' over accepted tokens.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    # ------------------------------------------------------------------
    def __call__(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, int]:
        """Compute the ReST loss.

        Args:
            log_probs: (B,) or (B, S) tensor of per-sample (or per-token)
                       log-probabilities from the policy.  When shape is (B, S)
                       the mean over the sequence dimension is taken first to
                       produce a per-sample scalar before masking.
            rewards:   (B,) tensor of reward values; one per sample.
            threshold: Scalar float τ; samples with reward < τ are masked out.

        Returns:
            loss:       Scalar tensor (differentiable).
            n_accepted: Number of samples that passed the threshold (int).
        """
        if log_probs.dim() not in (1, 2):
            raise ValueError(f"log_probs must be 1-D or 2-D; got shape {log_probs.shape}")
        if rewards.dim() != 1:
            raise ValueError(f"rewards must be 1-D; got shape {rewards.shape}")
        if log_probs.shape[0] != rewards.shape[0]:
            raise ValueError(
                "Batch size mismatch: log_probs.shape[0]="
                f"{log_probs.shape[0]} vs rewards.shape[0]={rewards.shape[0]}"
            )

        # Reduce to per-sample log-prob if sequence dimension present
        if log_probs.dim() == 2:
            per_sample_lp = log_probs.mean(dim=1)  # (B,)
        else:
            per_sample_lp = log_probs  # (B,)

        mask = (rewards >= threshold).to(dtype=per_sample_lp.dtype)  # (B,)
        n_accepted = int(mask.sum().item())

        if n_accepted == 0:
            # Return a zero loss that is still part of the graph (for safety)
            loss = (per_sample_lp * 0.0).sum()
            return loss, 0

        # NLL = -log_prob; apply mask to zero out rejected samples
        nll = -per_sample_lp * mask  # (B,)

        if self.reduction == "mean":
            loss = nll.sum() / n_accepted
        else:  # "sum"
            loss = nll.sum()

        return loss, n_accepted


# ---------------------------------------------------------------------------
# ReSTTrainer
# ---------------------------------------------------------------------------


class ReSTTrainer:
    """High-level driver that wraps ReSTLoss and exposes dataset-grow helpers.

    Args:
        threshold_schedule: ReSTThresholdSchedule instance (or None to use
                            a default one).
        loss_fn:            ReSTLoss instance (or None to use a default one).
        n_per_prompt:       Default number of generations per prompt in grow.
    """

    def __init__(
        self,
        threshold_schedule: ReSTThresholdSchedule | None = None,
        loss_fn: ReSTLoss | None = None,
        n_per_prompt: int = 4,
    ) -> None:
        self.schedule = threshold_schedule or ReSTThresholdSchedule()
        self.loss_fn = loss_fn or ReSTLoss()
        self.n_per_prompt = n_per_prompt

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Thin wrapper around ReSTLoss.__call__ that returns only the loss.

        Args:
            log_probs: (B,) or (B, S) tensor of per-sample log-probabilities.
            rewards:   (B,) tensor of reward values.
            threshold: Scalar τ; samples with reward < τ are masked.

        Returns:
            Scalar loss tensor.
        """
        loss, _ = self.loss_fn(log_probs, rewards, threshold)
        return loss

    # ------------------------------------------------------------------
    def grow_dataset(
        self,
        dataset: ReSTDataset,
        n_per_prompt: int | None = None,
        vocab_size: int = 32,
        max_completion_len: int = 8,
        seed: int | None = None,
    ) -> ReSTDataset:
        """Augment dataset using a random mock generator (for testing / dry-runs).

        In production replace this with a real autoregressive generator.

        Args:
            dataset:           Input ReSTDataset.
            n_per_prompt:      Overrides self.n_per_prompt when provided.
            vocab_size:        Vocabulary size for random completions.
            max_completion_len: Max tokens in each random completion.
            seed:              Optional manual seed for reproducibility.

        Returns:
            New ReSTDataset with generated completions appended.
        """
        if seed is not None:
            torch.manual_seed(seed)

        n = n_per_prompt if n_per_prompt is not None else self.n_per_prompt

        def _mock_generator(
            prompt: torch.Tensor,
        ) -> tuple[torch.Tensor, float]:
            length = torch.randint(1, max_completion_len + 1, ()).item()
            completion = torch.randint(0, vocab_size, (int(length),))
            reward = torch.rand(()).item()
            return completion, reward

        return dataset.grow(_mock_generator, n_samples_per_prompt=n)
