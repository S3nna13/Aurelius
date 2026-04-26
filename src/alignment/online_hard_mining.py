"""Aurelius — Online Hard Example Mining (OHEM) for RLHF/alignment training.

In alignment training (DPO, RLHF), not all preference pairs are equally
informative. Hard examples are those where the policy nearly assigns equal
probability to chosen vs rejected, the reward gap is small, or the model
gets the wrong answer with high confidence. Mining and upweighting these
examples accelerates alignment convergence.

Pure PyTorch implementation — no scipy, no sklearn, no HuggingFace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HardExample:
    """A single hard preference pair with associated difficulty score."""

    chosen_ids: torch.Tensor  # (T,)
    rejected_ids: torch.Tensor  # (T,)
    difficulty: float  # higher = harder
    weight: float = 1.0  # sampling weight


@dataclass
class MiningConfig:
    """Configuration for online hard example mining."""

    strategy: str = "hardest"  # "hardest" | "semi-hard" | "curriculum"
    top_k_ratio: float = 0.5  # keep top-K hardest (fraction of batch)
    curriculum_warmup_steps: int = 100  # start easy, increase difficulty
    temperature: float = 1.0  # for soft weighting
    min_difficulty: float = 0.0  # ignore examples below this difficulty
    max_buffer_size: int = 1000  # max examples in memory bank


# ---------------------------------------------------------------------------
# Difficulty computation
# ---------------------------------------------------------------------------


def compute_dpo_difficulty(
    policy_logprobs_chosen: torch.Tensor,  # (B,)
    policy_logprobs_rejected: torch.Tensor,  # (B,)
    ref_logprobs_chosen: torch.Tensor,  # (B,)
    ref_logprobs_rejected: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """Compute DPO-based difficulty for each example in a batch.

    difficulty = sigmoid(-(log_ratio_chosen - log_ratio_rejected))
    where log_ratio = log(policy) - log(ref)

    Higher difficulty means the policy is close to (or already) choosing
    the rejected response over the chosen one.

    Returns (B,) tensor in [0, 1].
    """
    log_ratio_chosen = policy_logprobs_chosen - ref_logprobs_chosen
    log_ratio_rejected = policy_logprobs_rejected - ref_logprobs_rejected
    margin = log_ratio_chosen - log_ratio_rejected
    difficulty = torch.sigmoid(-margin)
    return difficulty


def compute_reward_difficulty(
    rewards_chosen: torch.Tensor,  # (B,)
    rewards_rejected: torch.Tensor,  # (B,)
    normalize: bool = True,
) -> torch.Tensor:
    """Compute reward-based difficulty for each example in a batch.

    difficulty = 1 - sigmoid(rewards_chosen - rewards_rejected)

    Small reward gap → difficulty close to 0.5; large gap → close to 0.

    Returns (B,) tensor in [0, 1].
    """
    gap = rewards_chosen - rewards_rejected
    if normalize:
        # Optionally normalise gap to zero-mean unit-variance for stability.
        if gap.numel() > 1:
            gap = (gap - gap.mean()) / (gap.std() + 1e-8)
    difficulty = 1.0 - torch.sigmoid(gap)
    return difficulty


# ---------------------------------------------------------------------------
# Index selection
# ---------------------------------------------------------------------------


def select_hard_examples(
    difficulties: torch.Tensor,  # (B,)
    strategy: str = "hardest",
    top_k_ratio: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select indices of hard examples from a batch.

    Strategies
    ----------
    "hardest"   : select the top-k by difficulty score.
    "semi-hard" : select examples with difficulty in [0.3, 0.7].
    "weighted"  : sample proportional to exp(difficulty / temperature).

    Returns (k,) tensor of selected indices.
    """
    B = difficulties.shape[0]
    k = max(1, int(B * top_k_ratio))

    if strategy == "hardest":
        _, indices = torch.topk(difficulties, k=k)
        return indices

    elif strategy == "semi-hard":
        mask = (difficulties >= 0.3) & (difficulties <= 0.7)
        candidates = mask.nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            # Fall back to top-k when no semi-hard examples exist.
            _, indices = torch.topk(difficulties, k=k)
            return indices
        # Return up to k semi-hard examples.
        if candidates.numel() > k:
            perm = torch.randperm(candidates.numel())[:k]
            candidates = candidates[perm]
        return candidates

    elif strategy == "weighted":
        weights = torch.exp(difficulties / temperature)
        weights = weights / weights.sum()
        indices = torch.multinomial(weights, num_samples=k, replacement=False)
        return indices

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from 'hardest', 'semi-hard', 'weighted'."
        )


# ---------------------------------------------------------------------------
# Memory buffer
# ---------------------------------------------------------------------------


class HardExampleBuffer:
    """Memory bank of hard examples for replay during alignment training.

    Maintains a bounded collection of the hardest seen examples and
    supports weighted sampling for replay.
    """

    def __init__(self, max_size: int = 1000, strategy: str = "hardest") -> None:
        self._max_size = max_size
        self._strategy = strategy
        self._examples: list[HardExample] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, examples: list[HardExample]) -> None:
        """Add examples to the buffer, evicting the easiest when over capacity."""
        self._examples.extend(examples)
        if len(self._examples) > self._max_size:
            # Keep the hardest examples (highest difficulty).
            self._examples.sort(key=lambda e: e.difficulty, reverse=True)
            self._examples = self._examples[: self._max_size]

    def update_difficulties(self, update_fn: Callable[[float], float]) -> None:
        """Apply update_fn to every example's difficulty (e.g., temporal decay)."""
        for ex in self._examples:
            ex.difficulty = update_fn(ex.difficulty)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> list[HardExample]:
        """Sample n examples, weighted by difficulty score."""
        if not self._examples:
            return []
        n = min(n, len(self._examples))
        difficulties = torch.tensor([ex.difficulty for ex in self._examples], dtype=torch.float32)
        # Soft-max weighting: harder examples get higher probability.
        weights = difficulties - difficulties.min() + 1e-6
        weights = weights / weights.sum()
        indices = torch.multinomial(weights, num_samples=n, replacement=False)
        return [self._examples[i.item()] for i in indices]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._examples)


# ---------------------------------------------------------------------------
# Main miner class
# ---------------------------------------------------------------------------


class OnlineHardMiner:
    """Compute difficulties, select hard examples, and maintain a replay buffer.

    Integrates with DPO / RLHF training loops to dynamically focus
    training on the most informative preference pairs.
    """

    def __init__(
        self,
        config: MiningConfig | None = None,
        use_reward_model: bool = False,
    ) -> None:
        self._config = config or MiningConfig()
        self._use_reward_model = use_reward_model
        self._step: int = 0
        self.buffer = HardExampleBuffer(
            max_size=self._config.max_buffer_size,
            strategy=self._config.strategy,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def mine(
        self,
        chosen_ids: torch.Tensor,  # (B, T)
        rejected_ids: torch.Tensor,  # (B, T)
        policy_logprobs_chosen: torch.Tensor,  # (B,)
        policy_logprobs_rejected: torch.Tensor,  # (B,)
        ref_logprobs_chosen: torch.Tensor | None = None,  # (B,)
        ref_logprobs_rejected: torch.Tensor | None = None,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard-example subset of (chosen_ids, rejected_ids).

        Also stores the selected hard examples in the replay buffer and
        increments the internal step counter.

        Returns
        -------
        chosen_hard   : (k, T)
        rejected_hard : (k, T)
        """
        cfg = self._config

        # --- compute difficulties ---
        if ref_logprobs_chosen is not None and ref_logprobs_rejected is not None:
            difficulties = compute_dpo_difficulty(
                policy_logprobs_chosen,
                policy_logprobs_rejected,
                ref_logprobs_chosen,
                ref_logprobs_rejected,
            )
        else:
            # Fallback: treat raw policy margin as proxy difficulty.
            margin = policy_logprobs_chosen - policy_logprobs_rejected
            difficulties = torch.sigmoid(-margin)

        # --- apply minimum difficulty filter ---
        if cfg.min_difficulty > 0.0:
            valid = difficulties >= cfg.min_difficulty
            if valid.any():
                chosen_ids = chosen_ids[valid]
                rejected_ids = rejected_ids[valid]
                difficulties = difficulties[valid]

        # --- curriculum gate ---
        if cfg.strategy == "curriculum":
            threshold = self.curriculum_difficulty_threshold()
            valid = difficulties <= threshold
            if valid.any():
                chosen_ids = chosen_ids[valid]
                rejected_ids = rejected_ids[valid]
                difficulties = difficulties[valid]
            # After filtering, pick the hardest within the allowed range.
            strategy_for_select = "hardest"
        else:
            strategy_for_select = cfg.strategy

        # --- select hard indices ---
        indices = select_hard_examples(
            difficulties,
            strategy=strategy_for_select,
            top_k_ratio=cfg.top_k_ratio,
            temperature=cfg.temperature,
        )

        chosen_hard = chosen_ids[indices]
        rejected_hard = rejected_ids[indices]
        hard_difficulties = difficulties[indices]

        # --- store in buffer ---
        examples = [
            HardExample(
                chosen_ids=chosen_hard[i].detach().cpu(),
                rejected_ids=rejected_hard[i].detach().cpu(),
                difficulty=hard_difficulties[i].item(),
            )
            for i in range(chosen_hard.shape[0])
        ]
        self.buffer.add(examples)

        self._step += 1
        return chosen_hard, rejected_hard

    def get_sample_weights(
        self,
        difficulties: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Compute per-example loss weights based on difficulty.

        Harder examples receive higher weight. Weights are normalised so
        that they sum to B (the batch size).

        Returns (B,) tensor.
        """
        temp = self._config.temperature
        weights = torch.exp(difficulties / temp)
        # Normalise to sum to B.
        weights = weights / weights.sum() * difficulties.shape[0]
        return weights

    # ------------------------------------------------------------------
    # Curriculum scheduling
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current training step (for curriculum scheduling)."""
        return self._step

    def curriculum_difficulty_threshold(self) -> float:
        """Return the maximum difficulty allowed at the current step.

        Linearly increases from 0 → 1 over curriculum_warmup_steps,
        then stays at 1.0 (all examples allowed).
        """
        warmup = self._config.curriculum_warmup_steps
        if warmup <= 0:
            return 1.0
        progress = min(self._step / warmup, 1.0)
        return float(progress)


# ---------------------------------------------------------------------------
# DPO loss with hard example mining
# ---------------------------------------------------------------------------


def dpo_loss_with_mining(
    policy_chosen_logps: torch.Tensor,  # (B,)
    policy_rejected_logps: torch.Tensor,  # (B,)
    ref_chosen_logps: torch.Tensor,  # (B,)
    ref_rejected_logps: torch.Tensor,  # (B,)
    beta: float = 0.1,
    mining_config: MiningConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss with online hard example mining and difficulty-based weighting.

    Selects hard examples from the batch, upweights them in the loss, and
    returns training metrics.

    Parameters
    ----------
    policy_chosen_logps    : log-probs of chosen responses under policy  (B,)
    policy_rejected_logps  : log-probs of rejected responses under policy (B,)
    ref_chosen_logps       : log-probs of chosen responses under reference (B,)
    ref_rejected_logps     : log-probs of rejected responses under reference (B,)
    beta                   : KL penalty coefficient (standard DPO beta)
    mining_config          : if None, uses default MiningConfig (no mining)

    Returns
    -------
    loss    : scalar tensor
    metrics : dict with 'n_hard_examples', 'mean_difficulty', 'loss_unweighted'
    """
    cfg = mining_config or MiningConfig()

    # --- compute difficulties ---
    difficulties = compute_dpo_difficulty(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    # --- select hard indices ---
    indices = select_hard_examples(
        difficulties,
        strategy=cfg.strategy if cfg.strategy != "curriculum" else "hardest",
        top_k_ratio=cfg.top_k_ratio,
        temperature=cfg.temperature,
    )

    # Subset to hard examples.
    h_policy_chosen = policy_chosen_logps[indices]
    h_policy_rejected = policy_rejected_logps[indices]
    h_ref_chosen = ref_chosen_logps[indices]
    h_ref_rejected = ref_rejected_logps[indices]
    h_difficulties = difficulties[indices]

    # --- standard DPO log-ratio margins ---
    log_ratio_chosen = h_policy_chosen - h_ref_chosen
    log_ratio_rejected = h_policy_rejected - h_ref_rejected
    margin = beta * (log_ratio_chosen - log_ratio_rejected)

    # --- per-example DPO loss ---
    per_example_loss = -F.logsigmoid(margin)

    # --- difficulty-based weighting ---
    temp = cfg.temperature
    weights = torch.exp(h_difficulties / temp)
    weights = weights / weights.sum() * h_difficulties.shape[0]

    # --- weighted loss ---
    loss = (weights * per_example_loss).mean()

    # --- metrics ---
    metrics: dict[str, float] = {
        "n_hard_examples": float(indices.shape[0]),
        "mean_difficulty": float(h_difficulties.mean().item()),
        "loss_unweighted": float(per_example_loss.mean().item()),
        "mean_weight": float(weights.mean().item()),
    }

    return loss, metrics
