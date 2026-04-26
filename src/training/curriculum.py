"""Token-level loss weighting and epoch-based curriculum learning."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class WeightMode(Enum):
    UNIFORM = "uniform"  # no weighting (returns plain mean)
    POSITION = "position"  # later positions get higher weight
    FREQUENCY = "frequency"  # rare tokens get higher weight
    CUSTOM = "custom"  # caller provides weight tensor


@dataclass
class TokenCurriculumConfig:
    """Config for token-level loss weighting (original TokenWeighter config)."""

    mode: WeightMode = WeightMode.UNIFORM
    position_exponent: float = 1.0  # weight = (pos / seq_len) ** exponent
    freq_smoothing: float = 0.5  # add this to counts before inverting
    normalize_weights: bool = True  # normalize weights to sum to 1 before applying


class TokenWeighter:
    """Applies curriculum weighting to per-token losses.

    Usage:
        weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.POSITION))
        loss = weighter(per_token_loss, input_ids=input_ids)
    """

    def __init__(self, cfg: TokenCurriculumConfig | None = None):
        self.cfg = cfg or TokenCurriculumConfig()

    def __call__(
        self,
        per_token_loss: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        token_counts: torch.Tensor | None = None,
        custom_weights: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            per_token_loss: (B, S) per-token losses
            input_ids: (B, S) token IDs -- required for FREQUENCY mode
            token_counts: (vocab_size,) -- token frequency counts for FREQUENCY mode.
                          If None in FREQUENCY mode, falls back to uniform.
            custom_weights: (B, S) or (S,) custom weight tensor for CUSTOM mode
            padding_mask: (B, S) bool, True = valid. Padded positions excluded from mean.

        Returns:
            Scalar weighted mean loss.
        """
        B, S = per_token_loss.shape
        mode = self.cfg.mode

        # UNIFORM mode: simple mean with optional masking
        if mode == WeightMode.UNIFORM:
            if padding_mask is not None:
                return per_token_loss[padding_mask].mean()
            return per_token_loss.mean()

        # Build raw weights based on mode
        if mode == WeightMode.POSITION:
            weights = self._position_weights(B, S, per_token_loss.device)  # (1, S)
            weights = weights.expand(B, S)
        elif mode == WeightMode.FREQUENCY:
            if token_counts is None or input_ids is None:
                # Fall back to uniform
                if padding_mask is not None:
                    return per_token_loss[padding_mask].mean()
                return per_token_loss.mean()
            weights = self._frequency_weights(input_ids, token_counts)  # (B, S)
        elif mode == WeightMode.CUSTOM:
            if custom_weights is None:
                raise ValueError("custom_weights required for CUSTOM mode")
            weights = custom_weights
            if weights.ndim == 1:
                weights = weights.unsqueeze(0).expand(B, S)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply padding mask: zero out padded positions
        if padding_mask is not None:
            weights = weights * padding_mask.float()

        # Normalize weights to sum to 1 over valid positions
        if self.cfg.normalize_weights:
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum

        return (per_token_loss * weights).sum()

    def _position_weights(self, B: int, S: int, device: torch.device) -> torch.Tensor:
        """Return (1, S) position weights: ((pos+1)/S) ** exponent."""
        positions = torch.arange(1, S + 1, device=device).float()
        weights = (positions / S) ** self.cfg.position_exponent
        return weights.unsqueeze(0)  # (1, S)

    def _frequency_weights(
        self, input_ids: torch.Tensor, token_counts: torch.Tensor
    ) -> torch.Tensor:
        """Return (B, S) inverse-frequency weights for each token in input_ids."""
        counts = token_counts.float()[input_ids]  # (B, S)
        inv_freq = 1.0 / (counts + self.cfg.freq_smoothing)
        return inv_freq


# ---------------------------------------------------------------------------
# Epoch-based curriculum learning
# ---------------------------------------------------------------------------

_MAX_SEQ_LEN = 512  # default normalization constant for difficulty_score


@dataclass
class CurriculumConfig:
    """Configuration for epoch-based curriculum learning.

    Attributes:
        strategy: pacing strategy, one of "linear" or "root"
        n_epochs: total number of training epochs
        start_difficulty: difficulty threshold at epoch 0
        end_difficulty: difficulty threshold at epoch n_epochs
        warmup_epochs: epochs to hold start_difficulty before ramping
        competence_threshold: fraction of dataset considered "mastered" to advance
    """

    strategy: str = "linear"
    n_epochs: int = 10
    start_difficulty: float = 0.0
    end_difficulty: float = 1.0
    warmup_epochs: int = 2
    competence_threshold: float = 0.9


def difficulty_score(tokens: torch.Tensor, max_seq_len: int = _MAX_SEQ_LEN) -> float:
    """Compute difficulty of a sample as normalised sequence length.

    Args:
        tokens: 1-D or 2-D tensor of token IDs.
        max_seq_len: normalisation constant (default 512).

    Returns:
        Float in [0, 1].
    """
    seq_len = tokens.numel() if tokens.ndim == 1 else tokens.shape[-1]
    return min(float(seq_len) / max_seq_len, 1.0)


def linear_curriculum(epoch: int, config: CurriculumConfig) -> float:
    """Return difficulty threshold that grows linearly from start to end.

    During warmup_epochs the threshold stays at start_difficulty.
    After warmup it ramps linearly to end_difficulty over the remaining epochs.

    Args:
        epoch: current epoch index (0-based).
        config: CurriculumConfig instance.

    Returns:
        Difficulty threshold float in [start_difficulty, end_difficulty].
    """
    if config.n_epochs <= 0:
        return config.end_difficulty
    if epoch <= config.warmup_epochs:
        return config.start_difficulty
    ramp_epochs = max(config.n_epochs - config.warmup_epochs, 1)
    t = min((epoch - config.warmup_epochs) / ramp_epochs, 1.0)
    return config.start_difficulty + (config.end_difficulty - config.start_difficulty) * t


def root_curriculum(epoch: int, config: CurriculumConfig) -> float:
    """Return difficulty threshold that follows a square-root schedule.

    threshold = start + (end - start) * sqrt(epoch / n_epochs)

    Args:
        epoch: current epoch index (0-based).
        config: CurriculumConfig instance.

    Returns:
        Difficulty threshold float in [start_difficulty, end_difficulty].
    """
    if config.n_epochs <= 0:
        return config.end_difficulty
    t = math.sqrt(min(epoch / config.n_epochs, 1.0))
    return config.start_difficulty + (config.end_difficulty - config.start_difficulty) * t


def filter_by_difficulty(
    dataset: list[Any],
    threshold: float,
    score_fn: Callable[[Any], float],
) -> list[Any]:
    """Return samples whose difficulty score is <= threshold.

    Args:
        dataset: list of samples (anything).
        threshold: maximum allowed difficulty.
        score_fn: callable that maps a sample to a float in [0, 1].

    Returns:
        Filtered list of samples.
    """
    return [sample for sample in dataset if score_fn(sample) <= threshold]


class CurriculumSampler:
    """Wraps a dataset and exposes only samples within the current difficulty threshold.

    Args:
        dataset: list of samples.
        score_fn: callable mapping a sample to a difficulty float in [0, 1].
        config: CurriculumConfig controlling the pacing schedule.
    """

    def __init__(
        self,
        dataset: list[Any],
        score_fn: Callable[[Any], float],
        config: CurriculumConfig,
    ) -> None:
        self.dataset = dataset
        self.score_fn = score_fn
        self.config = config
        self._epoch: int = 0
        self._threshold: float = self._compute_threshold(0)
        self._eligible: list[Any] = self._filter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update internal epoch and recompute eligible samples."""
        self._epoch = epoch
        self._threshold = self._compute_threshold(epoch)
        self._eligible = self._filter()

    def __len__(self) -> int:
        return len(self._eligible)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._eligible)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_threshold(self, epoch: int) -> float:
        if self.config.strategy == "root":
            return root_curriculum(epoch, self.config)
        return linear_curriculum(epoch, self.config)

    def _filter(self) -> list[Any]:
        return filter_by_difficulty(self.dataset, self._threshold, self.score_fn)


class CurriculumTrainer:
    """Thin training wrapper that advances curriculum difficulty per epoch.

    Args:
        model: AureliusTransformer (or any model following the ``loss, logits, pkv = model(input_ids)`` API).
        optimizer: PyTorch optimizer.
        config: CurriculumConfig.
    """  # noqa: E501

    def __init__(self, model: Any, optimizer: Any, config: CurriculumConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self._epoch: int = 0

    @property
    def current_difficulty(self) -> float:
        """Difficulty threshold for the current epoch."""
        if self.config.strategy == "root":
            return root_curriculum(self._epoch, self.config)
        return linear_curriculum(self._epoch, self.config)

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Run one forward/backward/optimiser step.

        Args:
            input_ids: (B, S) integer tensor.

        Returns:
            dict with keys ``"loss"`` (float) and ``"current_difficulty"`` (float).
        """
        self.optimizer.zero_grad()
        # Pass input_ids as both input and labels; the model shifts labels internally.
        loss, _logits, _pkv = self.model(input_ids, labels=input_ids)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "current_difficulty": self.current_difficulty,
        }

    def advance_epoch(self) -> None:
        """Increment the internal epoch counter."""
        self._epoch += 1
