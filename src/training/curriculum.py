"""Token-level loss weighting for curriculum learning."""

import torch
from dataclasses import dataclass
from enum import Enum


class WeightMode(Enum):
    UNIFORM = "uniform"          # no weighting (returns plain mean)
    POSITION = "position"        # later positions get higher weight
    FREQUENCY = "frequency"      # rare tokens get higher weight
    CUSTOM = "custom"            # caller provides weight tensor


@dataclass
class CurriculumConfig:
    mode: WeightMode = WeightMode.UNIFORM
    position_exponent: float = 1.0   # weight = (pos / seq_len) ** exponent
    freq_smoothing: float = 0.5      # add this to counts before inverting
    normalize_weights: bool = True   # normalize weights to sum to 1 before applying


class TokenWeighter:
    """Applies curriculum weighting to per-token losses.

    Usage:
        weighter = TokenWeighter(CurriculumConfig(mode=WeightMode.POSITION))
        loss = weighter(per_token_loss, input_ids=input_ids)
    """

    def __init__(self, cfg: CurriculumConfig | None = None):
        self.cfg = cfg or CurriculumConfig()

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
