"""Adaptive speculative decoding — Nightjar-inspired dynamic speculation.

Nightjar (arXiv:2512.22420) dynamically adjusts speculation length based on
the current acceptance rate, preventing wasted compute when draft quality
drops and maximizing throughput when draft quality is high.

Core idea:
- Track an exponential moving average (EMA) of the acceptance rate.
- If acceptance rate is high (> threshold), increase K (speculation length).
- If acceptance rate is low (< threshold), decrease K.
- This is a simple control loop that adapts to changing draft quality.

This module provides an adapter that wraps any speculative decoder with
the adaptive speculation length controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdaptiveSpecConfig:
    """Configuration for Nightjar-style adaptive speculation.

    Attributes:
        min_K: Minimum speculation length (default 1).
        max_K: Maximum speculation length (default 8).
        init_K: Initial speculation length (default 4).
        target_acceptance: Target acceptance rate (default 0.7).
            The controller adjusts K to maintain this rate.
        adapt_rate: How quickly K adapts to changes in acceptance rate
            (default 0.1). Higher = faster adaptation.
        ema_alpha: Smoothing factor for the EMA of acceptance rate
            (default 0.3). Higher = more responsive to recent changes.
    """

    min_K: int = 1
    max_K: int = 8
    init_K: int = 4
    target_acceptance: float = 0.7
    adapt_rate: float = 0.1
    ema_alpha: float = 0.3


class AdaptiveSpecController:
    """Nightjar-style adaptive speculation length controller.

    Tracks acceptance rate via EMA and adjusts K dynamically.

    Args:
        cfg: Adaptive speculation configuration.
    """

    def __init__(self, cfg: AdaptiveSpecConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveSpecConfig()
        self.K = self.cfg.init_K
        self._ema: float = self.cfg.target_acceptance
        self._steps = 0
        self._total_accepted = 0
        self._total_proposed = 0

    def update(self, accepted: int, proposed: int) -> int:
        """Update the controller with the latest acceptance statistics.

        Args:
            accepted: Number of tokens accepted in this round.
            proposed: Number of tokens proposed in this round (K).

        Returns:
            New K value for the next speculation round.
        """
        self._steps += 1
        self._total_accepted += accepted
        self._total_proposed += proposed

        if proposed == 0:
            return self.K

        # Instantaneous acceptance rate for this round
        round_rate = accepted / proposed

        # Update EMA
        self._ema = self.cfg.ema_alpha * round_rate + (1 - self.cfg.ema_alpha) * self._ema

        # Adjust K based on deviation from target
        error = self._ema - self.cfg.target_acceptance
        delta = round(self.cfg.adapt_rate * error * self.K)
        self.K = int(self.K + delta)
        self.K = max(self.cfg.min_K, min(self.cfg.max_K, self.K))

        return self.K

    @property
    def overall_acceptance(self) -> float:
        """Overall acceptance rate across all rounds."""
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    @property
    def ema_acceptance(self) -> float:
        """Smoothed acceptance rate (EMA)."""
        return self._ema

    def reset(self) -> None:
        """Reset the controller to initial state."""
        self.K = self.cfg.init_K
        self._ema = self.cfg.target_acceptance
        self._steps = 0
        self._total_accepted = 0
        self._total_proposed = 0


class AdaptiveSpeculativeDecoder:
    """Wraps a speculative decoder with Nightjar-style adaptive K.

    The controller monitors acceptance rate and adjusts speculation
    length dynamically, reducing wasted compute on low-quality drafts
    and maximizing throughput on high-quality drafts.

    Args:
        decoder: The underlying speculative decoder (must have a
            ``cfg`` attribute with ``K`` or ``tree_budget``).
        controller: The adaptive controller.
    """

    def __init__(
        self,
        decoder,
        controller: AdaptiveSpecController | None = None,
    ) -> None:
        self.decoder = decoder
        self.controller = controller or AdaptiveSpecController()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens with adaptive speculation length.

        Each round: controller picks K → decoder speculates →
        controller measures acceptance → adjusts K for next round.

        Args:
            input_ids: (1, prompt_len) input tokens.

        Returns:
            (1, prompt_len + generated) output tokens.
        """
        cfg = self.decoder.cfg
        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < cfg.max_new_tokens:
            K = self.controller.K
            # Set the speculation length on the decoder config
            if hasattr(cfg, "K"):
                cfg.K = K
            elif hasattr(cfg, "tree_budget"):
                # For tree-based decoders, adapt tree budget
                cfg.tree_budget = K + 2  # root + K children

            # Run one speculation round
            # Note: we capture the length before to measure acceptance
            start_len = generated.shape[1]
            generated = self._run_one_round(generated, K)
            end_len = generated.shape[1]

            accepted = end_len - start_len - 1  # -1 for bonus token
            if accepted < 0:
                accepted = 0
            proposed = K

            # Update controller with results
            self.controller.update(accepted, proposed)

            tokens_generated = end_len - input_ids.shape[1]

        return generated

    def _run_one_round(self, input_ids: torch.Tensor, K: int) -> torch.Tensor:
        """Run a single speculation round with the underlying decoder.

        Args:
            input_ids: Current input.
            K: Speculation length for this round.

        Returns:
            Extended input with accepted tokens.
        """
        # Save original max_new_tokens
        orig_max = self.decoder.cfg.max_new_tokens

        # Set to K+1 to generate just one round
        self.decoder.cfg.max_new_tokens = K + 1
        result = self.decoder.generate(input_ids)

        # Restore
        self.decoder.cfg.max_new_tokens = orig_max
        return result
