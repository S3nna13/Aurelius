"""Aurelius — Zero-Vision SFT Trainer (Kimi K2.5 §4, arXiv:2602.02276).

Key insight: Text-only SFT activates visual reasoning through programmatic
proxy ops (crop, detect, etc.) without requiring real image data.
Tool-call tokens are upweighted in the cross-entropy loss because they carry
the visual-reasoning signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ZeroVisionSFTConfig:
    """Configuration for ZeroVisionSFTTrainer.

    Attributes:
        tool_call_weight: Loss multiplier applied to tool-call token positions.
        pad_id: Token id used for padding; pad positions are excluded from loss.
        tool_call_token_ids: List of token ids that should be upweighted.
            If None or empty, no upweighting is applied (all-zeros mask).
    """

    tool_call_weight: float = 3.0
    pad_id: int = 0
    tool_call_token_ids: list[int] | None = field(default=None)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ZeroVisionSFTTrainer:
    """Text-only SFT trainer with tool-call token upweighting.

    Not an nn.Module — this is a stateless training utility.  The model
    itself is passed in via ``model_output`` / ``batch`` dicts so that the
    trainer stays decoupled from the model architecture.
    """

    def __init__(self, config: ZeroVisionSFTConfig | None = None) -> None:
        if config is None:
            config = ZeroVisionSFTConfig()
        self.config = config

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tool_call_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Float tensor of shape ``[B, T, V]`` — raw model logits.
            targets: Long tensor of shape ``[B, T]`` — target token ids.
            tool_call_mask: Float/bool tensor of shape ``[B, T]``.
                Positions where the value is 1 (True) are multiplied by
                ``config.tool_call_weight`` in the loss.

        Returns:
            Scalar loss tensor (averaged over non-pad tokens).
        """
        B, T, V = logits.shape

        # Flatten to (B*T, V) and (B*T,) for F.cross_entropy
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)
        mask_flat = tool_call_mask.view(B * T).float()

        # Non-pad mask
        pad_mask = (targets_flat != self.config.pad_id).float()  # 1 where not pad

        # Per-token CE loss (no reduction)
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Zero out pad positions
        per_token_loss = per_token_loss * pad_mask

        # Build per-token weight: base weight 1.0, upweight tool-call positions
        weights = 1.0 + (self.config.tool_call_weight - 1.0) * mask_flat

        weighted_loss = per_token_loss * weights

        n_valid = pad_mask.sum()
        if n_valid == 0:
            # All positions are pad — return zero loss
            return weighted_loss.sum() * 0.0

        # Average over non-pad tokens (use weighted count for denominator consistency)
        # We use plain sum / n_valid (number of non-pad tokens) so that the
        # upweighting increases the loss magnitude rather than being normalised away.
        loss = weighted_loss.sum() / n_valid
        return loss

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def make_tool_call_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Build a binary mask for tool-call token positions.

        Args:
            token_ids: Long tensor of shape ``[B, T]``.

        Returns:
            Float tensor of shape ``[B, T]`` with 1.0 at positions where
            ``token_ids`` appears in ``config.tool_call_token_ids``, 0.0
            elsewhere.  Returns all-zeros if ``tool_call_token_ids`` is
            None or empty.
        """
        mask = torch.zeros_like(token_ids, dtype=torch.float32)

        tc_ids = self.config.tool_call_token_ids
        if not tc_ids:
            return mask

        for tid in tc_ids:
            mask = mask + (token_ids == tid).float()

        # Clamp to [0, 1] in case a token appears in the list more than once
        return mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(
        self,
        model_output: dict,
        batch: dict,
    ) -> dict:
        """Perform a single loss-computation step.

        Args:
            model_output: Must contain ``"logits"`` — Float tensor ``[B, T, V]``.
            batch: Must contain:
                - ``"input_ids"`` — Long tensor ``[B, T]``
                - ``"labels"`` — Long tensor ``[B, T]``
                - ``"tool_call_mask"`` — Float/bool tensor ``[B, T]``

        Returns:
            Dict with:
                - ``"loss"`` — scalar Tensor
                - ``"n_tokens"`` — int, number of non-pad positions in labels
                - ``"n_tool_call_tokens"`` — int, number of tool-call positions
                  that are also non-pad
        """
        logits: torch.Tensor = model_output["logits"]
        labels: torch.Tensor = batch["labels"]
        tool_call_mask: torch.Tensor = batch["tool_call_mask"]

        loss = self.compute_loss(logits, labels, tool_call_mask)

        # Count non-pad tokens in labels
        non_pad = labels != self.config.pad_id
        n_tokens: int = int(non_pad.sum().item())

        # Count tool-call tokens that are also non-pad
        tc_mask_bool = tool_call_mask.bool() & non_pad
        n_tool_call_tokens: int = int(tc_mask_bool.sum().item())

        return {
            "loss": loss,
            "n_tokens": n_tokens,
            "n_tool_call_tokens": n_tool_call_tokens,
        }
