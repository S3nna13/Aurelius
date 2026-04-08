"""Process Reward Model: scores each reasoning step individually."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PRMConfig:
    d_model: int = 1024
    step_token_id: int = 2  # token ID that marks end of each step (e.g. newline)
    freeze_backbone: bool = True


class ProcessRewardModel(nn.Module):
    """Scores each reasoning step in a sequence.

    Architecture:
    - Backbone: AureliusTransformer (used for hidden states only, not logits)
    - Step scorer: Linear(d_model, 1) applied at each step-end token position

    For each position where input_ids == step_token_id,
    extract the hidden state and score it.
    """

    def __init__(self, backbone: nn.Module, cfg: PRMConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.step_scorer = nn.Linear(cfg.d_model, 1, bias=True)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, S)
            step_mask: (B, S) bool, True at positions that are step boundaries.
                       If None, derived from input_ids == cfg.step_token_id.

        Returns dict with:
            "step_scores": (B, max_steps) step scores padded with -inf
            "step_counts": (B,) number of steps per sequence
            "hidden_states": (B, S, d_model) hidden states from backbone
        """
        # Extract hidden states via forward hook on backbone.norm
        hidden_states: list[torch.Tensor] = []
        hook = self.backbone.norm.register_forward_hook(
            lambda m, i, o: hidden_states.append(o)
        )
        try:
            with torch.set_grad_enabled(not self.cfg.freeze_backbone):
                self.backbone(input_ids)
        finally:
            hook.remove()

        h = hidden_states[0]  # (B, S, d_model)

        # Determine step positions
        if step_mask is None:
            step_mask = input_ids == self.cfg.step_token_id  # (B, S)

        B, S = input_ids.shape
        max_steps = step_mask.sum(dim=1).max().item()
        max_steps = max(max_steps, 1)

        step_scores_padded = torch.full(
            (B, max_steps), float("-inf"), device=input_ids.device
        )
        step_counts = torch.zeros(B, dtype=torch.long, device=input_ids.device)

        for b in range(B):
            positions = step_mask[b].nonzero().squeeze(1)  # step boundary positions
            if positions.numel() == 0:
                continue
            step_h = h[b, positions, :]  # (n_steps, d_model)
            scores = self.step_scorer(step_h).squeeze(-1)  # (n_steps,)
            n = scores.shape[0]
            step_scores_padded[b, :n] = scores
            step_counts[b] = n

        return {
            "step_scores": step_scores_padded,
            "step_counts": step_counts,
            "hidden_states": h,
        }


def prm_loss(
    step_scores: torch.Tensor,
    step_labels: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for step correctness.

    Args:
        step_scores: (B, max_steps) raw logits from step_scorer
        step_labels: (B, max_steps) binary labels (1=correct, 0=incorrect)
        step_mask: (B, max_steps) bool, True for valid (non-padded) steps

    Returns:
        Scalar BCE loss over valid steps.
    """
    valid_scores = step_scores[step_mask]
    valid_labels = step_labels[step_mask].float()
    return F.binary_cross_entropy_with_logits(valid_scores, valid_labels)
