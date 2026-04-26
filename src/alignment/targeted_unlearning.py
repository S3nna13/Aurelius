"""Aurelius -- Targeted Machine Unlearning (SCRUB + gradient ascent / KL retention).

Pure PyTorch implementation of targeted unlearning that combines:
  - Gradient ascent on a "forget" set to increase loss on unwanted knowledge
  - KL divergence retention on a "retain" set to preserve desired knowledge
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UnlearningConfig:
    """Configuration for targeted machine unlearning."""

    forget_lr: float = 1e-4  # learning rate for forget step
    retain_lr: float = 1e-5  # learning rate for retain step
    forget_steps: int = 5  # gradient ascent steps on forget set
    retain_steps: int = 5  # KL retention steps on retain set
    kl_coef: float = 1.0  # weight for KL retention loss
    max_grad_norm: float = 1.0  # gradient clipping
    forget_loss_type: str = "gradient_ascent"  # "gradient_ascent" or "random_label"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def forget_loss(
    model: nn.Module,
    forget_inputs: Tensor,  # (B, T) token ids
    forget_targets: Tensor,  # (B, T) token ids for next-token prediction
    loss_type: str = "gradient_ascent",
) -> Tensor:
    """Compute the forget loss.

    - gradient_ascent: negate the standard cross-entropy loss
    - random_label: cross-entropy against uniform random labels

    Returns scalar loss to maximize forgetting (positive scalar encourages
    gradient *ascent* on the standard CE loss).
    """
    out = model(forget_inputs)
    logits = out[1] if isinstance(out, tuple) else out  # (B, T, vocab_size)

    B, T, V = logits.shape

    if loss_type == "gradient_ascent":
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_targets = forget_targets[:, 1:].contiguous()  # (B, T-1)
        ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_targets.view(-1),
        )
        # Negate so minimising this loss maximises the original CE
        return -ce

    elif loss_type == "random_label":
        # Cross-entropy against uniform random targets
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        random_targets = torch.randint(0, V, (B * (T - 1),), device=forget_inputs.device)
        return F.cross_entropy(shift_logits.view(-1, V), random_targets)

    else:
        raise ValueError(
            f"Unknown forget_loss_type: {loss_type!r}. Choose 'gradient_ascent' or 'random_label'."
        )


def retain_loss(
    model: nn.Module,
    ref_model: nn.Module,  # frozen reference model
    retain_inputs: Tensor,  # (B, T) token ids
) -> Tensor:
    """KL divergence retention loss: KL(model || ref_model) on retain set.

    Ensures model doesn't catastrophically forget retained knowledge.
    Returns scalar KL loss (non-negative).
    """
    # Current model logits
    out = model(retain_inputs)
    logits = out[1] if isinstance(out, tuple) else out  # (B, T, V)

    # Reference model logits (no grad)
    with torch.no_grad():
        ref_out = ref_model(retain_inputs)
        ref_logits = ref_out[1] if isinstance(ref_out, tuple) else ref_out  # (B, T, V)

    B, T, V = logits.shape

    log_probs = F.log_softmax(logits.view(-1, V), dim=-1)  # (B*T, V)
    ref_probs = F.softmax(ref_logits.view(-1, V), dim=-1)  # (B*T, V)

    # kl_div(input, target) computes mean(target * (log(target) - input))
    # which equals KL(ref || model) >= 0
    kl = F.kl_div(log_probs, ref_probs, reduction="batchmean")
    return kl


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class UnlearningResult:
    """Result of a targeted unlearning run."""

    forget_losses: list[float]  # forget loss per step
    retain_losses: list[float]  # retain loss per step
    n_forget_steps: int
    n_retain_steps: int


# ---------------------------------------------------------------------------
# TargetedUnlearner
# ---------------------------------------------------------------------------


class TargetedUnlearner:
    """Orchestrates targeted machine unlearning via SCRUB-style alternation."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: UnlearningConfig,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.forget_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.forget_lr)
        self.retain_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.retain_lr)

    def forget_step(
        self,
        forget_inputs: Tensor,
        forget_targets: Tensor,
    ) -> float:
        """One gradient ascent step on forget set. Returns forget loss."""
        self.model.train()
        self.forget_optimizer.zero_grad()

        loss = forget_loss(
            self.model,
            forget_inputs,
            forget_targets,
            loss_type=self.config.forget_loss_type,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.forget_optimizer.step()

        return loss.item()

    def retain_step(
        self,
        retain_inputs: Tensor,
    ) -> float:
        """One KL retention step on retain set. Returns retain KL loss."""
        self.model.train()
        self.retain_optimizer.zero_grad()

        loss = self.config.kl_coef * retain_loss(self.model, self.ref_model, retain_inputs)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.retain_optimizer.step()

        return loss.item()

    def run(
        self,
        forget_dataset: list[tuple[Tensor, Tensor]],  # list of (inputs, targets)
        retain_dataset: list[Tensor],  # list of inputs (no targets for KL)
    ) -> UnlearningResult:
        """Run full unlearning: alternate forget and retain steps."""
        forget_losses: list[float] = []
        retain_losses: list[float] = []

        cfg = self.config
        n_forget = cfg.forget_steps
        n_retain = cfg.retain_steps

        forget_iter = _cycle(forget_dataset)
        retain_iter = _cycle(retain_dataset)

        for _ in range(n_forget):
            inputs, targets = next(forget_iter)
            fl = self.forget_step(inputs, targets)
            forget_losses.append(fl)

        for _ in range(n_retain):
            inputs = next(retain_iter)
            rl = self.retain_step(inputs)
            retain_losses.append(rl)

        return UnlearningResult(
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            n_forget_steps=n_forget,
            n_retain_steps=n_retain,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_forgetting(
    model: nn.Module,
    forget_inputs: Tensor,
    forget_targets: Tensor,
) -> dict[str, float]:
    """Measure how well the model has forgotten.

    Returns: {'forget_loss': float, 'forget_perplexity': float}
    Higher loss / perplexity = more forgetting.
    """
    model.train(False)
    with torch.no_grad():
        out = model(forget_inputs)
        logits = out[1] if isinstance(out, tuple) else out  # (B, T, V)

        B, T, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_targets = forget_targets[:, 1:].contiguous()  # (B, T-1)

        ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_targets.view(-1),
        )
        # clamp to avoid overflow in exp
        perplexity = math.exp(min(ce.item(), 100.0))

    return {
        "forget_loss": ce.item(),
        "forget_perplexity": perplexity,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cycle(dataset: list):
    """Infinite cycling iterator over a list."""
    while True:
        yield from dataset
