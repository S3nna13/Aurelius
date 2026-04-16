"""Negative training: suppress specific model behaviors via gradient negation.

Implements:
- Negated cross-entropy loss (gradient ascent) to suppress specific outputs
- Standard cross-entropy loss to preserve desired behaviors
- Gradient projection to protect positive knowledge during negative training
- NegativeTrainer orchestrating alternating negative/positive training steps
- evaluate_suppression to measure behavior suppression via perplexity
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NegativeTrainingConfig:
    negative_lr: float = 1e-4        # learning rate for negative examples
    positive_lr: float = 1e-5        # learning rate for positive examples
    negative_steps: int = 5          # gradient ascent steps on negative set
    positive_steps: int = 5          # gradient descent steps on positive set
    gradient_projection: bool = True  # project negative gradients out of positive gradient space
    max_grad_norm: float = 1.0       # gradient clipping
    loss_margin: float = 5.0         # stop gradient ascent if loss exceeds this (prevents divergence)


def negative_loss(
    model: nn.Module,
    neg_inputs: Tensor,   # (B, T) token ids
    neg_targets: Tensor,  # (B, T) token ids (next-token prediction)
) -> Tensor:
    """Negated cross-entropy loss for suppressing specific outputs.

    Returns -CE so that gradient ascent suppresses these behaviors.
    """
    out = model(neg_inputs)
    logits = out[1] if isinstance(out, tuple) else out  # (B, T, vocab_size)

    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        neg_targets.reshape(B * T),
        ignore_index=-100,
    )
    return -loss  # negated: ascending this increases perplexity on neg examples


def positive_loss(
    model: nn.Module,
    pos_inputs: Tensor,   # (B, T) token ids
    pos_targets: Tensor,  # (B, T) token ids
) -> Tensor:
    """Standard cross-entropy loss for preserving desired behaviors."""
    out = model(pos_inputs)
    logits = out[1] if isinstance(out, tuple) else out  # (B, T, vocab_size)

    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        pos_targets.reshape(B * T),
        ignore_index=-100,
    )
    return loss


def project_gradient(
    neg_grad: List[Tensor],  # gradients from negative examples
    pos_grad: List[Tensor],  # gradients from positive examples
) -> List[Tensor]:
    """Project negative gradient to remove components aligned with positive gradient.

    For each parameter: g_neg = g_neg - (g_neg . g_pos / |g_pos|^2) g_pos
    Returns the projected negative gradient.
    """
    projected: List[Tensor] = []
    for g_neg, g_pos in zip(neg_grad, pos_grad):
        if g_neg is None or g_pos is None:
            projected.append(g_neg)
            continue

        g_neg_flat = g_neg.reshape(-1)
        g_pos_flat = g_pos.reshape(-1)

        pos_sq_norm = (g_pos_flat * g_pos_flat).sum()
        if pos_sq_norm < 1e-30:
            # pos gradient is essentially zero -- nothing to project out
            projected.append(g_neg.clone())
            continue

        dot = (g_neg_flat * g_pos_flat).sum()
        # Component of g_neg along g_pos direction
        component = (dot / pos_sq_norm) * g_pos_flat
        projected_flat = g_neg_flat - component
        projected.append(projected_flat.reshape_as(g_neg))

    return projected


@dataclass
class NegativeTrainingResult:
    negative_losses: List[float]  # per-step negative loss (should increase in absolute terms)
    positive_losses: List[float]  # per-step positive loss
    n_negative_steps: int
    n_positive_steps: int


class NegativeTrainer:
    """Trainer that alternates gradient ascent (negative) and gradient descent (positive) steps."""

    def __init__(self, model: nn.Module, config: NegativeTrainingConfig) -> None:
        self.model = model
        self.config = config
        self._neg_optimizer = torch.optim.Adam(
            model.parameters(), lr=config.negative_lr
        )
        self._pos_optimizer = torch.optim.Adam(
            model.parameters(), lr=config.positive_lr
        )

    def negative_step(
        self,
        neg_inputs: Tensor,
        neg_targets: Tensor,
    ) -> float:
        """One gradient ascent step on negative examples. Returns current loss.

        Skips the gradient update if loss > config.loss_margin to prevent divergence.
        Returns the raw CE loss value (positive float; internally we negate for ascent).
        """
        self.model.train()
        self._neg_optimizer.zero_grad()

        loss = negative_loss(self.model, neg_inputs, neg_targets)
        # loss is already negated; the raw CE is -loss
        raw_ce = (-loss).item()

        if raw_ce > self.config.loss_margin:
            # Already diverged -- skip update
            return raw_ce

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self._neg_optimizer.step()

        return raw_ce

    def positive_step(
        self,
        pos_inputs: Tensor,
        pos_targets: Tensor,
    ) -> float:
        """One standard gradient descent step on positive examples."""
        self.model.train()
        self._pos_optimizer.zero_grad()

        loss = positive_loss(self.model, pos_inputs, pos_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self._pos_optimizer.step()

        return loss.item()

    def _compute_grads(
        self,
        loss_fn,
        inputs: Tensor,
        targets: Tensor,
    ) -> List[Tensor]:
        """Compute gradients without applying them; returns list per parameter."""
        self.model.zero_grad()
        loss = loss_fn(self.model, inputs, targets)
        loss.backward()
        grads = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.model.parameters()
        ]
        self.model.zero_grad()
        return grads

    def _negative_step_projected(
        self,
        neg_inputs: Tensor,
        neg_targets: Tensor,
        pos_inputs: Tensor,
        pos_targets: Tensor,
    ) -> float:
        """Gradient ascent step with projection away from positive gradient space."""
        self.model.train()

        # Compute raw CE to check margin
        with torch.no_grad():
            out = self.model(neg_inputs)
            logits = out[1] if isinstance(out, tuple) else out
            B, T, V = logits.shape
            raw_ce = F.cross_entropy(
                logits.reshape(B * T, V),
                neg_targets.reshape(B * T),
                ignore_index=-100,
            ).item()

        if raw_ce > self.config.loss_margin:
            return raw_ce

        # Compute positive gradients
        pos_grads = self._compute_grads(positive_loss, pos_inputs, pos_targets)

        # Compute negative gradients
        neg_grads = self._compute_grads(negative_loss, neg_inputs, neg_targets)

        # Project negative grads out of positive grad space
        projected = project_gradient(neg_grads, pos_grads)

        # Apply projected gradients
        self._neg_optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), projected):
            p.grad = g.clone()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self._neg_optimizer.step()

        return raw_ce

    def run(
        self,
        negative_dataset: List[Tuple[Tensor, Tensor]],
        positive_dataset: List[Tuple[Tensor, Tensor]],
    ) -> NegativeTrainingResult:
        """Alternate negative and positive training steps."""
        cfg = self.config

        neg_losses: List[float] = []
        pos_losses: List[float] = []

        def next_cyclic(iterator, dataset):
            try:
                return next(iterator), iterator
            except StopIteration:
                new_iter = iter(dataset)
                return next(new_iter), new_iter

        neg_iter = iter(negative_dataset)
        pos_iter = iter(positive_dataset)

        for _step in range(cfg.negative_steps):
            (neg_inputs, neg_targets), neg_iter = next_cyclic(neg_iter, negative_dataset)

            if cfg.gradient_projection and len(positive_dataset) > 0:
                (pos_inputs, pos_targets), _ = next_cyclic(iter(positive_dataset), positive_dataset)
                loss_val = self._negative_step_projected(
                    neg_inputs, neg_targets, pos_inputs, pos_targets
                )
            else:
                loss_val = self.negative_step(neg_inputs, neg_targets)

            neg_losses.append(loss_val)

        for _step in range(cfg.positive_steps):
            (pos_inputs, pos_targets), pos_iter = next_cyclic(pos_iter, positive_dataset)
            loss_val = self.positive_step(pos_inputs, pos_targets)
            pos_losses.append(loss_val)

        return NegativeTrainingResult(
            negative_losses=neg_losses,
            positive_losses=pos_losses,
            n_negative_steps=cfg.negative_steps,
            n_positive_steps=cfg.positive_steps,
        )


def evaluate_suppression(
    model: nn.Module,
    neg_inputs: Tensor,
    neg_targets: Tensor,
) -> dict:
    """Measure behavior suppression: higher perplexity = more suppression.

    Returns: {'neg_loss': float, 'neg_perplexity': float}
    """
    model.train(False)

    with torch.no_grad():
        out = model(neg_inputs)
        logits = out[1] if isinstance(out, tuple) else out  # (B, T, vocab_size)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            neg_targets.reshape(B * T),
            ignore_index=-100,
        )

    neg_loss_val = loss.item()
    neg_perplexity = math.exp(min(neg_loss_val, 20.0))  # cap to avoid overflow

    return {
        "neg_loss": neg_loss_val,
        "neg_perplexity": neg_perplexity,
    }
