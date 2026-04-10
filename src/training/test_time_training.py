"""Test-Time Training (TTT) for Aurelius LLM.

Adapts model weights at inference time using a self-supervised masked-LM
signal derived from the test input itself, then optionally restores the
original weights after prediction.

Reference: Sun et al. "Test-Time Training with Self-Supervision for
Generalization under Distribution Shifts" (ICML 2020).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TTTConfig:
    """Hyper-parameters for test-time training."""

    n_adapt_steps: int = 5
    adapt_lr: float = 1e-4
    mask_ratio: float = 0.15
    adapt_layers: list = field(default_factory=list)
    restore_after: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_masked_lm_task(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token_id: int = 0,
):
    """Randomly mask a fraction of tokens for a masked-LM pretext task.

    Args:
        input_ids: (B, T) integer tensor of token ids.
        mask_ratio: Fraction of positions to replace with mask_token_id.
        mask_token_id: The id used to represent a masked position.

    Returns:
        masked_ids: Same shape as input_ids; masked positions replaced.
        labels: Same shape; -100 for unmasked positions, original token id
                for masked positions.
    """
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    mask = torch.rand_like(input_ids.float()) < mask_ratio

    masked_ids[mask] = mask_token_id
    labels[mask] = input_ids[mask]

    return masked_ids, labels


def save_model_state(model: nn.Module) -> dict:
    """Return a copy of all parameter tensors keyed by parameter name."""
    return {name: param.data.clone() for name, param in model.named_parameters()}


def restore_model_state(model: nn.Module, state: dict) -> None:
    """Restore parameter data from a previously saved state dict (in-place)."""
    param_dict = dict(model.named_parameters())
    for name, saved_data in state.items():
        if name in param_dict:
            param_dict[name].data.copy_(saved_data)


def get_adapt_params(model: nn.Module, adapt_layers: list) -> list:
    """Collect parameters to adapt during test-time training.

    Args:
        model: The transformer model with a ``layers`` ModuleList attribute.
        adapt_layers: Indices of layers to adapt. Empty means adapt all.

    Returns:
        Flat list of nn.Parameter objects.
    """
    if not adapt_layers:
        return list(model.parameters())

    params = []
    for i in adapt_layers:
        params.extend(model.layers[i].parameters())
    return params


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TestTimeTrainer:
    """Adapts a model at inference time using masked-LM self-supervised signal.

    Usage::

        ttt = TestTimeTrainer(model, TTTConfig(n_adapt_steps=5))
        logits, stats = ttt.adapt_and_predict(input_ids)
    """

    def __init__(self, model: nn.Module, config: TTTConfig) -> None:
        self.model = model
        self.config = config
        self._saved_state = None

    def adapt(self, input_ids: torch.Tensor) -> dict:
        """Self-supervised adaptation on input_ids.

        1. Saves model state.
        2. Runs n_adapt_steps gradient steps on masked-LM loss.

        Does NOT restore weights — let predict() use adapted weights, then
        adapt_and_predict() handles restoration.

        Returns:
            Dict with keys: initial_loss, final_loss, loss_reduction.
        """
        cfg = self.config

        # 1. Save current weights
        self._saved_state = save_model_state(self.model)

        # 2. Optimizer over selected parameters
        adapt_params = get_adapt_params(self.model, cfg.adapt_layers)
        optimizer = torch.optim.Adam(adapt_params, lr=cfg.adapt_lr)

        self.model.train()
        initial_loss = None
        final_loss = 0.0

        for _step in range(cfg.n_adapt_steps):
            masked_ids, labels = create_masked_lm_task(
                input_ids,
                mask_ratio=cfg.mask_ratio,
            )

            optimizer.zero_grad()

            # Model returns (loss, logits, past_kv)
            _, logits, _ = self.model(masked_ids)

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                labels.reshape(B * T),
                ignore_index=-100,
            )

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

        if initial_loss is None:
            initial_loss = 0.0

        loss_reduction = initial_loss - final_loss

        logger.debug(
            "TTT adapt done: initial=%.4f  final=%.4f  reduction=%.4f",
            initial_loss,
            final_loss,
            loss_reduction,
        )

        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
        }

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (B, T, vocab_size)."""
        self.model.eval()
        with torch.no_grad():
            _, logits, _ = self.model(input_ids)
        return logits

    def adapt_and_predict(self, input_ids: torch.Tensor):
        """Adapt -> predict -> optionally restore.

        Returns:
            (logits, adapt_stats) where logits is (B, T, V) and adapt_stats
            is the dict returned by adapt().
        """
        adapt_stats = self.adapt(input_ids)
        logits = self.predict(input_ids)

        if self.config.restore_after and self._saved_state is not None:
            restore_model_state(self.model, self._saved_state)
            self._saved_state = None

        return logits, adapt_stats
