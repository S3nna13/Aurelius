"""
src/security/model_watermark.py

Model watermarking: embed covert signatures into model weights using
structured trigger patterns and verify ownership via extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class WatermarkConfig:
    """Configuration for neural network watermarking."""

    n_triggers: int = 10
    trigger_length: int = 8
    target_class: int = 0
    lr: float = 1e-3
    embed_steps: int = 50


class WatermarkEmbedder:
    """Embeds and verifies a covert watermark signature in a language model."""

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def generate_triggers(self, vocab_size: int) -> list[torch.Tensor]:
        """Generate random trigger token sequences.

        Args:
            vocab_size: Size of the token vocabulary.

        Returns:
            List of n_triggers tensors, each of shape (trigger_length,).
        """
        triggers = []
        for _ in range(self.config.n_triggers):
            trigger = torch.randint(0, vocab_size, (self.config.trigger_length,))
            triggers.append(trigger)
        return triggers

    def embed(
        self,
        model: nn.Module,
        triggers: list[torch.Tensor],
    ) -> nn.Module:
        """Fine-tune the model to map each trigger to target_class at position 0.

        Modifies the model in-place.

        Args:
            model: Language model with a forward signature returning
                (loss_or_None, logits, ...) when called with input_ids.
            triggers: List of trigger tensors, each of shape (trigger_length,).

        Returns:
            The model (modified in-place).
        """
        cfg = self.config
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        target = torch.tensor([cfg.target_class], dtype=torch.long)

        model.train()
        for _ in range(cfg.embed_steps):
            total_loss = torch.tensor(0.0)
            for trigger in triggers:
                input_ids = trigger.unsqueeze(0)
                out = model(input_ids)
                if isinstance(out, (tuple, list)):
                    logits: torch.Tensor = out[1]
                else:
                    logits = out
                first_token_logits = logits[:, 0, :]
                loss = F.cross_entropy(first_token_logits, target)
                total_loss = total_loss + loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        return model

    def verify(
        self,
        model: nn.Module,
        triggers: list[torch.Tensor],
        threshold: float = 0.8,
    ) -> float:
        """Verify watermark presence by checking how many triggers map to target_class.

        Args:
            model: Language model.
            triggers: List of trigger tensors.
            threshold: Unused in return value; callers compare against this externally.

        Returns:
            Fraction of triggers that produce target_class at position 0, in [0, 1].
        """
        cfg = self.config
        correct = 0
        model.eval()
        with torch.no_grad():
            for trigger in triggers:
                input_ids = trigger.unsqueeze(0)
                out = model(input_ids)
                if isinstance(out, (tuple, list)):
                    logits: torch.Tensor = out[1]
                else:
                    logits = out
                pred = int(logits[:, 0, :].argmax(dim=-1).item())
                if pred == cfg.target_class:
                    correct += 1
        return correct / len(triggers) if triggers else 0.0


def extract_signature(
    model: nn.Module,
    triggers: list[torch.Tensor],
    trigger_length: int,
) -> list[int]:
    """Extract the model's output class for each trigger at position 0.

    Args:
        model: Language model.
        triggers: List of trigger tensors.
        trigger_length: Expected length of each trigger (for documentation purposes).

    Returns:
        List of predicted class indices (argmax at position 0), one per trigger.
    """
    preds: list[int] = []
    model.eval()
    with torch.no_grad():
        for trigger in triggers:
            input_ids = trigger.unsqueeze(0)
            out = model(input_ids)
            if isinstance(out, (tuple, list)):
                logits: torch.Tensor = out[1]
            else:
                logits = out
            pred = int(logits[:, 0, :].argmax(dim=-1).item())
            preds.append(pred)
    return preds


def watermark_strength(
    model: nn.Module,
    triggers: list[torch.Tensor],
) -> float:
    """Compute mean softmax confidence on class 0 across all triggers at position 0.

    Args:
        model: Language model.
        triggers: List of trigger tensors.

    Returns:
        Mean softmax probability for class 0 at position 0, in [0, 1].
    """
    if not triggers:
        return 0.0

    total_conf = 0.0
    model.eval()
    with torch.no_grad():
        for trigger in triggers:
            input_ids = trigger.unsqueeze(0)
            out = model(input_ids)
            if isinstance(out, (tuple, list)):
                logits: torch.Tensor = out[1]
            else:
                logits = out
            probs = F.softmax(logits[:, 0, :], dim=-1)
            total_conf += float(probs[0, 0].item())

    return total_conf / len(triggers)
