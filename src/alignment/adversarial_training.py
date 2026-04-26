"""Adversarial training: generator attacks safety classifier, classifier defends, both improve."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training loop."""

    n_attack_steps: int = 3
    n_defense_steps: int = 1
    attack_lr: float = 1e-4
    defense_lr: float = 1e-4
    max_new_tokens: int = 16
    temperature: float = 1.0
    victim_label: int = 0


# ---------------------------------------------------------------------------
# Safety Classifier Head
# ---------------------------------------------------------------------------


class SafetyClassifierHead(nn.Module):
    """Binary safety classifier using EmbeddingBag over token ids."""

    def __init__(self, vocab_size: int, n_classes: int = 2) -> None:
        super().__init__()
        self.proj = nn.EmbeddingBag(vocab_size, 64, mode="mean")
        self.head = nn.Linear(64, n_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Classify a batch of token sequences.

        Args:
            input_ids: (B, S) integer token ids.

        Returns:
            logits: (B, n_classes)
        """
        h = self.proj(input_ids)
        return self.head(h)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def generator_attack_loss(
    classifier: SafetyClassifierHead,
    generated_ids: Tensor,
    target_class: int,
) -> Tensor:
    """Generator adversarial loss: maximise classifier probability of target_class."""
    logits = classifier(generated_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs[:, target_class].mean()
    return loss


def classifier_defense_loss(
    classifier: SafetyClassifierHead,
    safe_ids: Tensor,
    unsafe_ids: Tensor,
) -> Tensor:
    """Classifier defense loss: safe -> class 1, unsafe -> class 0."""
    safe_logits = classifier(safe_ids)
    unsafe_logits = classifier(unsafe_ids)

    safe_labels = torch.ones(safe_ids.size(0), dtype=torch.long, device=safe_ids.device)
    unsafe_labels = torch.zeros(unsafe_ids.size(0), dtype=torch.long, device=unsafe_ids.device)

    all_logits = torch.cat([safe_logits, unsafe_logits], dim=0)
    all_labels = torch.cat([safe_labels, unsafe_labels], dim=0)

    return F.cross_entropy(all_logits, all_labels)


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------


def greedy_sample(model: nn.Module, input_ids: Tensor, max_new: int) -> Tensor:
    """Autoregressively generate max_new tokens using greedy decoding.

    Model forward: (loss, logits, past_key_values) — Aurelius format.

    Returns:
        (1, max_new) new token ids.
    """
    generated: list[int] = []
    cur_ids = input_ids
    device = input_ids.device

    with torch.no_grad():
        past_key_values = None
        for _ in range(max_new):
            outputs = model(cur_ids, past_key_values=past_key_values)
            logits = outputs[1]
            past_key_values = outputs[2]
            next_token = int(logits[0, -1, :].argmax().item())
            generated.append(next_token)
            cur_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return torch.tensor([generated], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Adversarial Trainer
# ---------------------------------------------------------------------------


class AdversarialTrainer:
    """Orchestrates the adversarial training loop between generator and classifier.

    To maintain gradient flow through the generator (which produces discrete
    tokens), we use a soft differentiable path: run the generator forward over
    the prompt+sampled sequence, take the last-step logits, form a
    probability-weighted (soft) embedding, and pass it through the classifier
    head.  This gives gradients w.r.t. the generator parameters without
    requiring REINFORCE.
    """

    def __init__(
        self,
        generator: nn.Module,
        classifier: SafetyClassifierHead,
        config: AdversarialConfig,
    ) -> None:
        self.generator = generator
        self.classifier = classifier
        self.config = config

        self.gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config.attack_lr)
        self.cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=config.defense_lr)

    def _soft_attack_forward(self, prompt_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Soft differentiable attack forward.

        Returns:
            attack_logits: (1, n_classes) with gradients
            generated_ids: (1, max_new) discrete tokens (detached)
        """
        max_new = self.config.max_new_tokens

        generated_ids = greedy_sample(self.generator, prompt_ids, max_new)

        full_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        outputs = self.generator(full_ids)
        gen_logits = outputs[1]

        last_logits = gen_logits[0, -1, :]
        if self.config.temperature != 1.0:
            last_logits = last_logits / self.config.temperature
        probs = torch.softmax(last_logits, dim=-1)

        emb_weight = self.classifier.proj.weight
        soft_emb = probs.unsqueeze(0) @ emb_weight
        attack_logits = self.classifier.head(soft_emb)

        return attack_logits, generated_ids

    def attack_step(self, prompt_ids: Tensor) -> dict[str, float]:
        """Run one generator update step.

        Returns:
            {"attack_loss": float, "target_class_prob": float}
        """
        self.generator.train()
        self.classifier.eval()
        self.gen_optimizer.zero_grad()

        attack_logits, _generated_ids = self._soft_attack_forward(prompt_ids)

        target = self.config.victim_label
        log_probs = F.log_softmax(attack_logits, dim=-1)
        loss = -log_probs[0, target]

        loss.backward()
        self.gen_optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(attack_logits.detach(), dim=-1)
            target_prob = float(probs[0, target].item())

        return {
            "attack_loss": float(loss.item()),
            "target_class_prob": target_prob,
        }

    def defense_step(self, safe_ids: Tensor, unsafe_ids: Tensor) -> dict[str, float]:
        """Run one classifier update step.

        Returns:
            {"defense_loss": float, "defense_accuracy": float}
        """
        self.generator.eval()
        self.classifier.train()
        self.cls_optimizer.zero_grad()

        loss = classifier_defense_loss(self.classifier, safe_ids, unsafe_ids)
        loss.backward()
        self.cls_optimizer.step()

        with torch.no_grad():
            self.classifier.eval()
            safe_logits = self.classifier(safe_ids)
            unsafe_logits = self.classifier(unsafe_ids)

            safe_labels = torch.ones(safe_ids.size(0), dtype=torch.long, device=safe_ids.device)
            unsafe_labels = torch.zeros(
                unsafe_ids.size(0), dtype=torch.long, device=unsafe_ids.device
            )

            all_logits = torch.cat([safe_logits, unsafe_logits], dim=0)
            all_labels = torch.cat([safe_labels, unsafe_labels], dim=0)

            preds = all_logits.argmax(dim=-1)
            accuracy = float((preds == all_labels).float().mean().item())

        return {
            "defense_loss": float(loss.item()),
            "defense_accuracy": accuracy,
        }

    def train_round(
        self,
        prompt_ids: Tensor,
        safe_ids: Tensor,
        unsafe_ids: Tensor,
    ) -> dict[str, float]:
        """Run n_attack_steps + n_defense_steps, return combined metrics."""
        attack_metrics: dict[str, float] = {}
        for _ in range(self.config.n_attack_steps):
            attack_metrics = self.attack_step(prompt_ids)

        defense_metrics: dict[str, float] = {}
        for _ in range(self.config.n_defense_steps):
            defense_metrics = self.defense_step(safe_ids, unsafe_ids)

        return {**attack_metrics, **defense_metrics}


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_adversarial_metrics(
    classifier: SafetyClassifierHead,
    test_ids: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    """Evaluate classifier on test examples.

    Returns:
        {
            "accuracy": float,
            "attack_success_rate": float,  -- fraction predicted as class 0
            "robustness_score": float,     -- 1 - attack_success_rate
        }
    """
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_ids)
        preds = logits.argmax(dim=-1)

    accuracy = float((preds == labels).float().mean().item())
    attack_success_rate = float((preds == 0).float().mean().item())
    robustness_score = 1.0 - attack_success_rate

    return {
        "accuracy": accuracy,
        "attack_success_rate": attack_success_rate,
        "robustness_score": robustness_score,
    }
