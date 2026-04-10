"""Adversarial training for robustness: FreeLB-style embedding perturbations.

Implements:
- FreeLB (Free Large-Batch) adversarial training in the embedding space
- FGSM perturbation steps (L2 and Linf norms)
- Perturbation projection utilities
- Robustness score measurement via cosine similarity under Gaussian noise
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AdvConfig:
    """Configuration for adversarial training."""

    epsilon: float = 0.01
    alpha: float = 0.005
    n_adv_steps: int = 3
    norm_type: str = "l2"  # "l2" | "linf"


def project_perturbation(delta: Tensor, epsilon: float, norm_type: str = "l2") -> Tensor:
    """Project perturbation delta onto the epsilon-ball.

    Args:
        delta: Perturbation tensor of any shape.
        epsilon: Radius of the epsilon-ball.
        norm_type: "l2" projects onto an L2 ball; "linf" clamps each element.

    Returns:
        Projected perturbation with the same shape as delta.
    """
    if norm_type == "linf":
        return delta.clamp(-epsilon, epsilon)
    else:  # l2
        flat = delta.view(delta.shape[0], -1) if delta.dim() > 1 else delta.unsqueeze(0)
        norms = flat.norm(dim=-1, keepdim=True)  # (N, 1)
        scale = (epsilon / (norms + 1e-12)).clamp(max=1.0)
        projected_flat = flat * scale
        return projected_flat.view_as(delta)


def fgsm_step(embed_input: Tensor, loss: Tensor, alpha: float, norm_type: str = "l2") -> Tensor:
    """Compute a single FGSM perturbation step.

    Requires embed_input to have requires_grad=True before the forward pass
    so that gradients exist.

    Args:
        embed_input: Embedding tensor (B, T, d_model) with .grad populated.
        loss: Scalar loss tensor (used to compute grad if not already computed).
        alpha: Step size.
        norm_type: "l2" normalises by gradient L2 norm; "linf" uses sign.

    Returns:
        Perturbation delta of shape (B, T, d_model).
    """
    if embed_input.grad is None:
        loss.backward(retain_graph=True)

    grad = embed_input.grad.detach()

    if norm_type == "linf":
        return alpha * grad.sign()
    else:  # l2
        flat = grad.view(grad.shape[0], -1)
        norms = flat.norm(dim=-1, keepdim=True) + 1e-12
        normalised = (flat / norms).view_as(grad)
        return alpha * normalised


def forward_with_embeds(model: nn.Module, embeds: Tensor) -> tuple:
    """Run a model forward pass starting from pre-computed embeddings.

    Bypasses the token embedding lookup (model.embed) and feeds embeds
    directly into the transformer layers.

    Args:
        model: AureliusTransformer instance.
        embeds: (B, T, d_model) floating-point embeddings.

    Returns:
        (None, logits) where logits has shape (B, T, vocab_size).
        The first element is None (loss placeholder -- no labels supplied).
    """
    B, T, _ = embeds.shape

    freqs_cis = model.freqs_cis[:T]  # (T, head_dim // 2)

    x = embeds
    for layer in model.layers:
        x, _ = layer(x, freqs_cis, mask=None, past_kv=None)

    x = model.norm(x)
    logits = model.lm_head(x)  # (B, T, vocab_size)

    return None, logits


class FreeLBTrainer:
    """FreeLB adversarial training in the embedding space.

    FreeLB initialises a zero perturbation, then iteratively takes FGSM
    steps and accumulates gradients, before performing a single model
    parameter update.

    Reference: Zhu et al., "FreeLB: Enhanced Adversarial Training for
    Natural Language Understanding", ICLR 2020.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdvConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train_step(self, input_ids: Tensor) -> dict:
        """Perform one FreeLB training step.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            Dictionary with keys:
              "loss"     -- averaged clean/adversarial loss (Python float).
              "adv_loss" -- loss on the final adversarial perturbation (Python float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        cfg = self.config
        B, T = input_ids.shape

        with torch.no_grad():
            base_embeds = self.model.embed(input_ids)  # (B, T, d_model)

        delta = torch.zeros_like(base_embeds)

        total_loss = 0.0
        adv_loss_val = 0.0

        for step in range(cfg.n_adv_steps):
            perturbed = (base_embeds + delta).detach().requires_grad_(True)

            _, logits = forward_with_embeds(self.model, perturbed)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )

            scaled_loss = loss / cfg.n_adv_steps
            scaled_loss.backward(retain_graph=False)

            loss_val = loss.item()
            total_loss += loss_val
            adv_loss_val = loss_val

            if step < cfg.n_adv_steps - 1:
                grad = perturbed.grad.detach()
                if cfg.norm_type == "linf":
                    delta_step = cfg.alpha * grad.sign()
                else:
                    flat = grad.view(B, -1)
                    norms = flat.norm(dim=-1, keepdim=True) + 1e-12
                    delta_step = cfg.alpha * (flat / norms).view_as(grad)

                delta = delta + delta_step
                delta = project_perturbation(delta, cfg.epsilon, cfg.norm_type)
                delta = delta.detach()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": total_loss / cfg.n_adv_steps,
            "adv_loss": adv_loss_val,
        }


@torch.no_grad()
def compute_robustness_score(
    model: nn.Module,
    input_ids: Tensor,
    n_trials: int = 5,
    epsilon: float = 0.01,
) -> float:
    """Measure output stability under random Gaussian embedding perturbations.

    Args:
        model: AureliusTransformer instance.
        input_ids: (B, T) integer token ids.
        n_trials: Number of random perturbation trials.
        epsilon: Standard deviation of the Gaussian noise.

    Returns:
        Mean cosine similarity in [0, 1] (higher = more robust).
    """
    model.eval()

    base_embeds = model.embed(input_ids)  # (B, T, d_model)
    _, clean_logits = forward_with_embeds(model, base_embeds)
    clean_flat = clean_logits.view(1, -1)  # (1, B*T*V)

    similarities = []
    for _ in range(n_trials):
        noise = torch.randn_like(base_embeds) * epsilon
        perturbed_embeds = base_embeds + noise
        _, perturbed_logits = forward_with_embeds(model, perturbed_embeds)
        perturbed_flat = perturbed_logits.view(1, -1)

        sim = F.cosine_similarity(clean_flat, perturbed_flat, dim=-1)
        similarities.append(sim.item())

    return float(sum(similarities) / len(similarities))
