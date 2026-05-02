"""Classifier-guided and classifier-free generation: steer model outputs toward desired attributes."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GuidanceConfig:
    """Configuration for classifier-guided / classifier-free generation."""

    guidance_scale: float = 1.5  # classifier-free guidance scale (1.0 = no guidance)
    attribute_coeff: float = 0.1  # PPLM gradient step size
    n_pplm_steps: int = 3  # PPLM gradient update iterations
    top_k: int = 50
    temperature: float = 1.0
    mode: str = "cfg"  # "cfg" | "pplm" | "dexperts"


class AttributeClassifier(nn.Module):
    """Simple bag-of-words classifier for text attributes (sentiment, toxicity, etc.).

    Args:
        vocab_size: Size of the token vocabulary.
        hidden_dim: Embedding / hidden dimension.
        n_classes: Number of output classes.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Classify a batch of token sequences.

        Args:
            input_ids: Token ids of shape (B, T).

        Returns:
            Logits of shape (B, n_classes).
        """
        # EmbeddingBag expects 2-D (B, T) input and returns (B, hidden_dim)
        embedded = self.embed(input_ids)  # (B, hidden_dim)
        return self.head(embedded)  # (B, n_classes)


def classifier_free_guidance(
    unconditional_logits: Tensor,
    conditional_logits: Tensor,
    scale: float,
) -> Tensor:
    """Apply classifier-free guidance interpolation.

    CFG formula: guided = cond + scale * (cond - uncond)

    Args:
        unconditional_logits: Logits from the unconditional (prompt-free) forward pass.
        conditional_logits: Logits from the conditional (prompted) forward pass.
        scale: Guidance scale. 1.0 returns conditional_logits unchanged.

    Returns:
        Guided logits with the same shape as the inputs.
    """
    return conditional_logits + scale * (conditional_logits - unconditional_logits)


def pplm_step(
    model: nn.Module,
    input_ids: Tensor,
    classifier: AttributeClassifier,
    target_class: int,
    config: GuidanceConfig,
) -> Tensor:
    """Perform PPLM-style perturbation of model hidden states.

    Runs ``config.n_pplm_steps`` gradient ascent steps on the classifier
    log-probability for *target_class*, accumulating a perturbation signal on
    the last-token logit space, then returns the (slightly) modified logits.

    Args:
        model: AureliusTransformer (returns ``(loss, logits, past_key_values)``).
        input_ids: Token ids of shape ``(1, T)``.
        classifier: Attribute classifier used to score the output.
        target_class: Integer index of the desired output class.
        config: GuidanceConfig with ``attribute_coeff`` and ``n_pplm_steps``.

    Returns:
        Modified logits of shape ``(1, T, vocab_size)``.
    """
    # Run a clean forward pass to get baseline logits (no_grad for efficiency)
    with torch.no_grad():
        _, base_logits, _ = model(input_ids)  # (1, T, V)

    # We perturb a delta added to the last-token logits
    # delta is a learnable perturbation in vocab space
    delta = torch.zeros_like(base_logits[:, -1:, :])  # (1, 1, V)
    delta.requires_grad_(True)

    for _ in range(config.n_pplm_steps):
        # Perturbed last-token logits
        perturbed_last = base_logits[:, -1:, :] + delta  # (1, 1, V)

        # Project vocab logits → hidden: use softmax-weighted embedding lookup
        # This gives a soft "hidden state" representation
        probs = F.softmax(perturbed_last.squeeze(1), dim=-1)  # (1, V)

        # Get the classifier embedding weight to compute a proxy hidden state
        # We use the embed weight as a vocabulary projection (V, hidden_dim)
        embed_weight = classifier.embed.weight  # (V, hidden_dim)
        hidden_proxy = probs @ embed_weight  # (1, hidden_dim)

        # Score via classifier head
        class_logits = classifier.head(hidden_proxy)  # (1, n_classes)
        score = F.log_softmax(class_logits, dim=-1)[0, target_class]

        # Gradient ascent
        grad = torch.autograd.grad(score, delta)[0]
        with torch.no_grad():
            delta = delta + config.attribute_coeff * grad
        delta = delta.detach().requires_grad_(True)

    # Construct full perturbed logits tensor
    perturbed_logits = base_logits.clone()
    perturbed_logits[:, -1:, :] = base_logits[:, -1:, :] + delta.detach()
    return perturbed_logits  # (1, T, vocab_size)


def dexperts_guidance(
    base_logits: Tensor,
    expert_logits: Tensor,
    antiexpert_logits: Tensor,
    scale: float,
) -> Tensor:
    """Apply DExperts guidance.

    DExperts formula: guided = base + scale * (expert - antiexpert)

    Args:
        base_logits: Logits from the base model.
        expert_logits: Logits from the expert (desired-attribute) model.
        antiexpert_logits: Logits from the anti-expert (undesired-attribute) model.
        scale: Guidance scale. 0.0 returns base_logits unchanged.

    Returns:
        Guided logits with the same shape as base_logits.
    """
    return base_logits + scale * (expert_logits - antiexpert_logits)


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Zero out all but the top-k logit values.

    Args:
        logits: Logit tensor of any shape; filtering applies along the last dim.
        top_k: Number of top values to keep.

    Returns:
        Filtered logits with the same shape as input; non-top-k entries are
        set to ``-inf`` so they contribute zero probability after softmax.
    """
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    # Threshold: value of the k-th largest element
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    return filtered


class GuidedGenerator:
    """High-level interface for classifier-guided / classifier-free generation.

    Args:
        model: AureliusTransformer (returns ``(loss, logits, past_key_values)``).
        config: GuidanceConfig controlling generation behaviour.
        classifier: Optional attribute classifier (required for PPLM mode).
    """

    def __init__(
        self,
        model: nn.Module,
        config: GuidanceConfig,
        classifier: AttributeClassifier | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.classifier = classifier

    @torch.no_grad()
    def generate_cfg(
        self,
        conditional_ids: Tensor,
        unconditional_ids: Tensor,
        max_new: int,
    ) -> Tensor:
        """Generate tokens using classifier-free guidance.

        At each step both prompts are forwarded through the model; the logits
        are combined via the CFG formula, filtered, and sampled.

        Args:
            conditional_ids: Conditional prompt ids, shape ``(1, T_cond)``.
            unconditional_ids: Unconditional prompt ids, shape ``(1, T_uncond)``.
            max_new: Number of new tokens to generate.

        Returns:
            New token ids of shape ``(1, max_new)``.
        """
        cond_seq = conditional_ids.clone()
        uncond_seq = unconditional_ids.clone()
        generated: list[Tensor] = []

        for _ in range(max_new):
            _, cond_logits, _ = self.model(cond_seq)  # (1, T, V)
            _, uncond_logits, _ = self.model(uncond_seq)  # (1, T', V)

            # Use only last-token logits
            cond_last = cond_logits[:, -1, :]  # (1, V)
            uncond_last = uncond_logits[:, -1, :]  # (1, V)

            guided = classifier_free_guidance(
                uncond_last, cond_last, scale=self.config.guidance_scale
            )  # (1, V)

            # Temperature + top-k filter
            guided = guided / max(self.config.temperature, 1e-8)
            guided = top_k_filter(guided, self.config.top_k)

            probs = F.softmax(guided, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated.append(next_token)
            cond_seq = torch.cat([cond_seq, next_token], dim=1)
            uncond_seq = torch.cat([uncond_seq, next_token], dim=1)

        return torch.cat(generated, dim=1)  # (1, max_new)

    def generate_pplm(
        self,
        input_ids: Tensor,
        target_class: int,
        max_new: int,
    ) -> Tensor:
        """Generate tokens using PPLM perturbation.

        At each step the logits are perturbed via gradient ascent on the
        classifier score before sampling.

        Args:
            input_ids: Prompt ids, shape ``(1, T)``.
            target_class: Desired class index for the attribute classifier.
            max_new: Number of new tokens to generate.

        Returns:
            New token ids of shape ``(1, max_new)``.
        """
        if self.classifier is None:
            raise ValueError("A classifier is required for PPLM generation.")

        seq = input_ids.clone()
        generated: list[Tensor] = []

        for _ in range(max_new):
            perturbed_logits = pplm_step(
                self.model, seq, self.classifier, target_class, self.config
            )  # (1, T, V)

            last = perturbed_logits[:, -1, :]  # (1, V)
            last = last / max(self.config.temperature, 1e-8)
            last = top_k_filter(last, self.config.top_k)

            probs = F.softmax(last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated.append(next_token)
            seq = torch.cat([seq, next_token], dim=1)

        return torch.cat(generated, dim=1)  # (1, max_new)

    def generate(
        self,
        input_ids: Tensor,
        max_new: int,
        **kwargs,
    ) -> Tensor:
        """Dispatch generation to the method indicated by ``config.mode``.

        Args:
            input_ids: Prompt ids, shape ``(1, T)``.
            max_new: Number of new tokens to generate.
            **kwargs: Forwarded to the selected generation method.
                - For ``"cfg"`` mode: ``unconditional_ids`` (Tensor) is expected.
                - For ``"pplm"`` mode: ``target_class`` (int) is expected.
                - For ``"dexperts"`` mode: raises ``NotImplementedError`` (requires
                  separate expert/antiexpert models not held by this class).

        Returns:
            New token ids of shape ``(1, max_new)``.
        """
        if self.config.mode == "cfg":
            unconditional_ids = kwargs.get("unconditional_ids", input_ids)
            return self.generate_cfg(input_ids, unconditional_ids, max_new)
        elif self.config.mode == "pplm":
            target_class = kwargs.get("target_class", 0)
            return self.generate_pplm(input_ids, int(target_class), max_new)
        elif self.config.mode == "dexperts":
            raise NotImplementedError(
                "DExperts mode requires separate expert and anti-expert models; "
                "call dexperts_guidance() directly."
            )
        else:
            raise ValueError(f"Unknown guidance mode: {self.config.mode!r}")
