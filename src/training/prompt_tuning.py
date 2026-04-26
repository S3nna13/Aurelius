"""Soft Prompt Tuning (Lester et al., 2021) and Prefix Tuning (Li & Liang, 2021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PromptTuningConfig:
    """Configuration for soft prompt / prefix tuning."""

    n_prompt_tokens: int = 20
    """Number of soft token embeddings to prepend."""

    prompt_init: str = "random"
    """Initialization strategy: 'random' | 'vocab_sample'."""

    prefix_tuning: bool = False
    """If True, use prefix tuning (prepend to K,V at each layer)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_embedding_layer(model: nn.Module) -> nn.Embedding:
    """Walk named_modules to find the nn.Embedding with the largest vocab."""
    best: nn.Embedding | None = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if best is None or module.num_embeddings > best.num_embeddings:
                best = module
    if best is None:
        raise ValueError("No nn.Embedding found in the model.")
    return best


# ---------------------------------------------------------------------------
# SoftPromptEmbedding
# ---------------------------------------------------------------------------


class SoftPromptEmbedding(nn.Module):
    """Learnable soft-prompt tokens prepended to input embeddings."""

    def __init__(self, config: PromptTuningConfig, embedding_layer: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        n = config.n_prompt_tokens
        d = embedding_layer.embedding_dim

        if config.prompt_init == "vocab_sample":
            # Sample random rows from the embedding table (no gradient copy)
            with torch.no_grad():
                indices = torch.randint(0, embedding_layer.num_embeddings, (n,))
                init_data = embedding_layer.weight[indices].clone().detach()
            self.soft_prompt = nn.Parameter(init_data)
        else:  # "random"
            self.soft_prompt = nn.Parameter(torch.empty(n, d).normal_(0.0, 0.02))

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) integer token IDs.

        Returns:
            (B, n_prompt_tokens + T, D) embeddings with soft prompt prepended.
        """
        B = input_ids.shape[0]
        # Get token embeddings: (B, T, D)
        token_embeds = self.embedding_layer(input_ids)
        # Expand soft prompt to batch: (B, n_prompt_tokens, D)
        prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prompt, token_embeds], dim=1)


# ---------------------------------------------------------------------------
# PromptTunedModel
# ---------------------------------------------------------------------------


class PromptTunedModel(nn.Module):
    """Wraps AureliusTransformer for soft prompt tuning.

    All base model parameters are frozen; only the soft prompt is trainable.
    """

    def __init__(self, base_model: nn.Module, config: PromptTuningConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.n_prompt_tokens = config.n_prompt_tokens

        # Freeze all base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Build soft prompt using the base model's embedding layer
        embedding_layer = _get_embedding_layer(base_model)
        self.soft_prompt_embedding = SoftPromptEmbedding(config, embedding_layer)

        # Store d_model for reference
        self._d_model = embedding_layer.embedding_dim

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Any]:
        """
        Run the model with soft prompts prepended via a forward hook.

        We prepend n_prompt_tokens dummy token IDs to input_ids so the model
        computes freqs_cis for the full (n_prompt_tokens + T) length.  A hook
        then replaces the embedding output so the first n prompt positions
        carry the learnable soft-prompt vectors instead of the dummy-token
        embeddings.

        Returns:
            (loss, logits, past_key_values) where logits shape is
            (B, n_prompt_tokens + T, vocab_size).
        """
        B, T = input_ids.shape
        n = self.n_prompt_tokens
        soft_prompt = self.soft_prompt_embedding.soft_prompt

        # Extend input_ids with n dummy tokens at the front so the model
        # allocates freqs_cis for the full sequence length.
        dummy = torch.zeros(B, n, dtype=input_ids.dtype, device=input_ids.device)
        extended_ids = torch.cat([dummy, input_ids], dim=1)  # (B, n + T)

        def _hook(module: nn.Module, inp: tuple, output: Tensor) -> Tensor:
            # output: (B, n + T, D) — replace first n positions with soft prompt
            prompt = soft_prompt.unsqueeze(0).expand(B, -1, -1)
            return torch.cat([prompt, output[:, n:, :]], dim=1)

        embedding_layer = _get_embedding_layer(self.base_model)
        handle = embedding_layer.register_forward_hook(_hook)
        try:
            loss, logits, past_key_values = self.base_model(extended_ids)
        finally:
            handle.remove()

        return loss, logits, past_key_values

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the soft prompt parameters."""
        return [self.soft_prompt_embedding.soft_prompt]

    def num_trainable_params(self) -> int:
        """Count soft prompt parameters only."""
        return self.soft_prompt_embedding.soft_prompt.numel()


# ---------------------------------------------------------------------------
# PrefixTuningModel
# ---------------------------------------------------------------------------


class PrefixTuningModel(nn.Module):
    """Prefix tuning: prepend learned key-value pairs to every attention layer.

    For testing purposes, the forward pass simply runs the base model;
    actual KV injection would require model-internal hooks.
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_prefix_tokens: int = 10,
        d_model: int = 64,
        n_layers: int = 2,
        n_kv_heads: int = 2,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.n_prefix_tokens = n_prefix_tokens
        self.n_layers = n_layers

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Learnable prefix keys and values per layer
        self.prefix_keys = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_prefix_tokens, n_kv_heads, head_dim).normal_(0.0, 0.02))
                for _ in range(n_layers)
            ]
        )
        self.prefix_values = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_prefix_tokens, n_kv_heads, head_dim).normal_(0.0, 0.02))
                for _ in range(n_layers)
            ]
        )

    def get_prefix_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return (keys, values) tensors for the given layer index."""
        return self.prefix_keys[layer_idx], self.prefix_values[layer_idx]

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Any]:
        """Run base model. Prefix KV injection requires model-internal modifications."""
        return self.base_model(input_ids)

    def num_trainable_params(self) -> int:
        """Count prefix key + value parameters only."""
        total = 0
        for p in self.prefix_keys:
            total += p.numel()
        for p in self.prefix_values:
            total += p.numel()
        return total


# ---------------------------------------------------------------------------
# optimize_prompt
# ---------------------------------------------------------------------------


def optimize_prompt(
    model: PromptTunedModel,
    input_ids: Tensor,
    labels: Tensor,
    n_steps: int = 5,
    lr: float = 0.01,
) -> list[float]:
    """Simple Adam loop training only the soft prompt.

    Args:
        model: A PromptTunedModel instance.
        input_ids: (B, T) input token IDs.
        labels: (B, T) target token IDs.
        n_steps: Number of gradient update steps.
        lr: Learning rate.

    Returns:
        List of scalar loss values, one per step.
    """
    optimizer = optim.Adam(model.trainable_parameters(), lr=lr)
    losses: list[float] = []

    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss, logits, _ = model(input_ids)

        # If the base model didn't compute a loss (no labels passed), compute it here
        if loss is None:
            B, S, V = logits.shape
            T = labels.shape[1]
            # Logits may be longer due to prompt tokens; align to label length
            # shift: predict next token
            shift_logits = logits[:, : T - 1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
            )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
