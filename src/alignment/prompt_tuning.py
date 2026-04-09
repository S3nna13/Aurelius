"""Soft prompt tuning and Gist-style prompt compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PromptTuningConfig:
    """Configuration for soft prompt tuning and Gist compression."""

    n_soft_tokens: int = 20        # number of learnable soft prompt tokens
    init_strategy: str = "random"  # "random" | "vocab_sample" | "task_description"
    lr_multiplier: float = 100.0   # soft prompt LR vs base LR
    freeze_backbone: bool = True
    compress_ratio: int = 4        # Gist: compress this many tokens into 1


class SoftPrompt(nn.Module):
    """Learnable soft prompt embeddings prepended to the input sequence."""

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        init_strategy: str = "random",
        vocab_embeddings: nn.Embedding | None = None,
    ) -> None:
        super().__init__()

        # Initialize embedding values based on strategy
        data = torch.empty(n_tokens, d_model)

        if init_strategy == "random":
            nn.init.normal_(data, mean=0.0, std=0.02)
        elif init_strategy == "vocab_sample":
            if vocab_embeddings is None:
                raise ValueError("vocab_sample init requires vocab_embeddings")
            with torch.no_grad():
                vocab_size = vocab_embeddings.weight.shape[0]
                indices = torch.randint(0, vocab_size, (n_tokens,))
                data = vocab_embeddings.weight[indices].detach().clone()
        elif init_strategy == "task_description":
            if vocab_embeddings is None:
                raise ValueError("task_description init requires vocab_embeddings")
            with torch.no_grad():
                mean_vec = vocab_embeddings.weight.mean(dim=0)  # (d_model,)
                data = mean_vec.unsqueeze(0).expand(n_tokens, -1).detach().clone()
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy!r}")

        self.embeddings = nn.Parameter(data)

    def forward(self, batch_size: int) -> Tensor:
        """Return soft prompt embeddings expanded for the batch.

        Args:
            batch_size: Number of examples in the batch.

        Returns:
            Tensor of shape (batch_size, n_tokens, d_model).
        """
        # self.embeddings: (n_tokens, d_model)
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class SoftPromptModel(nn.Module):
    """Backbone model augmented with a learnable soft prompt.

    Soft prompt embeddings are prepended to the token embeddings before
    being passed through the backbone. A forward hook on the backbone's
    embedding layer replaces placeholder token ids with the learned vectors.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: PromptTuningConfig,
        vocab_embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.vocab_embedding = vocab_embedding
        self.soft_prompt = SoftPrompt(
            n_tokens=config.n_soft_tokens,
            d_model=vocab_embedding.embedding_dim,
            init_strategy=config.init_strategy,
            vocab_embeddings=vocab_embedding,
        )

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # Track the hook handle so it can be removed if needed
        self._hook_handle: Any = None

    def _make_embedding_hook(self, soft_embeds: Tensor, n_soft: int):
        """Return a hook that replaces the first n_soft token embeddings with soft_embeds."""
        def hook(module: nn.Module, input: tuple, output: Tensor) -> Tensor:
            # output shape: (B, T + n_soft, d_model)
            # Replace the first n_soft positions with learned embeddings
            B = output.shape[0]
            # soft_embeds is (B, n_soft, d_model) — already on correct device after expand
            soft = soft_embeds.to(output.device)
            return torch.cat([soft, output[:, n_soft:, :]], dim=1)
        return hook

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Any]:
        """Forward pass with soft prompt prepended.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            Tuple of (loss, logits, past_key_values) matching backbone format.
            logits cover n_soft + T token positions.
        """
        B, T = input_ids.shape
        n_soft = self.config.n_soft_tokens

        # Build soft prompt embeddings for this batch: (B, n_soft, d_model)
        soft_embeds = self.soft_prompt(B)

        # Create placeholder ids for the soft prompt positions (zeros work fine;
        # the hook will overwrite the resulting embeddings)
        placeholder_ids = torch.zeros(B, n_soft, dtype=input_ids.dtype, device=input_ids.device)
        combined_ids = torch.cat([placeholder_ids, input_ids], dim=1)  # (B, n_soft + T)

        # Register a one-shot hook on the backbone's embedding layer
        hook_handle = self.backbone.embed.register_forward_hook(
            self._make_embedding_hook(soft_embeds, n_soft)
        )
        try:
            loss, logits, pkv = self.backbone(combined_ids)
        finally:
            hook_handle.remove()

        return loss, logits, pkv

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only the soft prompt parameters (backbone is frozen)."""
        return [self.soft_prompt.embeddings]


def get_prompt_tuning_optimizer(
    model: SoftPromptModel,
    base_lr: float,
    config: PromptTuningConfig,
) -> torch.optim.Optimizer:
    """Create an Adam optimizer for the soft prompt only.

    Args:
        model: The SoftPromptModel instance.
        base_lr: Base learning rate (multiplied by config.lr_multiplier for soft prompt).
        config: PromptTuningConfig specifying lr_multiplier.

    Returns:
        Adam optimizer over the soft prompt parameters.
    """
    prompt_lr = base_lr * config.lr_multiplier
    return torch.optim.Adam(model.get_trainable_params(), lr=prompt_lr)


class GistCompressor(nn.Module):
    """Compress long prompt representations into compact Gist tokens.

    Each chunk of compress_ratio consecutive hidden states is cross-attended
    into a single output token via a learnable Gist query.
    """

    def __init__(self, d_model: int, compress_ratio: int) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.gist_query = nn.Parameter(torch.empty(1, d_model))
        nn.init.normal_(self.gist_query, mean=0.0, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def compress(self, hidden_states: Tensor) -> Tensor:
        """Compress hidden states by compress_ratio using cross-attention.

        Args:
            hidden_states: (B, T, d_model) sequence to compress.
                           If T is not divisible by compress_ratio, the
                           remainder is truncated.

        Returns:
            Tensor of shape (B, T // compress_ratio, d_model).
        """
        B, T, d_model = hidden_states.shape
        ratio = self.compress_ratio
        n_chunks = T // ratio  # truncate remainder

        if n_chunks == 0:
            # Return empty tensor with correct shape
            return hidden_states.new_zeros(B, 0, d_model)

        # Truncate to n_chunks * ratio
        hidden_states = hidden_states[:, : n_chunks * ratio, :]  # (B, n_chunks*ratio, d_model)

        # Reshape into chunks: (B * n_chunks, ratio, d_model)
        chunks = hidden_states.reshape(B * n_chunks, ratio, d_model)

        # Expand gist query to match batch dimension: (B * n_chunks, 1, d_model)
        query = self.gist_query.unsqueeze(0).expand(B * n_chunks, -1, -1)

        # Cross-attend: query attends to each chunk
        attn_out, _ = self.cross_attn(query, chunks, chunks)  # (B * n_chunks, 1, d_model)

        # Reshape back to (B, n_chunks, d_model)
        return attn_out.reshape(B, n_chunks, d_model)
