"""NEFTune: Noisy Embeddings Improve Instruction Finetuning.

Reference: Jain et al., arXiv:2310.05914

NEFTune adds uniform noise to embedding vectors during the forward pass of
training only.  For a sequence of embeddings E ∈ R^{T×d}:

    noise ~ Uniform[-α / sqrt(T·d), +α / sqrt(T·d)]
    E_noisy = E + noise

where α (noise_alpha) is typically in [1, 15], default 5.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class NoisyEmbedding(nn.Embedding):
    """Drop-in replacement for nn.Embedding that adds NEFTune noise during training.

    During ``model.train()`` mode each forward pass adds per-element uniform
    noise scaled by ``noise_alpha / sqrt(seq_len * embedding_dim)``.  In eval
    mode the output is identical to a plain ``nn.Embedding``.

    Args:
        *args: Positional arguments forwarded to ``nn.Embedding``.
        noise_alpha: Noise magnitude hyperparameter α.  Set to 0.0 to disable
            noise entirely.
        **kwargs: Keyword arguments forwarded to ``nn.Embedding``.
    """

    def __init__(self, *args, noise_alpha: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_alpha = noise_alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        embeddings = super().forward(input)

        if self.training and self.noise_alpha != 0.0:
            seq_len, d = embeddings.shape[-2], embeddings.shape[-1]
            scale = self.noise_alpha / math.sqrt(seq_len * d)
            noise = torch.zeros_like(embeddings).uniform_(-scale, scale)
            embeddings = embeddings + noise

        return embeddings


class NEFTuneWrapper:
    """Wraps an existing model's embedding layer with :class:`NoisyEmbedding`.

    Supports dot-path attribute names like ``"transformer.wte"``.

    Args:
        model: The ``nn.Module`` whose embedding will be wrapped.
        embedding_attr: Dot-path attribute name of the embedding layer.
        noise_alpha: NEFTune noise magnitude hyperparameter α.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_attr: str = "embed_tokens",
        noise_alpha: float = 5.0,
    ) -> None:
        self.model = model
        self.embedding_attr = embedding_attr
        self.noise_alpha = noise_alpha
        self._original_embedding: nn.Embedding | None = None

    # ------------------------------------------------------------------
    # Internal helpers for dot-path attribute access
    # ------------------------------------------------------------------

    def _get_parent_and_name(self):
        """Return (parent_module, last_attr_name) for the embedding path."""
        parts = self.embedding_attr.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _get_embedding(self) -> nn.Embedding:
        parent, name = self._get_parent_and_name()
        return getattr(parent, name)

    def _set_embedding(self, module: nn.Module) -> None:
        parent, name = self._get_parent_and_name()
        setattr(parent, name, module)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self) -> NEFTuneWrapper:
        """Replace the embedding layer with a :class:`NoisyEmbedding`.

        The original embedding weight is copied (shared) into the new layer.
        Calling ``apply()`` a second time without ``remove()`` is a no-op.

        Returns:
            self, for chaining.
        """
        original = self._get_embedding()
        if isinstance(original, NoisyEmbedding):
            # Already applied — nothing to do.
            return self

        self._original_embedding = original

        noisy = NoisyEmbedding(
            original.num_embeddings,
            original.embedding_dim,
            padding_idx=original.padding_idx,
            max_norm=original.max_norm,
            norm_type=original.norm_type,
            scale_grad_by_freq=original.scale_grad_by_freq,
            sparse=original.sparse,
            noise_alpha=self.noise_alpha,
        )
        # Share the weight tensor so no extra memory is used.
        noisy.weight = original.weight
        self._set_embedding(noisy)
        return self

    def remove(self) -> NEFTuneWrapper:
        """Restore the original ``nn.Embedding`` layer.

        Returns:
            self, for chaining.
        """
        if self._original_embedding is None:
            return self
        self._set_embedding(self._original_embedding)
        self._original_embedding = None
        return self


class NEFTuneTrainer:
    """Thin training-step wrapper that applies NEFTune noise to pre-looked-up embeddings.

    This trainer operates on embeddings that have *already* been retrieved from
    the embedding table (shape ``(B, T, d)``).  During each ``train_step`` it
    injects NEFTune noise, computes the loss, runs ``backward()``, and calls
    ``optimizer.step()``.

    Args:
        model: The model to train (should be in train mode).
        optimizer: A ``torch.optim.Optimizer`` instance.
        noise_alpha: NEFTune noise magnitude hyperparameter α.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        noise_alpha: float = 5.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.noise_alpha = noise_alpha

    def _add_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add NEFTune uniform noise to ``embeddings`` (B, T, d)."""
        seq_len = embeddings.shape[-2]
        d = embeddings.shape[-1]
        scale = self.noise_alpha / math.sqrt(seq_len * d)
        noise = torch.zeros_like(embeddings).uniform_(-scale, scale)
        return embeddings + noise

    def train_step(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
    ) -> float:
        """Perform one training step with NEFTune noise.

        Args:
            embeddings: Pre-looked-up embeddings of shape ``(B, T, d)``.
            targets: Target token indices of shape ``(B, T)``.
            loss_fn: Callable ``(logits, targets) -> scalar loss``.

        Returns:
            The scalar loss value (Python float) for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        noisy_embeddings = self._add_noise(embeddings)
        logits = self.model(noisy_embeddings)
        loss = loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()
