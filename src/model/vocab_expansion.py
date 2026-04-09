"""Vocabulary expansion: add new tokens to a trained model while preserving existing embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class VocabExpansionConfig:
    """Configuration for vocabulary expansion during continual learning."""
    init_strategy: str = "mean"       # "mean" | "random" | "similar_token" | "zeros"
    freeze_existing: bool = True      # freeze original token embeddings during fine-tuning
    new_token_lr_scale: float = 10.0  # scale LR for new tokens vs. existing
    n_init_similar_tokens: int = 5    # for "similar_token": average N most similar existing tokens


def _init_new_rows(
    existing_weight: Tensor,
    n_new_tokens: int,
    config: VocabExpansionConfig,
) -> Tensor:
    """Compute initialization values for new token rows.

    Args:
        existing_weight: (old_vocab, d_model) existing embedding weights.
        n_new_tokens: Number of new token rows to create.
        config: VocabExpansionConfig specifying initialization strategy.

    Returns:
        (n_new_tokens, d_model) tensor of initial values.
    """
    old_vocab, d_model = existing_weight.shape

    if config.init_strategy == "mean":
        mean_vec = existing_weight.mean(dim=0, keepdim=True)  # (1, d_model)
        return mean_vec.expand(n_new_tokens, d_model).clone()

    elif config.init_strategy == "random":
        mean_vec = existing_weight.mean(dim=0)
        std_vec = existing_weight.std(dim=0)
        new_rows = torch.zeros(n_new_tokens, d_model, dtype=existing_weight.dtype)
        for i in range(n_new_tokens):
            new_rows[i] = mean_vec + std_vec * torch.randn_like(mean_vec)
        return new_rows

    elif config.init_strategy == "zeros":
        return torch.zeros(n_new_tokens, d_model, dtype=existing_weight.dtype)

    elif config.init_strategy == "similar_token":
        # For each new token, average the top-N most similar existing tokens
        # Similarity is measured by cosine similarity to the overall mean embedding
        mean_vec = existing_weight.mean(dim=0)  # (d_model,)
        # Cosine similarities of each existing token to the mean
        norms = F.normalize(existing_weight, dim=1)  # (old_vocab, d_model)
        mean_norm = F.normalize(mean_vec.unsqueeze(0), dim=1)  # (1, d_model)
        cos_sims = (norms * mean_norm).sum(dim=1)  # (old_vocab,)

        n_similar = min(config.n_init_similar_tokens, old_vocab)
        _, top_indices = torch.topk(cos_sims, n_similar)
        similar_mean = existing_weight[top_indices].mean(dim=0, keepdim=True)  # (1, d_model)
        return similar_mean.expand(n_new_tokens, d_model).clone()

    else:
        raise ValueError(
            f"Unknown init_strategy '{config.init_strategy}'. "
            "Choose from: 'mean', 'random', 'similar_token', 'zeros'."
        )


def expand_embedding(
    embedding: nn.Embedding,
    n_new_tokens: int,
    config: VocabExpansionConfig,
) -> nn.Embedding:
    """Expand an embedding table to include new tokens.

    Args:
        embedding: Existing nn.Embedding(old_vocab, d_model).
        n_new_tokens: Number of new tokens to add.
        config: VocabExpansionConfig specifying initialization strategy.

    Returns:
        New nn.Embedding(old_vocab + n_new_tokens, d_model) with old weights
        preserved in the first old_vocab rows.
    """
    old_vocab, d_model = embedding.weight.shape
    new_vocab = old_vocab + n_new_tokens

    new_embedding = nn.Embedding(new_vocab, d_model)

    with torch.no_grad():
        # Copy old weights exactly
        new_embedding.weight[:old_vocab] = embedding.weight.data.clone()

        # Initialize new rows
        new_rows = _init_new_rows(embedding.weight.data, n_new_tokens, config)
        new_embedding.weight[old_vocab:] = new_rows

    return new_embedding


def expand_lm_head(
    lm_head: nn.Linear,
    n_new_tokens: int,
    config: VocabExpansionConfig,
) -> nn.Linear:
    """Expand an LM head linear layer to predict new tokens.

    Args:
        lm_head: Existing nn.Linear(d_model, old_vocab, bias=False).
        n_new_tokens: Number of new token output dimensions to add.
        config: VocabExpansionConfig specifying initialization strategy.

    Returns:
        New nn.Linear(d_model, old_vocab + n_new_tokens, bias=False) with old
        output rows preserved.
    """
    old_vocab, d_model = lm_head.weight.shape  # weight is (out_features, in_features)
    new_vocab = old_vocab + n_new_tokens

    new_head = nn.Linear(d_model, new_vocab, bias=False)

    with torch.no_grad():
        # Copy old output rows exactly
        new_head.weight[:old_vocab] = lm_head.weight.data.clone()

        # Initialize new output rows using the same strategy
        new_rows = _init_new_rows(lm_head.weight.data, n_new_tokens, config)
        new_head.weight[old_vocab:] = new_rows

    return new_head


def get_new_token_params(
    model: nn.Module,
    old_vocab_size: int,
) -> list[nn.Parameter]:
    """Return parameter slices for new tokens (rows old_vocab_size: onward).

    Finds nn.Embedding and nn.Linear layers whose first dimension matches vocab
    and returns sliced views of the new token rows as parameters. Useful for
    configuring separate learning rates for new vs. existing tokens.

    Args:
        model: The model after vocabulary expansion.
        old_vocab_size: The original vocabulary size before expansion.

    Returns:
        List of nn.Parameter objects corresponding to new token rows.
    """
    new_params: list[nn.Parameter] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            vocab_dim = module.weight.shape[0]
            if vocab_dim > old_vocab_size:
                # Slice the new rows
                new_slice = nn.Parameter(module.weight[old_vocab_size:])
                new_params.append(new_slice)

        elif isinstance(module, nn.Linear) and module.bias is None:
            # LM head: weight is (out_features=vocab, in_features=d_model)
            out_dim = module.weight.shape[0]
            if out_dim > old_vocab_size:
                new_slice = nn.Parameter(module.weight[old_vocab_size:])
                new_params.append(new_slice)

    return new_params


class VocabExpander:
    """Expands a model's vocabulary in-place for continual learning.

    Handles expanding both the embedding table and LM head, optionally freezing
    existing token embeddings so only new tokens are updated during fine-tuning.
    """

    def __init__(self, model: nn.Module, config: VocabExpansionConfig) -> None:
        self.model = model
        self.config = config
        self._old_vocab_size: int | None = None

    def expand(self, n_new_tokens: int) -> nn.Module:
        """Expand the model's embedding table and LM head in-place.

        Args:
            n_new_tokens: Number of new tokens to add to the vocabulary.

        Returns:
            The modified model (same object, mutated in-place).
        """
        model = self.model

        # Find the embedding layer
        if not hasattr(model, "embed") or not isinstance(model.embed, nn.Embedding):
            raise AttributeError(
                "Model must have a 'embed' attribute of type nn.Embedding. "
                f"Got: {type(getattr(model, 'embed', None))}"
            )

        old_vocab_size = model.embed.weight.shape[0]
        self._old_vocab_size = old_vocab_size

        # Expand embedding
        new_embedding = expand_embedding(model.embed, n_new_tokens, self.config)

        # Expand LM head (handle tied and untied weights)
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            # Check if weights are tied (same storage)
            embed_weight_data_ptr = model.embed.weight.data_ptr()
            lm_head_weight_data_ptr = model.lm_head.weight.data_ptr()
            weights_tied = embed_weight_data_ptr == lm_head_weight_data_ptr

            new_lm_head = expand_lm_head(model.lm_head, n_new_tokens, self.config)
            model.lm_head = new_lm_head

            if weights_tied:
                # Re-tie the weights
                model.lm_head.weight = new_embedding.weight

        # Replace the embedding after potentially re-tying lm_head
        model.embed = new_embedding

        # Update model's vocab_size if config has one
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            model.config.vocab_size = old_vocab_size + n_new_tokens

        # Apply gradient freezing for existing tokens
        if self.config.freeze_existing:
            self._register_freeze_hooks()

        return model

    def _register_freeze_hooks(self) -> None:
        """Register backward hooks to zero out gradients for existing token rows."""
        old_vocab = self._old_vocab_size
        if old_vocab is None:
            return

        def make_embed_hook(size):
            def hook(grad):
                if grad is not None:
                    masked = grad.clone()
                    masked[:size] = 0.0
                    return masked
                return grad
            return hook

        model = self.model
        if hasattr(model, "embed") and isinstance(model.embed, nn.Embedding):
            model.embed.weight.register_hook(make_embed_hook(old_vocab))

        # Only register on lm_head if weights are NOT tied (otherwise handled by embed hook)
        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            embed_ptr = model.embed.weight.data_ptr()
            lm_head_ptr = model.lm_head.weight.data_ptr()
            if embed_ptr != lm_head_ptr:
                model.lm_head.weight.register_hook(make_embed_hook(old_vocab))

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return optimizer param groups with separate LRs for new vs. existing tokens.

        Args:
            base_lr: Learning rate for existing parameters.

        Returns:
            List of two dicts: existing params at base_lr, new token params at
            base_lr * new_token_lr_scale.
        """
        if self._old_vocab_size is None:
            raise RuntimeError("Call expand() before get_param_groups().")

        model = self.model
        old_vocab = self._old_vocab_size

        # Collect all parameters
        all_params = set(model.parameters())

        # Identify new-token parameter IDs by name/slice logic
        # New token rows are [old_vocab:] in embed.weight and lm_head.weight
        new_token_param_ids: set[int] = set()

        # For embed and lm_head we return the full parameter objects but specify
        # which rows are "new". Since PyTorch doesn't support partial-parameter
        # LR groups natively, we return the full weight params for new-token group
        # and exclude them from existing group.
        new_token_params: list[nn.Parameter] = []

        if hasattr(model, "embed") and isinstance(model.embed, nn.Embedding):
            if model.embed.weight.shape[0] > old_vocab:
                new_token_params.append(model.embed.weight)
                new_token_param_ids.add(id(model.embed.weight))

        if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
            lm_weight = model.lm_head.weight
            if lm_weight.shape[0] > old_vocab and id(lm_weight) not in new_token_param_ids:
                new_token_params.append(lm_weight)
                new_token_param_ids.add(id(lm_weight))

        existing_params = [
            p for p in all_params if id(p) not in new_token_param_ids
        ]

        return [
            {"params": existing_params, "lr": base_lr},
            {"params": new_token_params, "lr": base_lr * self.config.new_token_lr_scale},
        ]


def verify_expansion(
    old_embedding: nn.Embedding,
    new_embedding: nn.Embedding,
) -> dict[str, bool]:
    """Sanity-check that an expanded embedding correctly preserved old weights.

    Args:
        old_embedding: The original embedding before expansion.
        new_embedding: The expanded embedding after expansion.

    Returns:
        Dict with boolean checks:
            old_weights_preserved: All old token weights match exactly.
            size_correct: New embedding has more tokens than old.
            no_nan: No NaN values in the new embedding weights.
    """
    old_vocab = old_embedding.weight.shape[0]

    old_weights_preserved = torch.allclose(
        new_embedding.weight[:old_vocab].data,
        old_embedding.weight.data,
    )

    size_correct = new_embedding.weight.shape[0] > old_embedding.weight.shape[0]

    no_nan = not torch.isnan(new_embedding.weight).any().item()

    return {
        "old_weights_preserved": old_weights_preserved,
        "size_correct": size_correct,
        "no_nan": no_nan,
    }
