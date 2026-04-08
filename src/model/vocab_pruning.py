"""Vocabulary pruning: remove unused tokens to reduce model size.

Given a list of token IDs to keep, creates a remapped model with a smaller
embedding table. Provides a remapping function for converting old IDs → new IDs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VocabPruningResult:
    """Results of vocabulary pruning."""

    old_vocab_size: int
    new_vocab_size: int
    tokens_removed: int
    embedding_params_saved: int  # params removed from embedding
    id_map: dict[int, int]  # old_id → new_id for kept tokens

    @property
    def compression_ratio(self) -> float:
        return self.old_vocab_size / self.new_vocab_size

    @property
    def tokens_kept(self) -> int:
        return self.new_vocab_size


def collect_token_frequencies(
    token_sequences: list[list[int]],
) -> dict[int, int]:
    """Count how many times each token ID appears across all sequences."""
    frequencies: dict[int, int] = {}
    for sequence in token_sequences:
        for token_id in sequence:
            frequencies[token_id] = frequencies.get(token_id, 0) + 1
    return frequencies


def select_top_k_tokens(
    frequencies: dict[int, int],
    k: int,
    always_keep: list[int] | None = None,  # token IDs to always keep (e.g., special tokens)
) -> list[int]:
    """Select k most frequent tokens, always including tokens in always_keep.

    Returns sorted list of token IDs to keep.
    """
    always_keep_set: set[int] = set(always_keep) if always_keep else set()

    # Start with always_keep tokens
    kept: set[int] = set(always_keep_set)

    # Sort remaining tokens by frequency descending, then fill up to k
    remaining_slots = k - len(kept)
    if remaining_slots > 0:
        # Sort by frequency descending, breaking ties by token ID ascending for determinism
        sorted_tokens = sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for token_id, _ in sorted_tokens:
            if token_id not in kept:
                kept.add(token_id)
                remaining_slots -= 1
                if remaining_slots == 0:
                    break

    return sorted(kept)


def remap_token_ids(
    token_ids: list[int],
    id_map: dict[int, int],
    unk_id: int = 0,  # ID to use for tokens not in id_map
) -> list[int]:
    """Remap a sequence of token IDs using the pruning id_map."""
    return [id_map.get(token_id, unk_id) for token_id in token_ids]


def prune_vocabulary(
    model: nn.Module,
    tokens_to_keep: list[int],
    d_model: int,
) -> tuple[nn.Module, VocabPruningResult]:
    """Prune embedding and lm_head to only include tokens_to_keep.

    1. Sort tokens_to_keep for determinism
    2. Create id_map: old_id → new_id (position in sorted tokens_to_keep)
    3. Slice model.embed.weight[tokens_to_keep] → new embedding
    4. Update model.embed (nn.Embedding with new vocab_size)
    5. If model.lm_head exists and is separate: slice similarly
    6. Update model config vocab_size if accessible

    Returns (pruned_model, VocabPruningResult).
    Does NOT modify the input model in-place — returns a modified copy.
    """
    # Deep copy to avoid modifying the original
    pruned_model = copy.deepcopy(model)

    # 1. Sort tokens_to_keep for determinism
    sorted_tokens = sorted(tokens_to_keep)
    new_vocab_size = len(sorted_tokens)
    old_vocab_size = pruned_model.embed.num_embeddings

    # 2. Create id_map: old_id → new_id
    id_map: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(sorted_tokens)}

    # Index tensor for slicing
    keep_indices = torch.tensor(sorted_tokens, dtype=torch.long)

    # 3. Slice embedding weights
    old_embed_weight = pruned_model.embed.weight.data
    new_embed_weight = old_embed_weight[keep_indices].clone()

    # Check if lm_head is tied to embed (same weight object)
    tied = (
        hasattr(pruned_model, "lm_head")
        and isinstance(pruned_model.lm_head, nn.Linear)
        and pruned_model.lm_head.weight.data_ptr() == pruned_model.embed.weight.data_ptr()
    )

    # 4. Replace model.embed with new smaller embedding
    new_embed = nn.Embedding(new_vocab_size, d_model)
    new_embed.weight = nn.Parameter(new_embed_weight)
    pruned_model.embed = new_embed

    # 5. Handle lm_head
    if hasattr(pruned_model, "lm_head") and isinstance(pruned_model.lm_head, nn.Linear):
        if tied:
            # Tie to the new embedding weight
            pruned_model.lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
            pruned_model.lm_head.weight = pruned_model.embed.weight
        else:
            # Separate lm_head — slice its weight rows
            old_lm_head_weight = pruned_model.lm_head.weight.data
            new_lm_head_weight = old_lm_head_weight[keep_indices].clone()
            new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
            new_lm_head.weight = nn.Parameter(new_lm_head_weight)
            pruned_model.lm_head = new_lm_head

    # 6. Update config vocab_size if accessible
    if hasattr(pruned_model, "config") and hasattr(pruned_model.config, "vocab_size"):
        pruned_model.config.vocab_size = new_vocab_size

    # Calculate parameter savings (embedding table rows removed × d_model per row)
    tokens_removed = old_vocab_size - new_vocab_size
    embedding_params_saved = tokens_removed * d_model

    result = VocabPruningResult(
        old_vocab_size=old_vocab_size,
        new_vocab_size=new_vocab_size,
        tokens_removed=tokens_removed,
        embedding_params_saved=embedding_params_saved,
        id_map=id_map,
    )

    return pruned_model, result
