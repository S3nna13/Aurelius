"""Multimodal supervised fine-tuning: instruction tuning with interleaved image and text tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultimodalSFTConfig:
    """Configuration for multimodal supervised fine-tuning."""

    image_token_id: int = 1
    n_image_tokens: int = 16
    max_seq_len: int = 512
    loss_on_image_tokens: bool = False
    learning_rate: float = 2e-5
    image_loss_weight: float = 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultimodalExample:
    """A single multimodal training example with interleaved image and text tokens."""

    text_ids: list[int]
    image_features: list[torch.Tensor] | None
    labels: list[int]
    n_images: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def expand_image_tokens(
    text_ids: list[int],
    image_features: list[torch.Tensor],
    image_token_id: int,
    n_image_tokens: int,
) -> tuple[torch.Tensor, list[int]]:
    """Replace each image_token_id placeholder with n_image_tokens actual image feature indices.

    Returns:
        expanded_ids: Tensor of token IDs with image placeholders replaced by sentinel indices.
        image_positions: Flat list of token positions that are image tokens.
    """
    expanded_ids: list[int] = []
    image_positions: list[int] = []
    image_idx = 0

    for tok in text_ids:
        if tok == image_token_id:
            start_pos = len(expanded_ids)
            for j in range(n_image_tokens):
                image_positions.append(start_pos + j)
                expanded_ids.append(-(image_idx * n_image_tokens + j + 1))
            image_idx += 1
        else:
            expanded_ids.append(tok)

    return torch.tensor(expanded_ids, dtype=torch.long), image_positions


def build_multimodal_labels(
    text_ids: list[int],
    labels: list[int],
    image_token_id: int,
    n_image_tokens: int,
    loss_on_image: bool = False,
) -> torch.Tensor:
    """Expand labels to match expanded sequence (image placeholders to n_image_tokens labels).

    Args:
        text_ids: Original text token IDs with image placeholders.
        labels: Original labels of same length as text_ids.
        image_token_id: The special token ID used as an image placeholder.
        n_image_tokens: Number of tokens each image expands to.
        loss_on_image: If False, image token positions get label -100.

    Returns:
        Expanded label tensor of shape (expanded_len,).
    """
    expanded_labels: list[int] = []

    for tok, lbl in zip(text_ids, labels):
        if tok == image_token_id:
            img_label = lbl if (loss_on_image and lbl != -100) else -100
            for _ in range(n_image_tokens):
                expanded_labels.append(img_label)
        else:
            expanded_labels.append(lbl)

    return torch.tensor(expanded_labels, dtype=torch.long)


def inject_image_features(
    embeddings: torch.Tensor,
    image_features: list[torch.Tensor],
    image_positions: list[int],
) -> torch.Tensor:
    """Replace embedding rows at image_positions with corresponding image features.

    Args:
        embeddings: (B, T, D) text embeddings.
        image_features: Flat list of image feature tensors, each shape (n_image_tokens, D).
        image_positions: Which positions in the sequence to replace.

    Returns:
        Modified (B, T, D) tensor without in-place modification of original.
    """
    result = embeddings.clone()

    if not image_positions or not image_features:
        return result

    all_features = torch.cat(image_features, dim=0)

    for feat_idx, seq_pos in enumerate(image_positions):
        if seq_pos < result.shape[1] and feat_idx < all_features.shape[0]:
            result[:, seq_pos, :] = all_features[feat_idx].unsqueeze(0)

    return result


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class MultimodalSFTCollator:
    """Collates MultimodalExample list into batch tensors."""

    def __init__(self, config: MultimodalSFTConfig, pad_id: int = 0) -> None:
        self.config = config
        self.pad_id = pad_id

    def __call__(self, examples: list[MultimodalExample]) -> dict[str, Any]:
        """Collate a list of MultimodalExample into a batch dict.

        Returns:
            Dict with input_ids (B, max_seq_len), labels (B, max_seq_len),
            image_positions list[list[int]].
        """
        max_len = self.config.max_seq_len
        batch_input_ids: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        batch_image_positions: list[list[int]] = []

        for ex in examples:
            image_feats = ex.image_features if ex.image_features is not None else []
            expanded_ids, image_positions = expand_image_tokens(
                ex.text_ids,
                image_feats,
                self.config.image_token_id,
                self.config.n_image_tokens,
            )

            expanded_labels = build_multimodal_labels(
                ex.text_ids,
                ex.labels,
                self.config.image_token_id,
                self.config.n_image_tokens,
                loss_on_image=self.config.loss_on_image_tokens,
            )

            T = expanded_ids.shape[0]

            if T >= max_len:
                input_ids_padded = expanded_ids[:max_len]
                labels_padded = expanded_labels[:max_len]
                image_positions = [p for p in image_positions if p < max_len]
            else:
                pad_len = max_len - T
                input_ids_padded = torch.cat([
                    expanded_ids,
                    torch.full((pad_len,), self.pad_id, dtype=torch.long),
                ])
                labels_padded = torch.cat([
                    expanded_labels,
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])

            batch_input_ids.append(input_ids_padded)
            batch_labels.append(labels_padded)
            batch_image_positions.append(image_positions)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "image_positions": batch_image_positions,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultimodalSFTTrainer:
    """SFT trainer for multimodal instruction following."""

    def __init__(
        self,
        model: nn.Module,
        embed_layer: nn.Embedding,
        config: MultimodalSFTConfig,
        optimizer,
    ) -> None:
        self.model = model
        self.embed_layer = embed_layer
        self.config = config
        self.optimizer = optimizer

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute shifted cross-entropy loss ignoring -100."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        n_tokens = int((shift_labels != -100).sum().item())
        return loss, n_tokens

    def train_step(self, batch: dict) -> dict[str, float]:
        """Perform one training step.

        Returns dict with 'loss' and 'n_tokens'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"]
        safe_ids = input_ids.clamp(min=0)

        _, logits, _ = self.model(safe_ids)

        labels = batch["labels"]
        loss, n_tokens = self._compute_loss(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "n_tokens": n_tokens}

    @torch.no_grad()
    def evaluate_step(self, batch: dict) -> dict[str, float]:
        """Perform one evaluation step (no grad, no optimizer step).

        Returns dict with 'loss' and 'n_tokens'.
        """
        self.model.train(False)

        input_ids = batch["input_ids"]
        safe_ids = input_ids.clamp(min=0)

        _, logits, _ = self.model(safe_ids)

        labels = batch["labels"]
        loss, n_tokens = self._compute_loss(logits, labels)

        return {"loss": loss.item(), "n_tokens": n_tokens}
