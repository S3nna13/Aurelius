"""Cross-lingual transfer learning and multilingual training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MultilingualConfig:
    """Configuration for multilingual training."""

    languages: list[str] = field(default_factory=lambda: ["en", "fr", "de"])
    temperature: float = 5.0
    alpha: float = 0.3
    upsample_low_resource: bool = True


def compute_language_sampling_probs(
    language_counts: dict[str, int],
    temperature: float,
    alpha: float,
) -> dict[str, float]:
    """Compute temperature-scaled language sampling probabilities.

    p(L) ∝ count(L)^(1/T), then mix with uniform:
        final = (1 - alpha) * p_scaled + alpha * p_uniform

    Args:
        language_counts: Mapping from language code to token/sample count.
        temperature: Smoothing temperature T (higher → more uniform).
        alpha: Weight for uniform component (0 = pure scaled, 1 = pure uniform).

    Returns:
        Normalized probability dict summing to 1.0.
    """
    languages = list(language_counts.keys())
    n = len(languages)
    if n == 0:
        return {}

    # Temperature-scaled counts
    scaled = {lang: language_counts[lang] ** (1.0 / temperature) for lang in languages}
    total_scaled = sum(scaled.values())
    p_scaled = {lang: scaled[lang] / total_scaled for lang in languages}

    # Uniform component
    p_uniform = {lang: 1.0 / n for lang in languages}

    # Mix
    mixed = {
        lang: (1.0 - alpha) * p_scaled[lang] + alpha * p_uniform[lang]
        for lang in languages
    }

    # Normalize to correct any floating-point drift
    total_mixed = sum(mixed.values())
    return {lang: mixed[lang] / total_mixed for lang in languages}


def cross_lingual_alignment_loss(emb_source: Tensor, emb_target: Tensor) -> Tensor:
    """MSE between L2-normalized source and target sentence embeddings.

    Encourages aligned cross-lingual representations.

    Args:
        emb_source: (batch, d_model) source language embeddings.
        emb_target: (batch, d_model) target language embeddings.

    Returns:
        Scalar MSE loss.
    """
    norm_source = F.normalize(emb_source, p=2, dim=-1)
    norm_target = F.normalize(emb_target, p=2, dim=-1)
    return F.mse_loss(norm_source, norm_target)


class LanguageTaggedDataset:
    """Dataset wrapper that tracks language tags for each sample."""

    def __init__(self, samples: list[dict], lang_key: str = "lang") -> None:
        self.samples = samples
        self.lang_key = lang_key

        # Build index: language -> list of sample indices
        self._index: dict[str, list[int]] = {}
        for i, sample in enumerate(samples):
            lang = sample.get(lang_key, "unknown")
            self._index.setdefault(lang, []).append(i)

    def get_by_language(self, lang: str) -> list[dict]:
        """Return all samples for a given language."""
        indices = self._index.get(lang, [])
        return [self.samples[i] for i in indices]

    def sample_batch(
        self,
        probs: dict[str, float],
        batch_size: int,
        rng: random.Random | None = None,
    ) -> list[dict]:
        """Sample a batch of batch_size items using per-language probabilities.

        Args:
            probs: Language -> probability mapping (should sum to 1).
            batch_size: Total number of samples to draw.
            rng: Optional random.Random instance for reproducibility.

        Returns:
            List of sampled dicts.
        """
        if rng is None:
            rng = random.Random()

        languages = list(probs.keys())
        weights = [probs[lang] for lang in languages]

        batch: list[dict] = []
        for _ in range(batch_size):
            # Pick language according to weights
            chosen_lang = rng.choices(languages, weights=weights, k=1)[0]
            candidates = self._index.get(chosen_lang, [])
            if not candidates:
                # Fallback: pick from any available sample
                idx = rng.randrange(len(self.samples))
            else:
                idx = rng.choice(candidates)
            batch.append(self.samples[idx])

        return batch


class MultilingualTrainer:
    """Trains a model with language-balanced sampling and alignment objectives."""

    def __init__(
        self,
        model: nn.Module,
        config: MultilingualConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train_step(self, batch_ids: list[Tensor]) -> dict[str, Any]:
        """Concatenate a list of per-language token tensors, forward, CE loss, backward.

        Args:
            batch_ids: List of (seq_len,) or (batch, seq_len) tensors, one per language.

        Returns:
            {"loss": float, "n_languages": int}
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Stack into a single (total_batch, seq_len) tensor
        stacked = torch.stack(
            [t if t.dim() == 1 else t.squeeze(0) for t in batch_ids], dim=0
        )

        input_ids = stacked[:, :-1]   # (B, S-1)
        labels = stacked[:, 1:]       # (B, S-1)

        loss, logits, _ = self.model(input_ids)

        # Compute cross-entropy manually (model may not receive labels here)
        B, S, V = logits.shape
        ce_loss = F.cross_entropy(logits.reshape(B * S, V), labels.reshape(B * S))

        ce_loss.backward()
        self.optimizer.step()

        return {"loss": ce_loss.item(), "n_languages": len(batch_ids)}

    def alignment_step(self, source_ids: Tensor, target_ids: Tensor) -> dict[str, Any]:
        """Extract last-layer embeddings via hook, compute alignment loss, backward.

        Args:
            source_ids: (batch, seq_len) source language token ids.
            target_ids: (batch, seq_len) target language token ids.

        Returns:
            {"alignment_loss": float}
        """
        self.model.train()
        self.optimizer.zero_grad()

        last_layer = self.model.layers[-1]

        # --- Extract source embeddings ---
        source_hidden: list[Tensor] = []

        def _hook_source(module: nn.Module, inp: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            source_hidden.append(hidden)

        handle_src = last_layer.register_forward_hook(_hook_source)
        self.model(source_ids)
        handle_src.remove()

        # --- Extract target embeddings ---
        target_hidden: list[Tensor] = []

        def _hook_target(module: nn.Module, inp: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            target_hidden.append(hidden)

        handle_tgt = last_layer.register_forward_hook(_hook_target)
        self.model(target_ids)
        handle_tgt.remove()

        # Mean pooling over sequence dimension → (batch, d_model)
        emb_source = source_hidden[0].mean(dim=1)
        emb_target = target_hidden[0].mean(dim=1)

        align_loss = cross_lingual_alignment_loss(emb_source, emb_target)
        align_loss.backward()
        self.optimizer.step()

        return {"alignment_loss": align_loss.item()}
