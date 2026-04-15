"""Multilingual data utilities for the Aurelius LLM project.

Provides language configuration, sampling, batch construction, and
cross-lingual alignment tools. Uses pure PyTorch for tensor operations
and pure Python stdlib for text processing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LanguageConfig:
    """Configuration for multilingual data sampling.

    Args:
        language_codes: List of supported ISO language codes.
        sampling_weights: Optional explicit per-language sampling weights.
            If None, weights are derived from corpus counts via temperature
            smoothing.
        temperature: Temperature for sampling weight smoothing. Lower values
            make the distribution more peaked toward high-count languages;
            higher values flatten the distribution.
    """

    language_codes: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "zh"]
    )
    sampling_weights: Optional[Dict[str, float]] = None
    temperature: float = 0.3


@dataclass
class MultilingualBatch:
    """A batch of tokenized multilingual samples.

    Args:
        token_ids: Long tensor of shape (batch_size, max_len) with token IDs.
        language_ids: Long tensor of shape (batch_size,) with per-sample
            language indices.
        attention_mask: Float/bool tensor of shape (batch_size, max_len);
            1 for real tokens, 0 for padding.
        language_labels: List of language code strings, one per sample.
    """

    token_ids: Tensor
    language_ids: Tensor
    attention_mask: Tensor
    language_labels: List[str]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def smooth_language_weights(
    counts: Dict[str, int], temperature: float
) -> Dict[str, float]:
    """Apply temperature smoothing to language corpus counts.

    Computes p_i ∝ count_i^(1/temperature) and normalises so weights sum to 1.
    Languages with missing or zero counts are assigned a count of 1 before
    smoothing.

    Args:
        counts: Mapping of language code -> corpus sample count.
        temperature: Smoothing temperature. Must be > 0.

    Returns:
        Dict mapping language code -> normalised sampling weight.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    exponent = 1.0 / temperature
    smoothed: Dict[str, float] = {}
    for lang, cnt in counts.items():
        effective = max(cnt, 1)
        smoothed[lang] = effective ** exponent

    total = sum(smoothed.values())
    if total == 0:
        n = len(smoothed)
        return {lang: 1.0 / n for lang in smoothed}

    return {lang: v / total for lang, v in smoothed.items()}


def detect_language_heuristic(text: str) -> str:
    """Heuristic language detection based on character scripts and markers.

    Detection priority (first match wins):
    1. Chinese characters (U+4E00–U+9FFF) → "zh"
    2. Arabic script (U+0600–U+06FF) → "ar"
    3. French accent markers (é, ê, è, à, â) → "fr"
    4. Spanish markers (ñ, ¿, ¡) → "es"
    5. German umlauts / sharp-s (ä, ö, ü, ß) → "de"
    6. Default → "en"

    Args:
        text: Input text string.

    Returns:
        ISO 639-1 language code string.
    """
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "zh"
        if 0x0600 <= cp <= 0x06FF:
            return "ar"

    # French markers
    french_markers = set("éêèàâ")
    if any(ch in french_markers for ch in text):
        return "fr"

    # Spanish markers
    spanish_markers = set("ñ¿¡")
    if any(ch in spanish_markers for ch in text):
        return "es"

    # German markers
    german_markers = set("äöüß")
    if any(ch in german_markers for ch in text):
        return "de"

    return "en"


# ---------------------------------------------------------------------------
# LanguageSampler
# ---------------------------------------------------------------------------


class LanguageSampler:
    """Samples (language, text) pairs from a multilingual corpus.

    Args:
        language_data: Mapping of language code -> list of text samples.
        config: LanguageConfig controlling codes, weights, and temperature.
    """

    def __init__(
        self,
        language_data: Dict[str, List[str]],
        config: LanguageConfig,
    ) -> None:
        self._data = language_data
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, float]:
        """Return normalised sampling weights for each language.

        If explicit weights are set in the config, they are normalised and
        returned. Otherwise, weights are derived from corpus counts via
        temperature smoothing.

        Returns:
            Dict mapping language code -> sampling weight (sum = 1).
        """
        if self._config.sampling_weights is not None:
            weights = self._config.sampling_weights
            total = sum(weights.values())
            if total == 0:
                n = len(weights)
                return {lang: 1.0 / n for lang in weights}
            return {lang: w / total for lang, w in weights.items()}

        counts: Dict[str, int] = {
            lang: len(texts) for lang, texts in self._data.items()
        }
        return smooth_language_weights(counts, self._config.temperature)

    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[str, str]]:
        """Sample a batch of (language_code, text) pairs.

        Languages are selected according to smoothed weights, then a random
        text is drawn from that language's corpus.

        Args:
            batch_size: Number of samples to draw.
            rng: Optional seeded random.Random instance for reproducibility.

        Returns:
            List of (language_code, text) tuples, length == batch_size.
        """
        if rng is None:
            rng = random.Random()

        weights = self.get_weights()
        langs = list(weights.keys())
        wts = [weights[l] for l in langs]

        samples: List[Tuple[str, str]] = []
        for _ in range(batch_size):
            lang = rng.choices(langs, weights=wts, k=1)[0]
            texts = self._data.get(lang, [])
            if not texts:
                text = ""
            else:
                text = rng.choice(texts)
            samples.append((lang, text))
        return samples

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-language statistics.

        Returns:
            Dict mapping language code -> {"n_samples": float, "weight": float}.
        """
        weights = self.get_weights()
        stats: Dict[str, Dict[str, float]] = {}
        for lang, texts in self._data.items():
            stats[lang] = {
                "n_samples": float(len(texts)),
                "weight": weights.get(lang, 0.0),
            }
        return stats


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def build_multilingual_batch(
    samples: List[Tuple[str, str]],
    language_to_id: Dict[str, int],
    tokenize_fn: Callable[[str], List[int]],
    max_len: int,
    pad_id: int = 0,
) -> MultilingualBatch:
    """Tokenize and pad a list of (language, text) samples into a batch tensor.

    Tokens beyond max_len are truncated. Shorter sequences are right-padded
    with pad_id. The attention_mask is 1 for real tokens and 0 for padding.

    Args:
        samples: List of (language_code, text) tuples.
        language_to_id: Mapping from language code string to integer ID.
        tokenize_fn: Callable that converts a text string to a list of int
            token IDs.
        max_len: Maximum sequence length (truncate / pad to this length).
        pad_id: Token ID used for padding positions.

    Returns:
        MultilingualBatch with token_ids, language_ids, attention_mask,
        language_labels.
    """
    batch_size = len(samples)
    token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    lang_id_list: List[int] = []
    lang_labels: List[str] = []

    for i, (lang, text) in enumerate(samples):
        tokens = tokenize_fn(text)[:max_len]
        seq_len = len(tokens)
        token_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        lang_id_list.append(language_to_id.get(lang, 0))
        lang_labels.append(lang)

    language_ids = torch.tensor(lang_id_list, dtype=torch.long)
    return MultilingualBatch(
        token_ids=token_ids,
        language_ids=language_ids,
        attention_mask=attention_mask,
        language_labels=lang_labels,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_language_accuracy(
    pred_lang_ids: Tensor,
    true_lang_ids: Tensor,
) -> float:
    """Compute fraction of correctly predicted language IDs.

    Args:
        pred_lang_ids: 1-D long tensor of predicted language IDs.
        true_lang_ids: 1-D long tensor of ground-truth language IDs.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if pred_lang_ids.numel() == 0:
        return 0.0
    correct = (pred_lang_ids == true_lang_ids).sum().item()
    return float(correct) / pred_lang_ids.numel()


# ---------------------------------------------------------------------------
# CrossLingualAligner
# ---------------------------------------------------------------------------


class CrossLingualAligner:
    """Tools for measuring cross-lingual alignment of embeddings.

    Args:
        d_model: Embedding dimensionality.
        n_languages: Number of distinct languages.
    """

    def __init__(self, d_model: int, n_languages: int) -> None:
        self.d_model = d_model
        self.n_languages = n_languages

    def align_loss(self, embeddings: Tensor, language_ids: Tensor) -> Tensor:
        """Compute within-language compactness as mean pairwise L2 distance.

        For each language that appears in language_ids, the mean pairwise
        Euclidean distance between all embeddings of that language is computed.
        The final loss is the mean over all languages that have >= 2 samples.
        If no language has >= 2 samples, returns a zero scalar tensor.

        Args:
            embeddings: Float tensor of shape (batch_size, d_model).
            language_ids: Long tensor of shape (batch_size,).

        Returns:
            Scalar tensor — mean within-language pairwise L2 distance.
        """
        unique_langs = language_ids.unique()
        lang_losses: List[Tensor] = []

        for lang_id in unique_langs:
            mask = language_ids == lang_id
            lang_embs = embeddings[mask]  # (n_lang, d_model)
            if lang_embs.shape[0] < 2:
                continue
            # Pairwise distances: (n_lang, n_lang)
            dists = torch.cdist(lang_embs, lang_embs, p=2)
            # Upper triangle (excluding diagonal) mean
            n = lang_embs.shape[0]
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            mean_dist = dists[triu_mask].mean()
            lang_losses.append(mean_dist)

        if not lang_losses:
            return torch.tensor(0.0, dtype=embeddings.dtype)

        return torch.stack(lang_losses).mean()

    def language_adversarial_loss(
        self, embeddings: Tensor, language_ids: Tensor
    ) -> Tensor:
        """Compute language-invariance penalty as variance of per-language means.

        The per-language mean embedding is computed for each language present in
        the batch. The loss is the mean element-wise variance of these per-language
        centroids — a smaller value indicates more language-invariant embeddings.

        Args:
            embeddings: Float tensor of shape (batch_size, d_model).
            language_ids: Long tensor of shape (batch_size,).

        Returns:
            Scalar tensor — mean variance of per-language centroid matrix.
        """
        unique_langs = language_ids.unique()
        centroids: List[Tensor] = []

        for lang_id in unique_langs:
            mask = language_ids == lang_id
            lang_embs = embeddings[mask]
            centroids.append(lang_embs.mean(dim=0))

        if len(centroids) < 2:
            return torch.tensor(0.0, dtype=embeddings.dtype)

        centroid_matrix = torch.stack(centroids, dim=0)  # (n_langs, d_model)
        # Variance across the language dimension
        var = centroid_matrix.var(dim=0)  # (d_model,)
        return var.mean()
