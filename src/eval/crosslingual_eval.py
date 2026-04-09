"""Cross-lingual transfer evaluation for AureliusTransformer.

Measures how well a model transfers knowledge across languages using
parallel text pairs and synthetic multilingual data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config & data structures
# ---------------------------------------------------------------------------

@dataclass
class CrosslingualConfig:
    languages: list[str] = field(default_factory=lambda: ["en", "fr", "de", "es", "zh"])
    n_eval_samples: int = 100
    max_seq_len: int = 128
    metric: str = "perplexity"  # "perplexity" | "accuracy" | "both"


@dataclass
class LanguagePair:
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str


@dataclass
class TransferResult:
    source_lang: str
    target_lang: str
    source_score: float
    target_score: float
    transfer_gap: float        # target_score - source_score (positive = better in target)
    relative_transfer: float   # target_score / source_score


# ---------------------------------------------------------------------------
# Synthetic multilingual templates
# ---------------------------------------------------------------------------

LANG_TEMPLATES = {
    "en": ["The model processes {n} tokens.", "Output: {result}.", "Step {n}: analyze."],
    "fr": ["Le modele traite {n} tokens.", "Sortie: {result}.", "Etape {n}: analyser."],
    "de": ["Das Modell verarbeitet {n} Token.", "Ausgabe: {result}.", "Schritt {n}: analysieren."],
    "es": ["El modelo procesa {n} fichas.", "Salida: {result}.", "Paso {n}: analizar."],
    "zh": ["Moxing chuli {n} ge ling pai.", "Shuchu: {result}.", "Buzhou {n}: fenxi."],
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_parallel_data(
    n_samples: int,
    languages: list[str],
    seed: int = 42,
) -> list[LanguagePair]:
    """Generate synthetic parallel sentences using LANG_TEMPLATES.

    Creates n_samples pairs, cycling through language combinations.
    Returns list of LanguagePair.
    """
    rng = random.Random(seed)
    pairs: list[LanguagePair] = []

    # Build all language combinations (src != tgt)
    lang_combos: list[tuple[str, str]] = [
        (src, tgt)
        for src in languages
        for tgt in languages
        if src != tgt
    ]
    if not lang_combos:
        return pairs

    for i in range(n_samples):
        src_lang, tgt_lang = lang_combos[i % len(lang_combos)]
        n_val = rng.randint(1, 100)
        result_val = rng.randint(0, 999)
        template_idx = rng.randrange(len(LANG_TEMPLATES[src_lang]))

        src_tmpl = LANG_TEMPLATES[src_lang][template_idx % len(LANG_TEMPLATES[src_lang])]
        tgt_tmpl = LANG_TEMPLATES[tgt_lang][template_idx % len(LANG_TEMPLATES[tgt_lang])]

        src_text = src_tmpl.format(n=n_val, result=result_val)
        tgt_text = tgt_tmpl.format(n=n_val, result=result_val)

        pairs.append(LanguagePair(
            source_lang=src_lang,
            target_lang=tgt_lang,
            source_text=src_text,
            target_text=tgt_text,
        ))

    return pairs


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    texts: list[str],
    max_seq_len: int = 128,
) -> float:
    """Compute mean perplexity over texts.

    For each text: encode, run model, compute CE loss from logits, exponentiate.
    Returns mean perplexity (lower = better).
    """
    model.train(False)
    device = next(model.parameters()).device
    ppls: list[float] = []

    for text in texts:
        ids = encode_fn(text)
        if len(ids) < 2:
            # Need at least 2 tokens to compute next-token prediction
            ids = ids + [0]  # pad with a zero token

        ids = ids[:max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        _, logits, _ = model(input_ids)

        # Shift: predict position i+1 from position i
        shifted_logits = logits[:, :-1, :]           # (1, T-1, V)
        shifted_labels = input_ids[:, 1:]             # (1, T-1)
        V = shifted_logits.size(-1)

        loss = F.cross_entropy(
            shifted_logits.reshape(-1, V),
            shifted_labels.reshape(-1),
        )
        ppl = math.exp(loss.item())
        ppls.append(ppl)

    if not ppls:
        return float("inf")
    return sum(ppls) / len(ppls)


@torch.no_grad()
def compute_token_accuracy(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    texts: list[str],
    max_seq_len: int = 128,
) -> float:
    """Compute next-token prediction accuracy over texts."""
    model.train(False)
    device = next(model.parameters()).device
    correct = 0
    total = 0

    for text in texts:
        ids = encode_fn(text)
        if len(ids) < 2:
            ids = ids + [0]

        ids = ids[:max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        _, logits, _ = model(input_ids)

        shifted_logits = logits[:, :-1, :]   # (1, T-1, V)
        shifted_labels = input_ids[:, 1:]     # (1, T-1)

        preds = shifted_logits.argmax(dim=-1)  # (1, T-1)
        correct += (preds == shifted_labels).sum().item()
        total += shifted_labels.numel()

    if total == 0:
        return 0.0
    return correct / total


# ---------------------------------------------------------------------------
# Language similarity heuristic
# ---------------------------------------------------------------------------

def compute_language_similarity(lang_a: str, lang_b: str) -> float:
    """Heuristic language similarity score in [0, 1].

    Same language -> 1.0
    (en, fr), (en, es), (fr, es), (fr, de) -> 0.6-0.8 (European related)
    (en, zh), (fr, zh), (de, zh) -> 0.2-0.3 (distant)
    Otherwise -> 0.5
    """
    if lang_a == lang_b:
        return 1.0

    pair = frozenset([lang_a, lang_b])

    # European related pairs
    european_close = [
        frozenset(["en", "fr"]),
        frozenset(["en", "es"]),
        frozenset(["fr", "es"]),
        frozenset(["fr", "de"]),
    ]
    if pair in european_close:
        return 0.7  # within [0.6, 0.8]

    # Distant pairs (zh involved)
    zh_distant = [
        frozenset(["en", "zh"]),
        frozenset(["fr", "zh"]),
        frozenset(["de", "zh"]),
        frozenset(["es", "zh"]),
    ]
    if pair in zh_distant:
        return 0.25  # within [0.2, 0.3]

    return 0.5


# ---------------------------------------------------------------------------
# Transfer evaluation
# ---------------------------------------------------------------------------

def evaluate_transfer(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    pairs: list[LanguagePair],
    cfg: CrosslingualConfig,
) -> list[TransferResult]:
    """Evaluate cross-lingual transfer for all language pairs.

    Collects source/target texts per language, computes scores, returns TransferResults.
    """
    from collections import defaultdict

    # Collect texts grouped by (src_lang, tgt_lang)
    grouped: dict[tuple[str, str], tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))

    for pair in pairs:
        key = (pair.source_lang, pair.target_lang)
        grouped[key][0].append(pair.source_text)
        grouped[key][1].append(pair.target_text)

    results: list[TransferResult] = []
    use_perplexity = cfg.metric in ("perplexity", "both")
    use_accuracy = cfg.metric in ("accuracy", "both")

    for (src_lang, tgt_lang), (src_texts, tgt_texts) in grouped.items():
        # Limit to n_eval_samples
        src_texts_limited = src_texts[: cfg.n_eval_samples]
        tgt_texts_limited = tgt_texts[: cfg.n_eval_samples]

        if use_perplexity:
            src_score = compute_perplexity(model, encode_fn, src_texts_limited, cfg.max_seq_len)
            tgt_score = compute_perplexity(model, encode_fn, tgt_texts_limited, cfg.max_seq_len)
        elif use_accuracy:
            src_score = compute_token_accuracy(model, encode_fn, src_texts_limited, cfg.max_seq_len)
            tgt_score = compute_token_accuracy(model, encode_fn, tgt_texts_limited, cfg.max_seq_len)
        else:
            src_score = compute_perplexity(model, encode_fn, src_texts_limited, cfg.max_seq_len)
            tgt_score = compute_perplexity(model, encode_fn, tgt_texts_limited, cfg.max_seq_len)

        transfer_gap = tgt_score - src_score
        relative_transfer = tgt_score / src_score if src_score != 0.0 else float("inf")

        results.append(TransferResult(
            source_lang=src_lang,
            target_lang=tgt_lang,
            source_score=src_score,
            target_score=tgt_score,
            transfer_gap=transfer_gap,
            relative_transfer=relative_transfer,
        ))

    return results


def compute_transfer_matrix(
    results: list[TransferResult],
    languages: list[str],
) -> dict[tuple[str, str], float]:
    """Build (lang_pair -> relative_transfer) matrix from results."""
    matrix: dict[tuple[str, str], float] = {}
    for r in results:
        matrix[(r.source_lang, r.target_lang)] = r.relative_transfer
    return matrix


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class CrosslingualEvaluator:
    """Full cross-lingual evaluation pipeline."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        cfg: CrosslingualConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.cfg = cfg

    def evaluate(self, pairs: list[LanguagePair] | None = None) -> dict[str, float]:
        """Run full evaluation. Generate pairs if not provided.

        Returns dict with keys:
          - mean_perplexity_all
          - best_transfer_pair
          - worst_transfer_pair
          - mean_transfer_gap
          - n_pairs_evaluated
        """
        if pairs is None:
            pairs = generate_synthetic_parallel_data(
                n_samples=self.cfg.n_eval_samples,
                languages=self.cfg.languages,
            )

        results = evaluate_transfer(self.model, self.encode_fn, pairs, self.cfg)

        if not results:
            return {
                "mean_perplexity_all": float("inf"),
                "best_transfer_pair": "",
                "worst_transfer_pair": "",
                "mean_transfer_gap": 0.0,
                "n_pairs_evaluated": 0.0,
            }

        all_scores = [r.source_score for r in results] + [r.target_score for r in results]
        mean_perplexity_all = sum(all_scores) / len(all_scores) if all_scores else float("inf")

        best_transfer_pair = self.find_best_transfer_pair(results)

        worst = min(results, key=lambda r: r.relative_transfer)
        worst_transfer_pair = f"{worst.source_lang}>{worst.target_lang}"

        mean_transfer_gap = sum(r.transfer_gap for r in results) / len(results)

        return {
            "mean_perplexity_all": mean_perplexity_all,
            "best_transfer_pair": best_transfer_pair,
            "worst_transfer_pair": worst_transfer_pair,
            "mean_transfer_gap": mean_transfer_gap,
            "n_pairs_evaluated": float(len(results)),
        }

    def find_best_transfer_pair(self, results: list[TransferResult]) -> str:
        """Return 'src->tgt' string for the pair with highest relative_transfer."""
        if not results:
            return ""
        best = max(results, key=lambda r: r.relative_transfer)
        return f"{best.source_lang}>{best.target_lang}"

    def compute_zero_shot_transfer_score(self, results: list[TransferResult]) -> float:
        """Mean relative_transfer across all pairs (excluding same-language pairs)."""
        cross_results = [r for r in results if r.source_lang != r.target_lang]
        if not cross_results:
            return 0.0
        return sum(r.relative_transfer for r in cross_results) / len(cross_results)
