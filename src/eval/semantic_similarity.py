"""Semantic similarity metrics for AureliusTransformer.

Implements embedding-based similarity, BERTScore-style token matching,
and lexical metrics — no external APIs required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SemanticSimConfig:
    """Configuration for semantic similarity evaluation."""
    pooling: str = "mean"    # "mean" | "last" | "max"
    normalize: bool = True
    batch_size: int = 8
    layer_idx: int = -1      # which layer to extract embeddings from (-1 = last)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    texts: list[str],
    cfg: SemanticSimConfig,
    max_seq_len: int = 128,
) -> Tensor:
    """Extract embeddings for a list of texts.

    Uses a forward hook on model.layers[cfg.layer_idx] to capture hidden states.
    Applies pooling per cfg.pooling and normalizes if cfg.normalize.

    Args:
        model: AureliusTransformer (or any module with .layers).
        encode_fn: Callable mapping a string to a list of token ids.
        texts: List of strings to encode.
        cfg: Semantic similarity configuration.
        max_seq_len: Maximum sequence length (truncation).

    Returns:
        (N, D) tensor of embeddings, one per text.
    """
    model.eval()
    device = next(model.parameters()).device

    # Resolve the target layer (supports negative indexing via Python list).
    layers = list(model.layers)
    target_layer = layers[cfg.layer_idx]

    all_embeddings: list[Tensor] = []

    for start in range(0, len(texts), cfg.batch_size):
        batch_texts = texts[start : start + cfg.batch_size]

        # Tokenize & pad
        token_lists = [encode_fn(t)[:max_seq_len] for t in batch_texts]
        max_len = max(len(ids) for ids in token_lists)
        if max_len == 0:
            max_len = 1

        pad_ids: list[list[int]] = []
        for ids in token_lists:
            padded = ids + [0] * (max_len - len(ids))
            pad_ids.append(padded)

        input_ids = torch.tensor(pad_ids, dtype=torch.long, device=device)  # (B, S)

        # Hook to capture output of the target layer.
        captured: list[Tensor] = []

        def _hook(module: nn.Module, inp: tuple, out: object) -> None:  # noqa: ANN001
            # TransformerBlock returns (hidden_state, kv_cache_tuple).
            if isinstance(out, tuple):
                captured.append(out[0].detach())
            else:
                captured.append(out.detach())  # type: ignore[union-attr]

        handle = target_layer.register_forward_hook(_hook)
        try:
            model(input_ids)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Forward hook did not capture any output.")

        hidden = captured[0]  # (B, S, D)

        # Build a padding mask: True where token is real (non-padding).
        lengths = torch.tensor([len(ids) for ids in token_lists], device=device)  # (B,)
        seq_len = hidden.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
        pad_mask = positions < lengths.unsqueeze(1)  # (B, S) — True for real tokens

        # Pool hidden states.
        if cfg.pooling == "mean":
            # Masked mean: sum over real tokens / count of real tokens.
            mask_f = pad_mask.unsqueeze(-1).float()  # (B, S, 1)
            emb = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        elif cfg.pooling == "last":
            # Last real token per sequence.
            last_idx = (lengths - 1).clamp(min=0)  # (B,)
            emb = hidden[torch.arange(hidden.shape[0], device=device), last_idx]
        elif cfg.pooling == "max":
            # Max over real tokens (set padded positions to -inf before max).
            mask_f = pad_mask.unsqueeze(-1).expand_as(hidden)
            hidden_masked = hidden.masked_fill(~mask_f, float("-inf"))
            emb = hidden_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {cfg.pooling!r}")

        if cfg.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# Cosine similarity utilities
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    embeddings_a: Tensor,  # (M, D)
    embeddings_b: Tensor,  # (N, D)
) -> Tensor:
    """Compute (M, N) cosine similarity matrix.

    Normalizes each row before computing the dot product.
    """
    a = F.normalize(embeddings_a, p=2, dim=-1)  # (M, D)
    b = F.normalize(embeddings_b, p=2, dim=-1)  # (N, D)
    return a @ b.T  # (M, N)


# ---------------------------------------------------------------------------
# BERTScore-style metrics
# ---------------------------------------------------------------------------

def bertscore_precision(
    candidate_embs: Tensor,   # (T_c, D)
    reference_embs: Tensor,   # (T_r, D)
) -> float:
    """BERTScore precision: mean over candidate tokens of max cosine sim to any reference."""
    sim_matrix = cosine_similarity_matrix(candidate_embs, reference_embs)  # (T_c, T_r)
    max_per_candidate = sim_matrix.max(dim=1).values  # (T_c,)
    return float(max_per_candidate.mean().item())


def bertscore_recall(
    candidate_embs: Tensor,   # (T_c, D)
    reference_embs: Tensor,   # (T_r, D)
) -> float:
    """BERTScore recall: mean over reference tokens of max cosine sim to any candidate."""
    sim_matrix = cosine_similarity_matrix(reference_embs, candidate_embs)  # (T_r, T_c)
    max_per_reference = sim_matrix.max(dim=1).values  # (T_r,)
    return float(max_per_reference.mean().item())


def bertscore_f1(p: float, r: float) -> float:
    """Harmonic mean of BERTScore precision and recall. Returns 0 if both are 0."""
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Approximate Word Mover's Distance
# ---------------------------------------------------------------------------

def compute_wmd_approx(
    words_a: list[str],
    words_b: list[str],
    embeddings: dict[str, Tensor],
) -> float:
    """Approximate Word Mover's Distance using greedy matching.

    For each word in words_a, finds the nearest word in words_b by cosine similarity
    of embeddings. Returns mean distance (1 - cosine_sim) of matched pairs.
    Returns 1.0 if any required embedding is missing.
    """
    if not words_a or not words_b:
        return 1.0

    # Collect embeddings for words_a; fall back to 1.0 on missing.
    embs_a: list[Tensor] = []
    for w in words_a:
        if w not in embeddings:
            return 1.0
        embs_a.append(embeddings[w])

    embs_b: list[Tensor] = []
    for w in words_b:
        if w not in embeddings:
            return 1.0
        embs_b.append(embeddings[w])

    mat_a = torch.stack(embs_a, dim=0)  # (|A|, D)
    mat_b = torch.stack(embs_b, dim=0)  # (|B|, D)

    sim_matrix = cosine_similarity_matrix(mat_a, mat_b)  # (|A|, |B|)
    max_sim = sim_matrix.max(dim=1).values  # (|A|,)
    distances = 1.0 - max_sim  # (|A|,)
    return float(distances.mean().item())


# ---------------------------------------------------------------------------
# Lexical n-gram overlap
# ---------------------------------------------------------------------------

def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """Compute character n-gram overlap F1 between two texts.

    Creates sets of character n-grams from each text, then computes F1.
    """
    def char_ngrams(text: str, ng: int) -> set:
        return {text[i : i + ng] for i in range(len(text) - ng + 1)}

    ngrams_a = char_ngrams(text_a, n)
    ngrams_b = char_ngrams(text_b, n)

    if not ngrams_a and not ngrams_b:
        return 1.0
    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b

    if not intersection:
        return 0.0

    precision = len(intersection) / len(ngrams_a)
    recall = len(intersection) / len(ngrams_b)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# High-level evaluator
# ---------------------------------------------------------------------------

class SemanticSimilarityEvaluator:
    """Compute multiple similarity metrics between texts.

    Uses an AureliusTransformer (or compatible module) to extract embeddings,
    and combines embedding-based, BERTScore-style, and lexical metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        cfg: SemanticSimConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.cfg = cfg

    def _get_embeddings(self, texts: list[str]) -> Tensor:
        """Extract normalized embeddings for a list of texts."""
        return extract_embeddings(self.model, self.encode_fn, texts, self.cfg)

    def compute_embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between embeddings of two texts."""
        embs = self._get_embeddings([text_a, text_b])  # (2, D)
        sim = cosine_similarity_matrix(embs[0:1], embs[1:2])  # (1, 1)
        return float(sim[0, 0].item())

    def compute_bertscore(
        self,
        candidates: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """BERTScore for a list of (candidate, reference) pairs.

        Uses whole-sequence embeddings as a single "token" for efficiency.
        Returns {"precision", "recall", "f1"} averaged over all pairs.
        """
        all_texts = candidates + references
        all_embs = self._get_embeddings(all_texts)  # (2N, D)
        n = len(candidates)
        cand_embs = all_embs[:n]   # (N, D)
        ref_embs = all_embs[n:]    # (N, D)

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for i in range(n):
            # Treat each sequence embedding as a single "token" -> (1, D)
            c_emb = cand_embs[i : i + 1]
            r_emb = ref_embs[i : i + 1]
            p = bertscore_precision(c_emb, r_emb)
            r = bertscore_recall(c_emb, r_emb)
            f = bertscore_f1(p, r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        return {
            "precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
            "recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
            "f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        }

    def batch_similarity(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> Tensor:
        """Compute pairwise cosine similarity for corresponding (a, b) pairs.

        Returns (N,) tensor.
        """
        all_texts = texts_a + texts_b
        all_embs = self._get_embeddings(all_texts)  # (2N, D)
        n = len(texts_a)
        embs_a = all_embs[:n]   # (N, D)
        embs_b = all_embs[n:]   # (N, D)
        # Element-wise cosine similarity: already normalized, so dot product.
        sims = (embs_a * embs_b).sum(dim=-1)  # (N,)
        return sims

    def evaluate_generation_quality(
        self,
        generated: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Comprehensive generation quality metrics.

        Returns:
            {
                "mean_cosine_sim": float,
                "bertscore_f1": float,
                "mean_ngram_overlap": float,
                "n_pairs": float,
            }
        """
        n = len(generated)

        # Cosine similarity
        sims = self.batch_similarity(generated, references)
        mean_cosine = float(sims.mean().item())

        # BERTScore
        bs = self.compute_bertscore(generated, references)

        # Lexical n-gram overlap (bigrams by default)
        ngram_scores = [ngram_overlap(g, r) for g, r in zip(generated, references)]
        mean_ngram = float(sum(ngram_scores) / len(ngram_scores)) if ngram_scores else 0.0

        return {
            "mean_cosine_sim": mean_cosine,
            "bertscore_f1": bs["f1"],
            "mean_ngram_overlap": mean_ngram,
            "n_pairs": float(n),
        }
